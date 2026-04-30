"""
EuroSAT 土地覆盖分类 —— 手写 MLP (纯 NumPy)
======================================================
用法:
    python mlp_eurosat.py --data_dir ./EuroSAT_RGB --mode train
    python mlp_eurosat.py --data_dir ./EuroSAT_RGB --mode search
    python mlp_eurosat.py --data_dir ./EuroSAT_RGB --mode test
"""

import os
import argparse
import itertools
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
'''nohup python -u main.py --data_dir ./EuroSAT_RGB --mode search --epochs 100 \
    > train.log 2>&1 &'''

# ============================================================
# 1. 数据加载与预处理
# ============================================================
IMG_SIZE = 64  # EuroSAT 原始就是 64x64


def load_data(data_dir):
    """读取 EuroSAT_RGB 数据集,所有图像 resize 到 64x64x3。"""
    img_list, label_list = [], []
    class_names = [c for c in sorted(os.listdir(data_dir))
                   if os.path.isdir(os.path.join(data_dir, c))]
    assert len(class_names) > 0, f"在 {data_dir} 下没找到类别目录"

    for label_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        n_before = len(img_list)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                img_list.append(np.array(img))
                label_list.append(label_idx)
            except Exception as e:
                print(f"[WARN] 跳过 {img_path}: {e}")
        print(f"  {class_name}: {len(img_list) - n_before} 张")

    X = np.array(img_list, dtype=np.uint8)   # (N, 64, 64, 3)
    y = np.array(label_list, dtype=np.int64)
    print(f"总样本: {len(X)} | 类别数: {len(class_names)}")
    return X, y, class_names


def preprocess(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    """展平 + 标准化 + 划分 train/val/test。"""
    rng = np.random.default_rng(seed)
    X_flat = X.reshape(X.shape[0], -1).astype(np.float64) / 255.0

    indices = rng.permutation(len(X_flat))
    X_flat, y = X_flat[indices], y[indices]

    n = len(X_flat)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    X_train, y_train = X_flat[:n_train], y[:n_train]
    X_val,   y_val   = X_flat[n_train:n_train + n_val], y[n_train:n_train + n_val]
    X_test,  y_test  = X_flat[n_train + n_val:],        y[n_train + n_val:]

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - X_mean) / X_std
    X_val   = (X_val   - X_mean) / X_std
    X_test  = (X_test  - X_mean) / X_std

    return X_train, y_train, X_val, y_val, X_test, y_test, X_mean, X_std


def compute_class_weights(y, num_classes):
    counts = np.bincount(y, minlength=num_classes)
    w = len(y) / (num_classes * counts + 1e-8)
    return w / w.sum() * num_classes


class DataLoader:
    def __init__(self, X, y, batch_size=64, shuffle=True, seed=None):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = len(X)
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(idx)
        for s in range(0, self.n, self.batch_size):
            b = idx[s:s + self.batch_size]
            yield self.X[b], self.y[b]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size


# ============================================================
# 2. 组件:激活/Dropout/线性层/损失
# ============================================================
class ReLU:
    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, g):
        return g * (self.x > 0)


class Sigmoid:
    def forward(self, x):
        x = np.clip(x, -500, 500)
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, g):
        return g * self.out * (1 - self.out)


class Tanh:
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out

    def backward(self, g):
        return g * (1 - self.out ** 2)


class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.training = True

    def forward(self, x):
        if self.training and self.p > 0:
            self.mask = (np.random.rand(*x.shape) > self.p).astype(x.dtype)
            return x * self.mask / (1 - self.p)
        return x

    def backward(self, g):
        if self.training and self.p > 0:
            return g * self.mask / (1 - self.p)
        return g

    def train(self): self.training = True
    def eval(self):  self.training = False


class Linear:
    """He 初始化的全连接层。无权重裁剪、无 forward 裁剪。"""
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = np.zeros((1, out_dim))
        self.grad_W = None
        self.grad_b = None

    def forward(self, x):
        self.x = x
        return x @ self.W + self.b

    def backward(self, grad_out):
        self.grad_W = self.x.T @ grad_out
        self.grad_b = np.sum(grad_out, axis=0, keepdims=True)
        return grad_out @ self.W.T


class WeightedCrossEntropyLoss:
    """数值稳定的 softmax + cross entropy,可选类别加权。"""
    def __init__(self, class_weights=None):
        self.class_weights = class_weights

    def forward(self, logits, labels):
        self.batch_size = logits.shape[0]
        self.labels = labels

        logits_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(logits_stable)
        self.probs = exp / np.sum(exp, axis=1, keepdims=True)

        correct = np.clip(self.probs[np.arange(self.batch_size), labels], 1e-12, 1.0)

        if self.class_weights is not None:
            sw = self.class_weights[labels]
            self.sample_weights = sw
            loss = -np.sum(sw * np.log(correct)) / (np.sum(sw) + 1e-8)
        else:
            self.sample_weights = None
            loss = -np.mean(np.log(correct))
        return loss

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(self.batch_size), self.labels] -= 1
        if self.class_weights is not None:
            grad = grad * self.sample_weights.reshape(-1, 1)
            grad = grad / (np.sum(self.sample_weights) + 1e-8)
        else:
            grad = grad / self.batch_size
        return grad


# ============================================================
# 3. SGD 优化器(带 momentum 与全局梯度范数裁剪)
# ============================================================
class SGD:
    def __init__(self, layers, lr=0.01, momentum=0.9, weight_decay=0.0,
                 max_grad_norm=5.0):
        self.layers = [l for l in layers if hasattr(l, 'W')]
        self.lr = lr
        self.initial_lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.vel = [{'W': np.zeros_like(l.W), 'b': np.zeros_like(l.b)}
                    for l in self.layers]

    def clip_grad_norm_(self):
        """全局 L2 范数裁剪。"""
        if self.max_grad_norm is None or self.max_grad_norm <= 0:
            return
        sq = 0.0
        for l in self.layers:
            sq += np.sum(l.grad_W ** 2) + np.sum(l.grad_b ** 2)
        norm = np.sqrt(sq)
        if norm > self.max_grad_norm:
            scale = self.max_grad_norm / (norm + 1e-8)
            for l in self.layers:
                l.grad_W *= scale
                l.grad_b *= scale

    def step(self):
        self.clip_grad_norm_()
        for l, v in zip(self.layers, self.vel):
            g_W = l.grad_W + self.weight_decay * l.W
            g_b = l.grad_b
            v['W'] = self.momentum * v['W'] - self.lr * g_W
            v['b'] = self.momentum * v['b'] - self.lr * g_b
            l.W += v['W']
            l.b += v['b']

    def lr_decay(self, epoch, decay_rate=0.01):
        self.lr = self.initial_lr / (1 + decay_rate * epoch)
        return self.lr


# ============================================================
# 4. MLP 模型(Linear -> Act -> Dropout -> Linear -> Act -> Dropout -> Linear)
# ============================================================
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim,
                 activation='relu', dropout_p=0.3, class_weights=None):
        self.linear1 = Linear(input_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, hidden_dim)
        self.linear3 = Linear(hidden_dim, output_dim)

        act_map = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}
        assert activation in act_map
        self.act1 = act_map[activation]()
        self.act2 = act_map[activation]()

        self.dropout1 = Dropout(p=dropout_p)
        self.dropout2 = Dropout(p=dropout_p)

        self.loss_fn = WeightedCrossEntropyLoss(class_weights=class_weights)
        self.layers = [self.linear1, self.act1, self.dropout1,
                       self.linear2, self.act2, self.dropout2,
                       self.linear3]

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.act1.forward(x)
        x = self.dropout1.forward(x)
        x = self.linear2.forward(x)
        x = self.act2.forward(x)
        x = self.dropout2.forward(x)
        x = self.linear3.forward(x)
        return x

    def backward(self, grad):
        grad = self.linear3.backward(grad)
        grad = self.dropout2.backward(grad)
        grad = self.act2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.dropout1.backward(grad)
        grad = self.act1.backward(grad)
        grad = self.linear1.backward(grad)

    def compute_loss(self, logits, labels):
        return self.loss_fn.forward(logits, labels)

    def compute_loss_grad(self):
        return self.loss_fn.backward()

    def predict(self, x):
        return np.argmax(self.forward(x), axis=1)

    def train_mode(self):
        self.dropout1.train(); self.dropout2.train()

    def eval_mode(self):
        self.dropout1.eval(); self.dropout2.eval()

    def save(self, path, X_mean=None, X_std=None, class_names=None, meta=None):
        d = dict(
            W1=self.linear1.W, b1=self.linear1.b,
            W2=self.linear2.W, b2=self.linear2.b,
            W3=self.linear3.W, b3=self.linear3.b,
        )
        if X_mean is not None: d['X_mean'] = X_mean
        if X_std  is not None: d['X_std']  = X_std
        if class_names is not None: d['class_names'] = np.array(class_names)
        if meta is not None: d['meta'] = np.array(json.dumps(meta))
        np.savez(path, **d)

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.linear1.W, self.linear1.b = data['W1'], data['b1']
        self.linear2.W, self.linear2.b = data['W2'], data['b2']
        self.linear3.W, self.linear3.b = data['W3'], data['b3']
        return data


# ============================================================
# 5. 训练 / 验证 / 测试
# ============================================================
def evaluate(model, loader):
    """返回 (avg_loss, accuracy)。"""
    model.eval_mode()
    losses, correct, total = [], 0, 0
    for X_b, y_b in loader:
        logits = model.forward(X_b)
        losses.append(model.compute_loss(logits, y_b))
        pred = np.argmax(logits, axis=1)
        correct += np.sum(pred == y_b)
        total += len(y_b)
    return float(np.mean(losses)), correct / total


def train(model, train_loader, val_loader, optimizer,
          epochs=50, lr_decay_rate=0.01, save_path='best_model.npz',
          X_mean=None, X_std=None, class_names=None, verbose=True):

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': [], 'lr': []}

    for ep in range(epochs):
        model.train_mode()
        tl, tc, tn = 0.0, 0, 0
        n_batches = 0
        for X_b, y_b in train_loader:
            logits = model.forward(X_b)
            loss = model.compute_loss(logits, y_b)
            if np.isnan(loss):
                if verbose: print(f"[WARN] NaN loss at epoch {ep}, skip batch")
                continue
            tl += loss
            n_batches += 1
            tc += np.sum(np.argmax(logits, axis=1) == y_b)
            tn += len(y_b)

            grad = model.compute_loss_grad()
            model.backward(grad)
            optimizer.step()

        train_loss = tl / max(n_batches, 1)
        train_acc  = tc / max(tn, 1)

        val_loss, val_acc = evaluate(model, val_loader)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(save_path, X_mean=X_mean, X_std=X_std,
                       class_names=class_names)
            if verbose:
                print(f"  ★ save best at epoch {ep+1}, val_acc={val_acc:.4f}")

        cur_lr = optimizer.lr_decay(ep + 1, lr_decay_rate)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(cur_lr)

        if verbose:
            print(f"Epoch {ep+1:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                  f"lr={cur_lr:.5f}")

    if verbose:
        print(f"\n训练完成, best val acc = {best_val_acc:.4f}")
    return history, best_val_acc


def test(model, test_loader, class_names, save_path='best_model.npz'):
    model.load(save_path)
    model.eval_mode()

    preds, labels, raw_X = [], [], []
    for X_b, y_b in test_loader:
        p = model.predict(X_b)
        preds.extend(p.tolist())
        labels.extend(y_b.tolist())
        raw_X.append(X_b)
    preds = np.array(preds)
    labels = np.array(labels)
    raw_X = np.concatenate(raw_X, axis=0)

    acc = np.mean(preds == labels)
    print(f"\n测试集准确率: {acc:.4f}")

    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        confusion[t, p] += 1

    print("\n混淆矩阵:")
    print(confusion)

    print("\n每类准确率:")
    for i, name in enumerate(class_names):
        denom = confusion[i].sum()
        ca = confusion[i, i] / denom if denom > 0 else 0.0
        print(f"  {name:20s}: {ca:.4f}")

    return acc, confusion, preds, labels, raw_X


# ============================================================
# 6. 可视化
# ============================================================
def plot_history(history, save_path='training_curves.png'):
    """画 train/val loss 曲线 + val acc 曲线。"""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history['train_loss'], label='train_loss')
    axes[0].plot(epochs, history['val_loss'],   label='val_loss')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Training / Validation Loss')
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], label='train_acc')
    axes[1].plot(epochs, history['val_acc'],   label='val_acc')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training / Validation Accuracy')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] 训练曲线已保存: {save_path}")


def visualize_first_layer_weights(model, n_show=64, save_path='w1_vis.png'):
    """
    可视化第一层权重。W1 形状 (64*64*3, hidden_dim)。
    每列 reshape 成 (64, 64, 3),归一化到 [0, 1],以网格展示。
    """
    W = model.linear1.W   # (D_in, hidden_dim)
    D_in, H = W.shape
    assert D_in == IMG_SIZE * IMG_SIZE * 3, f"W1 形状不是 64*64*3 = {IMG_SIZE*IMG_SIZE*3}"
    n_show = min(n_show, H)

    cols = int(np.ceil(np.sqrt(n_show)))
    rows = int(np.ceil(n_show / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.3, rows * 1.3))
    axes = np.array(axes).reshape(-1)

    for i in range(rows * cols):
        ax = axes[i]
        if i < n_show:
            w = W[:, i].reshape(IMG_SIZE, IMG_SIZE, 3)
            w = (w - w.min()) / (w.max() - w.min() + 1e-8)   # 归一化到 [0,1]
            ax.imshow(w)
            ax.set_title(f'#{i}', fontsize=6)
        ax.axis('off')

    plt.suptitle(f'First-layer weights (first {n_show} neurons)', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] 第一层权重已保存: {save_path}")


def error_analysis(preds, labels, raw_X_standardized, X_mean, X_std,
                   class_names, n_show=16, save_path='errors.png'):
    """
    从测试集里挑 n_show 个错分样本,反标准化后以原始图像展示。
    raw_X_standardized 是已经标准化的测试数据,需要乘回 X_std 加回 X_mean。
    """
    wrong_idx = np.where(preds != labels)[0]
    if len(wrong_idx) == 0:
        print("[INFO] 没有错分样本,跳过错例分析。")
        return
    pick = np.random.default_rng(0).choice(
        wrong_idx, size=min(n_show, len(wrong_idx)), replace=False
    )

    cols = int(np.ceil(np.sqrt(len(pick))))
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes):
        if i < len(pick):
            idx = pick[i]
            x = raw_X_standardized[idx] * X_std + X_mean   # 反标准化
            x = np.clip(x, 0, 1).reshape(IMG_SIZE, IMG_SIZE, 3)
            ax.imshow(x)
            ax.set_title(f"T:{class_names[labels[idx]]}\nP:{class_names[preds[idx]]}",
                         fontsize=7, color='red')
        ax.axis('off')

    plt.suptitle(f'Misclassified samples (n={len(pick)})', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOT] 错例分析已保存: {save_path}")


# ============================================================
# 7. 网格搜索
# ============================================================
def grid_search(X_train, y_train, X_val, y_val, X_mean, X_std,
                num_classes, class_weights, class_names,
                search_space=None, epochs=20,
                log_path='grid_search_log.json'):
    """
    简单网格搜索:超参数组合 × epochs 固定短轮次,用 val acc 评估,返回最优配置。
    """
    if search_space is None:
        search_space = {
            'lr':           [0.001, 0.005, 0.01],
            'hidden_dim':   [128, 256],
            'weight_decay': [0.0, 1e-4, 1e-3],
            'dropout_p':    [0.1, 0.3],
            'activation':   ['relu', 'tanh'],
        }

    keys = list(search_space.keys())
    combos = list(itertools.product(*[search_space[k] for k in keys]))
    print(f"[SEARCH] 组合数: {len(combos)}")

    results = []
    best_cfg, best_val = None, -1.0

    for i, values in enumerate(combos):
        cfg = dict(zip(keys, values))
        print(f"\n[SEARCH {i+1}/{len(combos)}] {cfg}")

        np.random.seed(42)   # 每次搜索固定种子,公平比较
        model = MLP(
            input_dim=X_train.shape[1], hidden_dim=cfg['hidden_dim'],
            output_dim=num_classes, activation=cfg['activation'],
            dropout_p=cfg['dropout_p'], class_weights=class_weights
        )
        optimizer = SGD(layers=model.layers, lr=cfg['lr'],
                        momentum=0.9, weight_decay=cfg['weight_decay'])
        train_loader = DataLoader(X_train, y_train, batch_size=64, shuffle=True, seed=42)
        val_loader   = DataLoader(X_val,   y_val,   batch_size=64, shuffle=False)

        _, best_val_acc = train(
            model, train_loader, val_loader, optimizer,
            epochs=epochs, lr_decay_rate=0.01,
            save_path=f'tmp_search_model.npz',
            X_mean=X_mean, X_std=X_std, class_names=class_names,
            verbose=False,
        )
        print(f"  → val_acc = {best_val_acc:.4f}")
        results.append({**cfg, 'val_acc': best_val_acc})

        if best_val_acc > best_val:
            best_val = best_val_acc
            best_cfg = cfg

    with open(log_path, 'w') as f:
        json.dump({'results': results, 'best_cfg': best_cfg, 'best_val': best_val},
                  f, indent=2)
    print(f"\n[SEARCH] 最优配置: {best_cfg}  val_acc={best_val:.4f}")
    print(f"[SEARCH] 日志: {log_path}")
    if os.path.exists('tmp_search_model.npz'):
        os.remove('tmp_search_model.npz')
    return best_cfg, results


# ============================================================
# 8. 主流程
# ============================================================
def build_everything(args):
    """加载 + 预处理 + 计算类权重,返回常用对象。"""
    X, y, class_names = load_data(args.data_dir)
    (X_train, y_train, X_val, y_val, X_test, y_test,
     X_mean, X_std) = preprocess(X, y, seed=args.seed)
    print(f"train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")
    class_weights = compute_class_weights(y_train, len(class_names))
    return (X_train, y_train, X_val, y_val, X_test, y_test,
            X_mean, X_std, class_names, class_weights)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./EuroSAT_RGB')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'search', 'test'])
    parser.add_argument('--save_path', type=str, default='best_model.npz')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--dropout_p', type=float, default=0.3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    (X_train, y_train, X_val, y_val, X_test, y_test,
     X_mean, X_std, class_names, class_weights) = build_everything(args)

    train_loader = DataLoader(X_train, y_train, batch_size=args.batch_size,
                              shuffle=True,  seed=args.seed)
    val_loader   = DataLoader(X_val,   y_val,   batch_size=args.batch_size,
                              shuffle=False)
    test_loader  = DataLoader(X_test,  y_test,  batch_size=args.batch_size,
                              shuffle=False)

    if args.mode == 'search':
        best_cfg, _ = grid_search(
            X_train, y_train, X_val, y_val, X_mean, X_std,
            num_classes=len(class_names),
            class_weights=class_weights, class_names=class_names,
            epochs=min(args.epochs, 20),
        )
        print("\n用最优配置重新完整训练...")
        args.lr           = best_cfg['lr']
        args.hidden_dim   = best_cfg['hidden_dim']
        args.weight_decay = best_cfg['weight_decay']
        args.dropout_p    = best_cfg['dropout_p']
        args.activation   = best_cfg['activation']
        args.mode = 'train'

    if args.mode == 'train':
        print(f"\n配置: lr={args.lr}  hidden={args.hidden_dim}  "
              f"act={args.activation}  dropout={args.dropout_p}  "
              f"wd={args.weight_decay}")
        model = MLP(
            input_dim=X_train.shape[1], hidden_dim=args.hidden_dim,
            output_dim=len(class_names), activation=args.activation,
            dropout_p=args.dropout_p, class_weights=class_weights,
        )
        optimizer = SGD(layers=model.layers, lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)

        history, _ = train(
            model, train_loader, val_loader, optimizer,
            epochs=args.epochs, save_path=args.save_path,
            X_mean=X_mean, X_std=X_std, class_names=class_names,
        )

        plot_history(history, save_path='training_curves.png')

        # 可视化 + 错例分析
        model.load(args.save_path)
        visualize_first_layer_weights(model, n_show=64,
                                       save_path='w1_vis.png')

        print("\n[TEST] 在测试集上评估最优模型...")
        _, _, preds, labels, raw_X = test(
            model, test_loader, class_names, save_path=args.save_path)
        error_analysis(preds, labels, raw_X, X_mean, X_std,
                       class_names, n_show=16, save_path='errors.png')

    elif args.mode == 'test':
        model = MLP(
            input_dim=X_train.shape[1], hidden_dim=args.hidden_dim,
            output_dim=len(class_names), activation=args.activation,
            dropout_p=args.dropout_p, class_weights=class_weights,
        )
        _, _, preds, labels, raw_X = test(
            model, test_loader, class_names, save_path=args.save_path)
        visualize_first_layer_weights(model, n_show=64,
                                       save_path='w1_vis.png')
        error_analysis(preds, labels, raw_X, X_mean, X_std,
                       class_names, n_show=16, save_path='errors.png')


if __name__ == '__main__':
    main()