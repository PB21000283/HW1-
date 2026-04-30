# EuroSAT 土地覆盖分类 — 手写 MLP (纯 NumPy)

不依赖任何深度学习框架,仅用 NumPy 实现三层 MLP,在 EuroSAT_RGB 数据集上完成 10 类土地覆盖分类。

- 验证集最佳准确率:**66.17%**
- 测试集准确率:**65.90%**

## 环境依赖

- Python >= 3.9
- numpy >= 1.24
- pillow >= 9.0
- matplotlib >= 3.6

安装:

```bash
pip install numpy pillow matplotlib
```

## 数据集

从 [https://github.com/phelber/EuroSAT](https://github.com/phelber/EuroSAT) 下载 `EuroSAT_RGB.zip`,解压后目录结构应为:

```
EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── HerbaceousVegetation/
├── Highway/
├── Industrial/
├── Pasture/
├── PermanentCrop/
├── Residential/
├── River/
└── SeaLake/
```

## 模型权重下载

训练好的最优权重 `best_model.npz` 见 Google Drive:

https://drive.google.com/drive/folders/1LrSJ-IBJQbkjXptu3kFJf7VmncE-sqEA?usp=sharing

下载后放到项目根目录即可使用 `--mode test` 直接评估。

## 运行方式

### 1. 网格搜索 + 用最优配置完整训练 100 epoch

```bash
python main.py --data_dir ./EuroSAT_RGB --mode search --epochs 100
```

### 2. 直接训练

```bash
python main.py --data_dir ./EuroSAT_RGB --mode train --epochs 100 \
    --lr 0.001 --hidden_dim 256 --dropout_p 0.1 \
    --weight_decay 1e-4 --activation relu
```

### 3. 用已训练好的权重做测试

```bash
python main.py --data_dir ./EuroSAT_RGB --mode test \
    --save_path best_model.npz \
    --hidden_dim 256 --dropout_p 0.1 --activation relu
```

## 输出文件

训练 / 测试完成后会在当前目录生成:

- `best_model.npz` — 验证集最佳模型权重
- `training_curves.png` — 训练 / 验证 Loss & Accuracy 曲线
- `w1_vis.png` — 第一层权重可视化(64 个神经元)
- `errors.png` — 测试集错例样本
- `grid_search_log.json` — 网格搜索日志(仅 `search` 模式)

## 项目结构

```
.
├── main.py              # 训练 / 搜索 / 测试主脚本
├── README.md
├── .gitignore
└── (运行后生成) best_model.npz, *.png, grid_search_log.json
```

## 模型结构

三层 MLP,输入 12288 维(64×64×3 展平),隐藏层 256 维,输出 10 类。

```
Input(12288) -> Linear -> ReLU -> Dropout(0.1)
             -> Linear -> ReLU -> Dropout(0.1)
             -> Linear -> Softmax(10)
```

- 优化器:SGD with Momentum=0.9,梯度范数裁剪 max_norm=5
- 损失:类别加权 Softmax + Cross-Entropy
- 学习率衰减:`lr_t = lr_0 / (1 + 0.01 * t)`
- 初始化:He 初始化
