# EuroSAT 土地覆盖分类 — 手写 MLP (纯 NumPy)

不依赖任何深度学习框架,仅用 NumPy 实现三层 MLP,
在 EuroSAT_RGB 数据集上完成 10 类土地覆盖分类。

## 环境依赖
- Python >= 3.9
- numpy >= 1.24
- pillow >= 9.0
- matplotlib >= 3.6

安装:
\`\`\`bash
pip install numpy pillow matplotlib
\`\`\`

## 数据集
从 https://github.com/phelber/EuroSAT 下载 EuroSAT_RGB.zip,
解压后目录结构应为:
\`\`\`
EuroSAT_RGB/
├── AnnualCrop/
├── Forest/
├── ...
└── SeaLake/
\`\`\`

## 模型权重下载
训练好的最优权重见 Google Drive:
https://github.com/<your-username>/eurosat-mlp-numpy

## 运行方式

### 1. 网格搜索 + 用最优配置完整训练 100 epoch
\`\`\`bash
python main.py --data_dir ./EuroSAT_RGB --mode search --epochs 100
\`\`\`

### 2. 直接训练
\`\`\`bash
python main.py --data_dir ./EuroSAT_RGB --mode train --epochs 100 \
    --lr 0.001 --hidden_dim 256 --dropout_p 0.1 \
    --weight_decay 1e-4 --activation relu
\`\`\`

### 3. 用已训练好的权重做测试
\`\`\`bash
python main.py --data_dir ./EuroSAT_RGB --mode test \
    --save_path best_model.npz \
    --hidden_dim 256 --dropout_p 0.1 --activation relu
\`\`\`

## 结果
- 验证集最佳准确率:66.17%
- 测试集准确率:65.90%