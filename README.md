# 🎯 Decision Boundary Lab

> **交互式 ML 决策边界可视化实验台** — 支持 6 种模型、5 种数据集、实时调参、模型对比

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
[![View on GitHub](https://img.shields.io/badge/GitHub-View%20Repo-blueviolet.svg)](https://github.com)

## ✨ Features

- 🧠 **6 种 ML 模型**: SVM (RBF/Linear), Logistic Regression, Decision Tree, Random Forest, KNN, MLP
- 📊 **5 种数据集**: Two Circles, Two Moons, Blobs, Linear Separation, Gaussian Quantiles
- 🎛️ **实时超参数调节**: 每个模型都有专属参数面板
- ⚡ **一键模型对比**: 同时跑完所有模型并排序准确率
- 🎨 **像素级 Canvas 可视化**: 决策边界热力图 + 数据点叠加
- 📱 **响应式布局**: 支持桌面和移动端

## 🎨 Demo

![Decision Boundary Screenshot](docs/screenshot.png)

> 上图：SVM (RBF kernel) 在 Two Moons 数据集上的决策边界。蓝色/粉色区域表示模型预测的两类，热线条为决策边界圆点为训练集，圆环为测试集。

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/decision-boundary-lab.git
cd decision-boundary-lab

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
python app.py
# 🎯 Decision Boundary Lab starting on http://localhost:5050

# 4. Open browser
open http://localhost:5050
```

**Docker:**
```bash
docker build -t boundary-lab .
docker run -p 5050:5050 boundary-lab
```

## 🧪 How It Works

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /api/train` | JSON | 训练单个模型并返回决策边界数据 |
| `POST /api/compare` | JSON | 同时训练多个模型并返回对比结果 |

### Train Request Example

```python
import requests
import matplotlib.pyplot as plt

resp = requests.post('http://localhost:5050/api/train', json={
    "model_type": "svm",
    "dataset_type": "moons",
    "dataset_params": {"n": 300, "noise": 0.15},
    "model_params": {"C": 1.0, "gamma": 0.5},
    "resolution": 80
})
data = resp.json()
print(f"Train Accuracy: {data['train_accuracy']:.2%}")
print(f"Test Accuracy:  {data['test_accuracy']:.2%}")
```

### Supported Datasets

| Name | Description | Key Param |
|------|-------------|-----------|
| `circles` | Two concentric circles | `noise`, `factor` |
| `moons` | Two interleaving half-circles | `noise` |
| `blobs` | 3 Gaussian clusters | `centers`, `cluster_std` |
| `classification` | Random n-class problem | `class_sep`, `n_classes` |
| `gaussian_quantiles` | Gaussian quantile boundaries | `n_mean_shift` |

### Supported Models & Key Hyperparameters

| Model | Parameters |
|-------|-----------|
| **SVM (RBF)** | `C` (regularization), `gamma` (RBF width) |
| **Logistic Regression** | `C` (inverse regularization) |
| **Decision Tree** | `max_depth` (tree depth) |
| **Random Forest** | `max_depth`, `n_estimators` (number of trees) |
| **KNN** | `n_neighbors` (K) |
| **MLP** | `hidden_layer_sizes` |

## 🗂️ Project Structure

```
decision-boundary-lab/
├── app.py              # Flask backend — ML training + boundary computation
├── requirements.txt    # Dependencies
├── README.md           # This file
├── Procfile            # For Render deployment
├── .flaskenv           # Flask config
└── templates/
    └── index.html      # Single-page frontend (HTML + CSS + Vanilla JS)
```

## 📐 Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Frontend    │────▶│   Flask API      │────▶│   scikit-learn  │
│  (Canvas)    │◀────│  /api/train       │◀────│   Models        │
│              │     │  /api/compare     │     │                 │
│  index.html  │     └──────────────────┘     └─────────────────┘
└──────────────┘
```

- **Backend**: Flask serves the HTML and handles ML training requests
- **ML Engine**: scikit-learn for model training and `model.predict()` on grid points
- **Frontend**: Vanilla JS + HTML5 Canvas for pixel-level boundary rendering
- **No React/Vue needed**: pure browser-side rendering for simplicity

## 🌐 Deploy to Render (Free)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

```bash
# Render automatically detects Flask and requirements.txt
# Set start command:  python app.py
# Or use Procfile:
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

## 📚 Educational Value

This tool demonstrates:

- **How different ML models capture different decision boundaries**
- **Underfitting vs. overfitting** — try high/low depth, high/low C
- **KNN behavior** — small k is wiggly, large k is smooth
- **SVM RBF flexibility** — can separate complex shapes
- **Linear vs. non-linear models** — LR/Linear SVM vs. RBF/Tree/MLP

## 📝 License

MIT © 2026
