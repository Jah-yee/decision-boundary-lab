"""
Decision Boundary Lab - Flask Backend
实时计算ML模型的决策边界，供前端可视化
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import io
import sys

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# ─── Dataset Generators ───────────────────────────────────────────────

def make_circles(n=300, noise=0.1, factor=0.7):
    X, y = datasets.make_circles(n_samples=n, noise=noise, factor=factor, random_state=42)
    return X, y

def make_moons(n=300, noise=0.15):
    X, y = datasets.make_moons(n_samples=n, noise=noise, random_state=42)
    return X, y

def make_blobs(n=300, centers=3, cluster_std=2.0):
    X, y = datasets.make_blobs(n_samples=n, centers=centers, cluster_std=cluster_std, random_state=42)
    return X, y

def make_classification(n=300, n_features=2, n_informative=2, n_redundant=0,
                        n_classes=2, n_clusters_per_class=1, class_sep=1.5):
    X, y = datasets.make_classification(
        n_samples=n, n_features=n_features, n_informative=n_informative,
        n_redundant=n_redundant, n_classes=n_classes,
        n_clusters_per_class=n_clusters_per_class, class_sep=class_sep,
        random_state=42
    )
    return X, y

def make_gaussian_quantiles(n=300, n_mean_shift=0.0):
    X, y = datasets.make_gaussian_quantiles(
        mean=[0, 0], cov=1.0, n_samples=n, n_classes=2,
        mean_shift=n_mean_shift, random_state=42
    )
    return X, y

DATASET_FACTORIES = {
    'circles': make_circles,
    'moons': make_moons,
    'blobs': make_blobs,
    'classification': make_classification,
    'gaussian_quantiles': make_gaussian_quantiles,
}


# ─── Model Factory ─────────────────────────────────────────────────────

def build_model(model_type, **kwargs):
    defaults = {
        'svm': {'C': 1.0, 'kernel': 'rbf', 'gamma': 'scale'},
        'logistic': {'C': 1.0, 'max_iter': 1000},
        'decision_tree': {'max_depth': 5, 'random_state': 42},
        'random_forest': {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
        'knn': {'n_neighbors': 5},
        'mlp': {'hidden_layer_sizes': (50,), 'max_iter': 500, 'random_state': 42},
    }
    opts = defaults.get(model_type, {}).copy()
    opts.update(kwargs)
    
    if model_type == 'svm':
        return SVC(**opts)
    elif model_type == 'logistic':
        return LogisticRegression(**opts)
    elif model_type == 'decision_tree':
        return DecisionTreeClassifier(**opts)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**opts)
    elif model_type == 'knn':
        return KNeighborsClassifier(**opts)
    elif model_type == 'mlp':
        return MLPClassifier(**opts)
    else:
        raise ValueError(f"Unknown model: {model_type}")


# ─── Decision Boundary Grid ────────────────────────────────────────────

def compute_decision_boundary(model, X_train, resolution=80):
    """计算决策边界网格"""
    # 标准化数据用于边界计算
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    
    # 找数据范围
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    
    # 生成网格
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    
    # 预测网格每个点的类别
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 也计算概率（如果模型支持）
    proba = None
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(grid_points)[:, 1]
            proba = proba.reshape(xx.shape)
        except Exception:
            pass
    
    # 计算决策函数距离（如果支持）
    decision = None
    if hasattr(model, 'decision_function'):
        try:
            decision = model.decision_function(grid_points)
            decision = decision.reshape(xx.shape)
        except Exception:
            pass
    
    return {
        'xx': xx.tolist(),
        'yy': yy.tolist(),
        'Z': Z.tolist(),
        'proba': proba.tolist() if proba is not None else None,
        'decision': decision.tolist() if decision is not None else None,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'x_min': float(x_min), 'x_max': float(x_max),
        'y_min': float(y_min), 'y_max': float(y_max),
    }


# ─── Core Training Endpoint ────────────────────────────────────────────

@app.route('/api/train', methods=['POST'])
def train():
    data = request.json
    model_type = data.get('model_type', 'svm')
    dataset_type = data.get('dataset_type', 'circles')
    dataset_params = data.get('dataset_params', {})
    model_params = data.get('model_params', {})
    resolution = int(data.get('resolution', 80))
    
    # 生成数据集
    factory = DATASET_FACTORIES.get(dataset_type, make_circles)
    X, y = factory(**dataset_params)
    
    # 分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练
    model = build_model(model_type, **model_params)
    model.fit(X_train_scaled, y_train)
    
    # 评估
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    # 决策边界
    boundary = compute_decision_boundary(model, X_train_scaled, resolution)
    
    return jsonify({
        'success': True,
        'train_accuracy': round(train_acc, 4),
        'test_accuracy': round(test_acc, 4),
        'boundary': boundary,
        'train_points': X_train_scaled.tolist(),
        'train_labels': y_train.tolist(),
        'test_points': X_test_scaled.tolist(),
        'test_labels': y_test.tolist(),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'classes': [0, 1] if len(np.unique(y)) == 2 else list(np.unique(y)),
    })


@app.route('/api/compare', methods=['POST'])
def compare():
    """对比多个模型在同一数据集上的表现"""
    data = request.json
    dataset_type = data.get('dataset_type', 'circles')
    dataset_params = data.get('dataset_params', {})
    model_types = data.get('model_types', ['svm', 'logistic', 'decision_tree'])
    model_params = data.get('model_params', {})
    resolution = int(data.get('resolution', 60))
    
    # 生成同一份数据
    factory = DATASET_FACTORIES.get(dataset_type, make_circles)
    X, y = factory(**dataset_params)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    for mt in model_types:
        mp = model_params.get(mt, {})
        model = build_model(mt, **mp)
        model.fit(X_train_scaled, y_train)
        y_pred_test = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred_test)
        boundary = compute_decision_boundary(model, X_train_scaled, resolution)
        results.append({
            'model_type': mt,
            'test_accuracy': round(acc, 4),
            'boundary': boundary,
            'train_points': X_train_scaled.tolist(),
            'train_labels': y_train.tolist(),
        })
    
    return jsonify({'success': True, 'results': results})


# ─── Frontend ───────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    print("🎯 Decision Boundary Lab starting on http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=True)
