"""
分類方法視覺化
包含 Naive Bayes, Logistic Regression, Confusion Matrix
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.datasets import make_classification, make_blobs
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import os

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出目錄
os.makedirs('output/images', exist_ok=True)


def sigmoid(x):
    """Sigmoid 函數"""
    return 1 / (1 + np.exp(-x))


def visualize_sigmoid():
    """視覺化 Sigmoid 函數及其性質"""
    print("生成 Sigmoid 函數視覺化...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Sigmoid 函數
    x = np.linspace(-10, 10, 1000)
    y = sigmoid(x)
    
    axes[0].plot(x, y, 'b-', linewidth=2)
    axes[0].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0].axhline(1, color='k', linestyle='--', alpha=0.3)
    axes[0].axhline(0.5, color='r', linestyle='--', alpha=0.5, label='Decision boundary')
    axes[0].axvline(0, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('σ(x)', fontsize=12)
    axes[0].set_title('Sigmoid Function σ(x) = 1/(1+e⁻ˣ)', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Sigmoid 導數
    y_derivative = y * (1 - y)
    axes[1].plot(x, y_derivative, 'g-', linewidth=2)
    axes[1].set_xlabel('x', fontsize=12)
    axes[1].set_ylabel("σ'(x)", fontsize=12)
    axes[1].set_title("Sigmoid Derivative: σ'(x) = σ(x)(1-σ(x))", fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # 與其他激活函數比較
    axes[2].plot(x, y, 'b-', linewidth=2, label='Sigmoid')
    axes[2].plot(x, np.tanh(x), 'r-', linewidth=2, label='Tanh')
    axes[2].plot(x, np.maximum(0, x), 'g-', linewidth=2, label='ReLU')
    axes[2].set_xlabel('x', fontsize=12)
    axes[2].set_ylabel('Activation', fontsize=12)
    axes[2].set_title('Activation Functions Comparison', fontsize=12, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-1.5, 3])
    
    plt.tight_layout()
    plt.savefig('output/images/10_sigmoid.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Sigmoid 函數視覺化完成")


def visualize_logistic_regression():
    """視覺化 Logistic Regression"""
    print("生成 Logistic Regression 視覺化...")
    
    # 生成二分類數據
    np.random.seed(42)
    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                               n_informative=2, n_clusters_per_class=1,
                               class_sep=1.5, random_state=42)
    
    # 訓練 Logistic Regression
    model = LogisticRegression()
    model.fit(X, y)
    
    # 創建網格用於繪製決策邊界
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖：決策邊界和機率
    contour = axes[0].contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    axes[0].contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    # 繪製數據點
    scatter = axes[0].scatter(X[y==0, 0], X[y==0, 1], c='blue', 
                             edgecolor='black', s=50, alpha=0.8, label='Class 0')
    axes[0].scatter(X[y==1, 0], X[y==1, 1], c='red', 
                   edgecolor='black', s=50, alpha=0.8, label='Class 1')
    
    plt.colorbar(contour, ax=axes[0], label='P(Class 1)')
    axes[0].set_xlabel('Feature 1', fontsize=11)
    axes[0].set_ylabel('Feature 2', fontsize=11)
    axes[0].set_title('Logistic Regression Decision Boundary', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 右圖：梯度下降過程（簡化示範）
    # 手動實現梯度下降
    def logistic_loss(w, X, y):
        """計算 Logistic Regression 的損失"""
        z = X @ w
        return -np.mean(y * np.log(sigmoid(z) + 1e-10) + 
                       (1 - y) * np.log(1 - sigmoid(z) + 1e-10))
    
    # 添加偏置項
    X_bias = np.c_[np.ones(X.shape[0]), X]
    
    # 初始化權重
    w = np.zeros(X_bias.shape[1])
    learning_rate = 0.01
    n_iterations = 100
    losses = []
    
    for i in range(n_iterations):
        # 計算預測
        z = X_bias @ w
        y_pred = sigmoid(z)
        
        # 計算梯度
        gradient = X_bias.T @ (y_pred - y) / len(y)
        
        # 更新權重
        w = w - learning_rate * gradient
        
        # 記錄損失
        loss = logistic_loss(w, X_bias, y)
        losses.append(loss)
    
    axes[1].plot(losses, linewidth=2)
    axes[1].set_xlabel('Iteration', fontsize=11)
    axes[1].set_ylabel('Cross-Entropy Loss', fontsize=11)
    axes[1].set_title('Training: Loss vs Iterations', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/images/11_logistic_regression.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Logistic Regression 視覺化完成")


def visualize_naive_bayes():
    """視覺化 Naive Bayes 分類器"""
    print("生成 Naive Bayes 視覺化...")
    
    # 生成數據
    np.random.seed(42)
    X, y = make_blobs(n_samples=300, centers=2, n_features=2, 
                      cluster_std=1.5, random_state=42)
    
    # 訓練 Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X, y)
    
    # 訓練 Logistic Regression 用於比較
    lr_model = LogisticRegression()
    lr_model.fit(X, y)
    
    # 創建網格
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Naive Bayes
    Z_nb = nb_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_nb = Z_nb.reshape(xx.shape)
    
    axes[0].contourf(xx, yy, Z_nb, alpha=0.4, cmap='RdYlBu')
    axes[0].scatter(X[y==0, 0], X[y==0, 1], c='blue', 
                   edgecolor='black', s=50, alpha=0.8, label='Class 0')
    axes[0].scatter(X[y==1, 0], X[y==1, 1], c='red', 
                   edgecolor='black', s=50, alpha=0.8, label='Class 1')
    
    # 繪製每個類別的高斯分佈中心
    for i in range(2):
        mean = nb_model.theta_[i]
        axes[0].plot(mean[0], mean[1], 'k*', markersize=20, 
                    markeredgecolor='yellow', markeredgewidth=2)
    
    axes[0].set_xlabel('Feature 1', fontsize=11)
    axes[0].set_ylabel('Feature 2', fontsize=11)
    axes[0].set_title('Naive Bayes Classifier', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Logistic Regression (比較)
    Z_lr = lr_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_lr = Z_lr.reshape(xx.shape)
    
    axes[1].contourf(xx, yy, Z_lr, alpha=0.4, cmap='RdYlBu')
    axes[1].scatter(X[y==0, 0], X[y==0, 1], c='blue', 
                   edgecolor='black', s=50, alpha=0.8, label='Class 0')
    axes[1].scatter(X[y==1, 0], X[y==1, 1], c='red', 
                   edgecolor='black', s=50, alpha=0.8, label='Class 1')
    
    axes[1].set_xlabel('Feature 1', fontsize=11)
    axes[1].set_ylabel('Feature 2', fontsize=11)
    axes[1].set_title('Logistic Regression (for comparison)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/images/12_naive_bayes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Naive Bayes 視覺化完成")


def visualize_confusion_matrix():
    """視覺化混淆矩陣和性能指標"""
    print("生成混淆矩陣視覺化...")
    
    # 生成測試數據
    np.random.seed(42)
    X, y_true = make_classification(n_samples=200, n_features=10, 
                                     n_informative=5, n_redundant=2,
                                     random_state=42)
    
    # 訓練模型
    model = LogisticRegression()
    model.fit(X[:150], y_true[:150])
    y_pred = model.predict(X[150:])
    y_true_test = y_true[150:]
    
    # 計算混淆矩陣
    cm = confusion_matrix(y_true_test, y_pred)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 混淆矩陣
    im = axes[0, 0].imshow(cm, cmap='Blues', alpha=0.8)
    axes[0, 0].set_xticks([0, 1])
    axes[0, 0].set_yticks([0, 1])
    axes[0, 0].set_xticklabels(['Predicted 0', 'Predicted 1'])
    axes[0, 0].set_yticklabels(['Actual 0', 'Actual 1'])
    
    # 在每個格子中顯示數字
    for i in range(2):
        for j in range(2):
            text = axes[0, 0].text(j, i, cm[i, j],
                                  ha="center", va="center", 
                                  color="white" if cm[i, j] > cm.max()/2 else "black",
                                  fontsize=24, fontweight='bold')
    
    axes[0, 0].set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=axes[0, 0])
    
    # 標註 TP, FP, TN, FN
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            axes[0, 0].text(j, i + 0.3, labels[i][j],
                           ha="center", va="center",
                           color="red", fontsize=14, fontweight='bold')
    
    # 計算指標
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 指標視覺化
    metrics = ['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']
    values = [accuracy, precision, recall, specificity, f1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    bars = axes[0, 1].barh(metrics, values, color=colors, alpha=0.8, 
                          edgecolor='black', linewidth=1.5)
    axes[0, 1].set_xlim([0, 1])
    axes[0, 1].set_xlabel('Score', fontsize=11)
    axes[0, 1].set_title('Performance Metrics', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # 在柱狀圖上顯示數值
    for i, (bar, value) in enumerate(zip(bars, values)):
        axes[0, 1].text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                       f'{value:.3f}',
                       va='center', fontsize=10, fontweight='bold')
    
    # 公式說明
    formulas = [
        'Accuracy = (TP + TN) / (TP + TN + FP + FN)',
        'Precision = TP / (TP + FP)',
        'Recall (Sensitivity) = TP / (TP + FN)',
        'Specificity = TN / (TN + FP)',
        'F1-Score = 2 × (Precision × Recall) / (Precision + Recall)'
    ]
    
    axes[1, 0].axis('off')
    y_pos = 0.9
    for formula in formulas:
        axes[1, 0].text(0.1, y_pos, formula, fontsize=10, 
                       family='monospace',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        y_pos -= 0.18
    axes[1, 0].set_title('Metric Formulas', fontsize=12, fontweight='bold')
    
    # ROC 曲線概念圖（簡化）
    # 獲取預測機率
    y_proba = model.predict_proba(X[150:])[:, 1]
    
    # 手動計算不同閾值下的 TPR 和 FPR
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        cm_threshold = confusion_matrix(y_true_test, y_pred_threshold)
        
        if cm_threshold.shape == (2, 2):
            tn, fp, fn, tp = cm_threshold.ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            tpr = 0
            fpr = 0
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    axes[1, 1].plot(fprs, tprs, linewidth=2, label='ROC Curve')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    axes[1, 1].fill_between(fprs, tprs, alpha=0.3)
    axes[1, 1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1, 1].set_ylabel('True Positive Rate (Recall)', fontsize=11)
    axes[1, 1].set_title('ROC Curve', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, 1])
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('output/images/13_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 混淆矩陣視覺化完成")


def main():
    """運行所有分類視覺化"""
    print("\n" + "="*60)
    print("開始生成分類方法視覺化")
    print("="*60 + "\n")
    
    visualize_sigmoid()
    visualize_logistic_regression()
    visualize_naive_bayes()
    visualize_confusion_matrix()
    
    print("\n" + "="*60)
    print("✓ 所有分類視覺化已完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
