"""
回歸分析視覺化
包含 LSE, Ridge, Lasso, GD, Newton's Method, Bias-Variance Trade-off
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非互動式後端
from matplotlib.animation import FuncAnimation
import os

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出目錄
os.makedirs('output/images', exist_ok=True)
os.makedirs('output/videos', exist_ok=True)


def generate_data(n_samples=100, noise=0.5, seed=42):
    """生成回歸數據"""
    np.random.seed(seed)
    X = np.linspace(0, 10, n_samples)
    y_true = 2 * X + 1
    y = y_true + np.random.normal(0, noise, n_samples)
    return X, y, y_true


def visualize_lse():
    """視覺化最小平方誤差 (LSE)"""
    print("生成 LSE 視覺化...")
    X, y, y_true = generate_data()
    
    # 轉換為矩陣形式
    X_matrix = np.column_stack([np.ones_like(X), X])
    
    # LSE 解析解
    w_lse = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y
    y_pred = X_matrix @ w_lse
    
    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖：數據和擬合線
    axes[0].scatter(X, y, alpha=0.6, label='Data points')
    axes[0].plot(X, y_true, 'g--', label='True function', linewidth=2)
    axes[0].plot(X, y_pred, 'r-', label='LSE fit', linewidth=2)
    
    # 顯示幾個誤差線
    for i in range(0, len(X), 10):
        axes[0].plot([X[i], X[i]], [y[i], y_pred[i]], 'k-', alpha=0.3, linewidth=1)
    
    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('y', fontsize=12)
    axes[0].set_title('Least Squares Error (LSE)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 右圖：誤差分佈
    errors = y - y_pred
    axes[1].hist(errors, bins=20, alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='r', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Error', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Error Distribution (should be ~ Gaussian)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/images/01_lse.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ LSE 視覺化完成")


def visualize_regularization():
    """視覺化 L1 (Lasso) 和 L2 (Ridge) 正規化"""
    print("生成正規化視覺化...")
    
    # 生成過度擬合的數據
    np.random.seed(42)
    X = np.linspace(0, 2, 20)
    y_true = 2 * X + 1
    y = y_true + np.random.normal(0, 0.5, len(X))
    
    # 使用高次多項式特徵
    def polynomial_features(X, degree):
        return np.column_stack([X**i for i in range(degree + 1)])
    
    degree = 10
    X_poly = polynomial_features(X, degree)
    X_test = np.linspace(0, 2, 100)
    X_test_poly = polynomial_features(X_test, degree)
    
    # 不同的 lambda 值
    lambdas = [0, 0.01, 0.1, 1.0]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, lam in enumerate(lambdas):
        # Ridge (L2) 回歸
        w_ridge = np.linalg.inv(X_poly.T @ X_poly + lam * np.eye(degree + 1)) @ X_poly.T @ y
        y_pred_ridge = X_test_poly @ w_ridge
        
        axes[idx].scatter(X, y, alpha=0.6, s=50, label='Data')
        axes[idx].plot(X_test, 2 * X_test + 1, 'g--', label='True function', linewidth=2)
        axes[idx].plot(X_test, y_pred_ridge, 'r-', label=f'Ridge (λ={lam})', linewidth=2)
        
        axes[idx].set_xlabel('X', fontsize=11)
        axes[idx].set_ylabel('y', fontsize=11)
        axes[idx].set_title(f'Ridge Regularization (λ={lam})', fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([-2, 6])
    
    plt.tight_layout()
    plt.savefig('output/images/02_regularization.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 正規化視覺化完成")


def visualize_gradient_descent():
    """視覺化梯度下降過程"""
    print("生成梯度下降視覺化...")
    
    # 簡單的二次函數
    def f(w):
        return (w[0] - 3)**2 + (w[1] - 2)**2 + 5
    
    def grad_f(w):
        return np.array([2 * (w[0] - 3), 2 * (w[1] - 2)])
    
    # 梯度下降
    w = np.array([0.0, 0.0])
    learning_rate = 0.1
    history = [w.copy()]
    
    for _ in range(30):
        w = w - learning_rate * grad_f(w)
        history.append(w.copy())
    
    history = np.array(history)
    
    # 繪製等高線圖和梯度下降路徑
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    w1 = np.linspace(-1, 5, 100)
    w2 = np.linspace(-1, 4, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Z = (W1 - 3)**2 + (W2 - 2)**2 + 5
    
    contour = ax.contour(W1, W2, Z, levels=20, cmap='viridis', alpha=0.6)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # 梯度下降路徑
    ax.plot(history[:, 0], history[:, 1], 'ro-', linewidth=2, markersize=6, 
            label='Gradient Descent Path', markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(history[0, 0], history[0, 1], 'go', markersize=12, 
            label='Start', markeredgecolor='black', markeredgewidth=1)
    ax.plot(history[-1, 0], history[-1, 1], 'r*', markersize=15, 
            label='End', markeredgecolor='black', markeredgewidth=1)
    ax.plot(3, 2, 'b*', markersize=15, label='True Minimum', 
            markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('w₁', fontsize=12)
    ax.set_ylabel('w₂', fontsize=12)
    ax.set_title('Gradient Descent Optimization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/images/03_gradient_descent.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 梯度下降視覺化完成")


def visualize_bias_variance():
    """視覺化 Bias-Variance Trade-off"""
    print("生成 Bias-Variance Trade-off 視覺化...")
    
    np.random.seed(42)
    
    # 真實函數
    def true_function(x):
        return np.sin(2 * np.pi * x)
    
    # 生成訓練數據
    n_samples = 20
    X_train = np.random.uniform(0, 1, n_samples)
    y_train = true_function(X_train) + np.random.normal(0, 0.1, n_samples)
    
    # 測試數據
    X_test = np.linspace(0, 1, 100)
    y_test = true_function(X_test)
    
    # 不同複雜度的模型
    degrees = [1, 3, 15]
    titles = ['High Bias (Underfitting)', 'Good Fit', 'High Variance (Overfitting)']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (degree, title) in enumerate(zip(degrees, titles)):
        # 多項式特徵
        X_train_poly = np.column_stack([X_train**i for i in range(degree + 1)])
        X_test_poly = np.column_stack([X_test**i for i in range(degree + 1)])
        
        # 擬合模型
        w = np.linalg.lstsq(X_train_poly, y_train, rcond=None)[0]
        y_pred = X_test_poly @ w
        
        # 繪圖
        axes[idx].scatter(X_train, y_train, alpha=0.7, s=50, label='Training data', 
                         edgecolor='black', linewidth=0.5)
        axes[idx].plot(X_test, y_test, 'g--', label='True function', linewidth=2)
        axes[idx].plot(X_test, y_pred, 'r-', label=f'Polynomial (degree={degree})', linewidth=2)
        
        axes[idx].set_xlabel('X', fontsize=11)
        axes[idx].set_ylabel('y', fontsize=11)
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_ylim([-2, 2])
    
    plt.tight_layout()
    plt.savefig('output/images/04_bias_variance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Bias-Variance Trade-off 視覺化完成")


def visualize_newton_vs_gd():
    """視覺化 Newton's Method vs Gradient Descent"""
    print("生成 Newton vs GD 視覺化...")
    
    # 定義函數和導數
    def f(x):
        return 0.5 * x**2 - 3 * x + 5
    
    def grad_f(x):
        return x - 3
    
    def hess_f(x):
        return 1
    
    # Gradient Descent
    x_gd = -2.0
    lr = 0.1
    history_gd = [x_gd]
    
    for _ in range(15):
        x_gd = x_gd - lr * grad_f(x_gd)
        history_gd.append(x_gd)
    
    # Newton's Method
    x_newton = -2.0
    history_newton = [x_newton]
    
    for _ in range(5):
        x_newton = x_newton - grad_f(x_newton) / hess_f(x_newton)
        history_newton.append(x_newton)
    
    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左圖：函數和收斂路徑
    x = np.linspace(-3, 5, 200)
    y = f(x)
    
    axes[0].plot(x, y, 'b-', linewidth=2, label='f(x) = 0.5x² - 3x + 5')
    axes[0].plot(history_gd, [f(xi) for xi in history_gd], 'ro-', 
                markersize=6, label='Gradient Descent', markeredgecolor='black', markeredgewidth=0.5)
    axes[0].plot(history_newton, [f(xi) for xi in history_newton], 'gs-', 
                markersize=8, label="Newton's Method", markeredgecolor='black', markeredgewidth=0.5)
    axes[0].plot(3, f(3), 'k*', markersize=15, label='Minimum', 
                markeredgecolor='yellow', markeredgewidth=1)
    
    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('f(x)', fontsize=12)
    axes[0].set_title('Comparison: Newton vs Gradient Descent', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 右圖：收斂速度比較
    iterations = range(max(len(history_gd), len(history_newton)))
    error_gd = [abs(x - 3) for x in history_gd]
    error_newton = [abs(x - 3) for x in history_newton]
    
    axes[1].semilogy(error_gd, 'ro-', label='Gradient Descent', markersize=6, linewidth=2)
    axes[1].semilogy(error_newton, 'gs-', label="Newton's Method", markersize=8, linewidth=2)
    
    axes[1].set_xlabel('Iteration', fontsize=12)
    axes[1].set_ylabel('Error (log scale)', fontsize=12)
    axes[1].set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/images/05_newton_vs_gd.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Newton vs GD 視覺化完成")


def main():
    """運行所有回歸視覺化"""
    print("\n" + "="*60)
    print("開始生成回歸分析視覺化")
    print("="*60 + "\n")
    
    visualize_lse()
    visualize_regularization()
    visualize_gradient_descent()
    visualize_bias_variance()
    visualize_newton_vs_gd()
    
    print("\n" + "="*60)
    print("✓ 所有回歸視覺化已完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
