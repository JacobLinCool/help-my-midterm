"""
EM 演算法視覺化
包含 Gaussian Mixture Model (GMM) 的 E-step 和 M-step
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture
import os

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出目錄
os.makedirs('output/images', exist_ok=True)


def plot_gaussian_ellipse(ax, mean, cov, color, alpha=0.5, label=None):
    """繪製二維高斯分佈的橢圓"""
    # 計算特徵值和特徵向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # 計算橢圓的角度
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # 繪製不同標準差的橢圓
    for n_std in [1, 2, 3]:
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])
        
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor='none', edgecolor=color,
                         linewidth=2, alpha=alpha * (4 - n_std) / 3,
                         label=label if n_std == 2 else None)
        ax.add_patch(ellipse)


def visualize_gmm_clustering():
    """視覺化 GMM 聚類過程"""
    print("生成 GMM 聚類視覺化...")
    
    # 生成混合高斯數據
    np.random.seed(42)
    n_samples = 300
    
    # 三個高斯分佈
    mean1 = [0, 0]
    cov1 = [[1, 0.5], [0.5, 1]]
    
    mean2 = [5, 5]
    cov2 = [[1.5, -0.3], [-0.3, 1]]
    
    mean3 = [2, 7]
    cov3 = [[0.8, 0.2], [0.2, 1.2]]
    
    # 生成數據
    X1 = np.random.multivariate_normal(mean1, cov1, n_samples // 3)
    X2 = np.random.multivariate_normal(mean2, cov2, n_samples // 3)
    X3 = np.random.multivariate_normal(mean3, cov3, n_samples // 3)
    X = np.vstack([X1, X2, X3])
    
    # 訓練 GMM
    gmm = GaussianMixture(n_components=3, random_state=42, max_iter=100)
    gmm.fit(X)
    labels = gmm.predict(X)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 原始數據
    axes[0, 0].scatter(X[:, 0], X[:, 1], alpha=0.5, s=30, c='gray', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Feature 1', fontsize=11)
    axes[0, 0].set_ylabel('Feature 2', fontsize=11)
    axes[0, 0].set_title('Original Data (unlabeled)', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # GMM 聚類結果
    colors = ['red', 'blue', 'green']
    for k in range(3):
        mask = labels == k
        axes[0, 1].scatter(X[mask, 0], X[mask, 1], 
                          c=colors[k], alpha=0.5, s=30, 
                          edgecolor='black', linewidth=0.5,
                          label=f'Cluster {k+1}')
        
        # 繪製高斯橢圓
        plot_gaussian_ellipse(axes[0, 1], gmm.means_[k], gmm.covariances_[k], 
                            colors[k], alpha=0.6)
        
        # 標記中心點
        axes[0, 1].plot(gmm.means_[k][0], gmm.means_[k][1], 
                       'k*', markersize=15, markeredgecolor='yellow', markeredgewidth=2)
    
    axes[0, 1].set_xlabel('Feature 1', fontsize=11)
    axes[0, 1].set_ylabel('Feature 2', fontsize=11)
    axes[0, 1].set_title('GMM Clustering Result', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Responsibilities (posterior probabilities)
    responsibilities = gmm.predict_proba(X)
    
    # 繪製每個點屬於第一個cluster的機率
    scatter = axes[1, 0].scatter(X[:, 0], X[:, 1], 
                                c=responsibilities[:, 0], 
                                cmap='RdYlBu_r', s=50, 
                                edgecolor='black', linewidth=0.5,
                                vmin=0, vmax=1)
    plt.colorbar(scatter, ax=axes[1, 0], label='P(Cluster 1 | x)')
    axes[1, 0].set_xlabel('Feature 1', fontsize=11)
    axes[1, 0].set_ylabel('Feature 2', fontsize=11)
    axes[1, 0].set_title('Responsibilities for Cluster 1', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Log-likelihood 收斂
    # 手動運行 GMM 以追蹤 log-likelihood
    log_likelihoods = []
    gmm_tracking = GaussianMixture(n_components=3, random_state=42, max_iter=1, warm_start=True)
    
    for i in range(50):
        gmm_tracking.fit(X)
        log_likelihoods.append(gmm_tracking.score(X) * len(X))
    
    axes[1, 1].plot(log_likelihoods, linewidth=2)
    axes[1, 1].set_xlabel('Iteration', fontsize=11)
    axes[1, 1].set_ylabel('Log-Likelihood', fontsize=11)
    axes[1, 1].set_title('EM Algorithm Convergence', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/images/14_gmm_clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ GMM 聚類視覺化完成")


def visualize_em_steps():
    """視覺化 EM 演算法的 E-step 和 M-step"""
    print("生成 EM 演算法步驟視覺化...")
    
    # 生成簡單的二維數據
    np.random.seed(42)
    X1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], 100)
    X2 = np.random.multivariate_normal([6, 6], [[0.5, 0], [0, 0.5]], 100)
    X = np.vstack([X1, X2])
    
    # 初始化參數（故意初始化得不好）
    means = np.array([[3, 3], [5, 5]], dtype=np.float64)
    covs = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]], dtype=np.float64)
    weights = np.array([0.5, 0.5], dtype=np.float64)
    
    # 手動實現 EM 的幾個迭代步驟
    iterations_to_show = [0, 1, 5, 20]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    for idx, n_iter in enumerate(iterations_to_show):
        # 運行 n_iter 次迭代
        current_means = means.copy()
        current_covs = covs.copy()
        current_weights = weights.copy()
        
        for iteration in range(n_iter):
            # E-step: 計算 responsibilities
            responsibilities = np.zeros((len(X), 2))
            
            for k in range(2):
                # 計算每個點在這個高斯分佈下的機率
                diff = X - current_means[k]
                cov_inv = np.linalg.inv(current_covs[k])
                exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
                norm_const = 1 / (2 * np.pi * np.sqrt(np.linalg.det(current_covs[k])))
                likelihood = norm_const * np.exp(exponent)
                responsibilities[:, k] = current_weights[k] * likelihood
            
            # 正規化
            responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
            
            # M-step: 更新參數
            Nk = responsibilities.sum(axis=0)
            
            for k in range(2):
                # 更新權重
                current_weights[k] = Nk[k] / len(X)
                
                # 更新均值
                current_means[k] = (responsibilities[:, k, np.newaxis] * X).sum(axis=0) / Nk[k]
                
                # 更新協方差（添加正規化項防止奇異矩陣）
                diff = X - current_means[k]
                current_covs[k] = (responsibilities[:, k, np.newaxis, np.newaxis] * 
                                  diff[:, :, np.newaxis] * diff[:, np.newaxis, :]).sum(axis=0) / Nk[k]
                # 添加小量到對角線防止奇異
                current_covs[k] += np.eye(2) * 1e-6
        
        # 繪製當前狀態
        # 根據 responsibilities 給點著色
        if n_iter > 0:
            responsibilities = np.zeros((len(X), 2))
            for k in range(2):
                diff = X - current_means[k]
                cov_inv = np.linalg.inv(current_covs[k])
                exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
                norm_const = 1 / (2 * np.pi * np.sqrt(np.linalg.det(current_covs[k])))
                likelihood = norm_const * np.exp(exponent)
                responsibilities[:, k] = current_weights[k] * likelihood
            responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
            
            # 根據最大 responsibility 著色
            labels = np.argmax(responsibilities, axis=1)
            colors_data = ['red' if l == 0 else 'blue' for l in labels]
        else:
            colors_data = 'gray'
        
        axes[idx].scatter(X[:, 0], X[:, 1], c=colors_data, alpha=0.5, s=30, 
                         edgecolor='black', linewidth=0.5)
        
        # 繪製高斯橢圓
        for k, color in enumerate(['red', 'blue']):
            plot_gaussian_ellipse(axes[idx], current_means[k], current_covs[k], 
                                color, alpha=0.7)
            axes[idx].plot(current_means[k][0], current_means[k][1], 
                          'k*', markersize=15, markeredgecolor='yellow', markeredgewidth=2)
        
        axes[idx].set_xlabel('Feature 1', fontsize=11)
        axes[idx].set_ylabel('Feature 2', fontsize=11)
        axes[idx].set_title(f'EM Algorithm - Iteration {n_iter}', 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim([0, 8])
        axes[idx].set_ylim([0, 8])
    
    plt.tight_layout()
    plt.savefig('output/images/15_em_steps.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ EM 演算法步驟視覺化完成")


def visualize_em_concept():
    """視覺化 EM 演算法的概念"""
    print("生成 EM 演算法概念視覺化...")
    
    fig = plt.figure(figsize=(14, 10))
    
    # 流程圖
    ax1 = plt.subplot(2, 2, 1)
    ax1.axis('off')
    
    # 繪製流程
    boxes = [
        (0.5, 0.9, 'Initialize\nparameters θ⁽⁰⁾', 'lightblue'),
        (0.5, 0.7, 'E-Step:\nCompute responsibilities\nγ(zₙₖ) = P(k|xₙ, θ⁽ᵗ⁾)', 'lightgreen'),
        (0.5, 0.5, 'M-Step:\nUpdate parameters\nθ⁽ᵗ⁺¹⁾ = argmax Q(θ|θ⁽ᵗ⁾)', 'lightyellow'),
        (0.5, 0.3, 'Converged?', 'lightcoral'),
        (0.5, 0.1, 'Done!', 'lightgray'),
    ]
    
    for x, y, text, color in boxes:
        ax1.add_patch(plt.Rectangle((x-0.15, y-0.06), 0.3, 0.1, 
                                   facecolor=color, edgecolor='black', linewidth=2))
        ax1.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # 箭頭
    ax1.annotate('', xy=(0.5, 0.8), xytext=(0.5, 0.84),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax1.annotate('', xy=(0.5, 0.6), xytext=(0.5, 0.64),
                arrowprops=dict(arrowstyle='->', lw=2))
    ax1.annotate('', xy=(0.5, 0.4), xytext=(0.5, 0.44),
                arrowprops=dict(arrowstyle='->', lw=2))
    
    # No: 回到 E-step
    ax1.annotate('No', xy=(0.5, 0.7), xytext=(0.7, 0.3),
                arrowprops=dict(arrowstyle='->', lw=2, color='red', linestyle='dashed'))
    
    # Yes: 完成
    ax1.annotate('Yes', xy=(0.5, 0.2), xytext=(0.5, 0.24),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title('EM Algorithm Flow', fontsize=12, fontweight='bold')
    
    # E-step 公式
    ax2 = plt.subplot(2, 2, 2)
    ax2.axis('off')
    
    formulas_e = [
        'E-Step (Expectation):',
        '',
        'Compute responsibilities:',
        'γ(zₙₖ) = P(k | xₙ, θ⁽ᵗ⁾)',
        '',
        '       πₖ N(xₙ | μₖ, Σₖ)',
        '= ─────────────────────────',
        '  Σⱼ πⱼ N(xₙ | μⱼ, Σⱼ)',
        '',
        'This tells us how much each',
        'data point "belongs" to',
        'each cluster.',
    ]
    
    y_pos = 0.95
    for formula in formulas_e:
        if formula.startswith('E-Step'):
            ax2.text(0.5, y_pos, formula, ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            ax2.text(0.5, y_pos, formula, ha='center', fontsize=10, family='monospace')
        y_pos -= 0.08
    
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_title('E-Step Details', fontsize=12, fontweight='bold')
    
    # M-step 公式
    ax3 = plt.subplot(2, 2, 3)
    ax3.axis('off')
    
    formulas_m = [
        'M-Step (Maximization):',
        '',
        'Update parameters to maximize',
        'expected log-likelihood:',
        '',
        'Nₖ = Σₙ γ(zₙₖ)',
        '',
        'πₖ⁽ᵗ⁺¹⁾ = Nₖ / N',
        '',
        'μₖ⁽ᵗ⁺¹⁾ = (1/Nₖ) Σₙ γ(zₙₖ) xₙ',
        '',
        'Σₖ⁽ᵗ⁺¹⁾ = (1/Nₖ) Σₙ γ(zₙₖ)',
        '         (xₙ - μₖ)(xₙ - μₖ)ᵀ',
    ]
    
    y_pos = 0.95
    for formula in formulas_m:
        if formula.startswith('M-Step'):
            ax3.text(0.5, y_pos, formula, ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        else:
            ax3.text(0.5, y_pos, formula, ha='center', fontsize=10, family='monospace')
        y_pos -= 0.075
    
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_title('M-Step Details', fontsize=12, fontweight='bold')
    
    # 關鍵概念
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    concepts = [
        'Key Concepts:',
        '',
        '• EM is used when we have',
        '  hidden/latent variables',
        '',
        '• E-step: Guess the hidden',
        '  variables (responsibilities)',
        '',
        '• M-step: Update parameters',
        '  assuming the guess is correct',
        '',
        '• Guaranteed to converge to',
        '  a local maximum',
        '',
        '• Result depends on',
        '  initialization',
    ]
    
    y_pos = 0.95
    for concept in concepts:
        if concept.startswith('Key'):
            ax4.text(0.1, y_pos, concept, fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        else:
            ax4.text(0.1, y_pos, concept, fontsize=10)
        y_pos -= 0.065
    
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.set_title('Important Notes', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('output/images/16_em_concept.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ EM 演算法概念視覺化完成")


def main():
    """運行所有 EM 演算法視覺化"""
    print("\n" + "="*60)
    print("開始生成 EM 演算法視覺化")
    print("="*60 + "\n")
    
    visualize_gmm_clustering()
    visualize_em_steps()
    visualize_em_concept()
    
    print("\n" + "="*60)
    print("✓ 所有 EM 演算法視覺化已完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
