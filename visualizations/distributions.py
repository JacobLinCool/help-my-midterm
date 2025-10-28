"""
機率分佈視覺化
包含 Gaussian, Beta, Binomial, Poisson, Gamma 分佈及其共軛關係
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import norm, beta, binom, poisson, gamma, bernoulli
from scipy.special import comb
import os

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 創建輸出目錄
os.makedirs('output/images', exist_ok=True)


def visualize_gaussian():
    """視覺化高斯分佈"""
    print("生成高斯分佈視覺化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1D 高斯分佈 - 不同均值
    x = np.linspace(-5, 10, 1000)
    axes[0, 0].plot(x, norm.pdf(x, 0, 1), label='μ=0, σ²=1', linewidth=2)
    axes[0, 0].plot(x, norm.pdf(x, 2, 1), label='μ=2, σ²=1', linewidth=2)
    axes[0, 0].plot(x, norm.pdf(x, 5, 1), label='μ=5, σ²=1', linewidth=2)
    axes[0, 0].set_xlabel('x', fontsize=11)
    axes[0, 0].set_ylabel('Probability Density', fontsize=11)
    axes[0, 0].set_title('Gaussian Distribution - Different Means', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 1D 高斯分佈 - 不同變異數
    x = np.linspace(-10, 10, 1000)
    axes[0, 1].plot(x, norm.pdf(x, 0, 0.5), label='μ=0, σ²=0.25', linewidth=2)
    axes[0, 1].plot(x, norm.pdf(x, 0, 1), label='μ=0, σ²=1', linewidth=2)
    axes[0, 1].plot(x, norm.pdf(x, 0, 2), label='μ=0, σ²=4', linewidth=2)
    axes[0, 1].set_xlabel('x', fontsize=11)
    axes[0, 1].set_ylabel('Probability Density', fontsize=11)
    axes[0, 1].set_title('Gaussian Distribution - Different Variances', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 2D 高斯分佈
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    
    mean = [0, 0]
    cov = [[1, 0.5], [0.5, 1]]
    
    # 計算 2D 高斯
    cov_inv = np.linalg.inv(cov)
    cov_det = np.linalg.det(cov)
    norm_const = 1 / (2 * np.pi * np.sqrt(cov_det))
    
    diff = pos - mean
    exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=2)
    Z = norm_const * np.exp(exponent)
    
    contour = axes[1, 0].contourf(X, Y, Z, levels=15, cmap='viridis')
    axes[1, 0].contour(X, Y, Z, levels=15, colors='black', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=axes[1, 0])
    axes[1, 0].set_xlabel('x₁', fontsize=11)
    axes[1, 0].set_ylabel('x₂', fontsize=11)
    axes[1, 0].set_title('2D Gaussian Distribution (with correlation)', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Affine 性質示範
    np.random.seed(42)
    samples = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 1000)
    A = np.array([[2, 1], [0, 1]])
    b = np.array([1, 2])
    transformed = samples @ A.T + b
    
    axes[1, 1].scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, label='Original ~ N(0, I)')
    axes[1, 1].scatter(transformed[:, 0], transformed[:, 1], alpha=0.3, s=10, 
                      label='Transformed = Ax + b')
    axes[1, 1].set_xlabel('x₁', fontsize=11)
    axes[1, 1].set_ylabel('x₂', fontsize=11)
    axes[1, 1].set_title('Affine Property: y=Ax+b also Gaussian', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axis('equal')
    
    plt.tight_layout()
    plt.savefig('output/images/06_gaussian.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 高斯分佈視覺化完成")


def visualize_beta_binomial():
    """視覺化 Beta-Binomial 共軛關係"""
    print("生成 Beta-Binomial 共軛關係視覺化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Beta 分佈 - 不同參數
    mu = np.linspace(0, 1, 1000)
    axes[0, 0].plot(mu, beta.pdf(mu, 1, 1), label='a=1, b=1 (uniform)', linewidth=2)
    axes[0, 0].plot(mu, beta.pdf(mu, 2, 2), label='a=2, b=2', linewidth=2)
    axes[0, 0].plot(mu, beta.pdf(mu, 5, 2), label='a=5, b=2', linewidth=2)
    axes[0, 0].plot(mu, beta.pdf(mu, 2, 5), label='a=2, b=5', linewidth=2)
    axes[0, 0].set_xlabel('μ', fontsize=11)
    axes[0, 0].set_ylabel('Probability Density', fontsize=11)
    axes[0, 0].set_title('Beta Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Binomial 分佈
    N = 20
    k = np.arange(0, N+1)
    axes[0, 1].bar(k, binom.pmf(k, N, 0.3), alpha=0.7, label='μ=0.3, N=20', 
                   edgecolor='black', linewidth=0.5)
    axes[0, 1].bar(k, binom.pmf(k, N, 0.5), alpha=0.7, label='μ=0.5, N=20', 
                   edgecolor='black', linewidth=0.5)
    axes[0, 1].bar(k, binom.pmf(k, N, 0.7), alpha=0.7, label='μ=0.7, N=20', 
                   edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('k (number of successes)', fontsize=11)
    axes[0, 1].set_ylabel('Probability', fontsize=11)
    axes[0, 1].set_title('Binomial Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 共軛更新示範
    # Prior: Beta(2, 2)
    a_prior, b_prior = 2, 2
    
    # 觀察到數據: 10 次試驗, 7 次成功
    m = 7  # successes
    l = 3  # failures
    
    # Posterior: Beta(a+m, b+l)
    a_post = a_prior + m
    b_post = b_prior + l
    
    axes[1, 0].plot(mu, beta.pdf(mu, a_prior, b_prior), 'b-', 
                   label=f'Prior: Beta({a_prior}, {b_prior})', linewidth=2)
    axes[1, 0].axvline(m/(m+l), color='g', linestyle='--', 
                      label=f'Data: {m}/{m+l} successes', linewidth=2)
    axes[1, 0].plot(mu, beta.pdf(mu, a_post, b_post), 'r-', 
                   label=f'Posterior: Beta({a_post}, {b_post})', linewidth=2)
    axes[1, 0].set_xlabel('μ', fontsize=11)
    axes[1, 0].set_ylabel('Probability Density', fontsize=11)
    axes[1, 0].set_title('Beta-Binomial Conjugacy', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sequential Update
    true_mu = 0.6
    n_observations = 50
    
    a, b = 1, 1  # Start with uniform prior
    means = []
    
    np.random.seed(42)
    for i in range(n_observations):
        # 觀察一次
        observation = 1 if np.random.rand() < true_mu else 0
        # 更新
        if observation == 1:
            a += 1
        else:
            b += 1
        # 記錄後驗均值
        means.append(a / (a + b))
    
    axes[1, 1].plot(means, linewidth=2, label='Posterior Mean')
    axes[1, 1].axhline(true_mu, color='r', linestyle='--', 
                      label=f'True μ = {true_mu}', linewidth=2)
    axes[1, 1].set_xlabel('Number of Observations', fontsize=11)
    axes[1, 1].set_ylabel('Estimated μ', fontsize=11)
    axes[1, 1].set_title('Sequential Bayesian Update', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/images/07_beta_binomial.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Beta-Binomial 共軛關係視覺化完成")


def visualize_poisson_gamma():
    """視覺化 Poisson-Gamma 共軛關係"""
    print("生成 Poisson-Gamma 共軛關係視覺化...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Poisson 分佈
    k = np.arange(0, 20)
    axes[0, 0].bar(k, poisson.pmf(k, 2), alpha=0.7, label='λ=2', 
                   edgecolor='black', linewidth=0.5)
    axes[0, 0].bar(k, poisson.pmf(k, 5), alpha=0.7, label='λ=5', 
                   edgecolor='black', linewidth=0.5)
    axes[0, 0].bar(k, poisson.pmf(k, 10), alpha=0.7, label='λ=10', 
                   edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('k (number of events)', fontsize=11)
    axes[0, 0].set_ylabel('Probability', fontsize=11)
    axes[0, 0].set_title('Poisson Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Gamma 分佈
    lam = np.linspace(0.01, 15, 1000)
    axes[0, 1].plot(lam, gamma.pdf(lam, 2, scale=1/1), label='a=2, b=1', linewidth=2)
    axes[0, 1].plot(lam, gamma.pdf(lam, 5, scale=1/1), label='a=5, b=1', linewidth=2)
    axes[0, 1].plot(lam, gamma.pdf(lam, 5, scale=1/2), label='a=5, b=2', linewidth=2)
    axes[0, 1].set_xlabel('λ', fontsize=11)
    axes[0, 1].set_ylabel('Probability Density', fontsize=11)
    axes[0, 1].set_title('Gamma Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 共軛更新示範
    # Prior: Gamma(2, 1)
    a_prior, b_prior = 2, 1
    
    # 觀察到數據: [3, 5, 4, 6, 5]
    data = [3, 5, 4, 6, 5]
    N = len(data)
    sum_k = sum(data)
    
    # Posterior: Gamma(a + sum(k), b + N)
    a_post = a_prior + sum_k
    b_post = b_prior + N
    
    lam = np.linspace(0.01, 10, 1000)
    axes[1, 0].plot(lam, gamma.pdf(lam, a_prior, scale=1/b_prior), 'b-', 
                   label=f'Prior: Gamma({a_prior}, {b_prior})', linewidth=2)
    axes[1, 0].axvline(np.mean(data), color='g', linestyle='--', 
                      label=f'Data mean = {np.mean(data):.2f}', linewidth=2)
    axes[1, 0].plot(lam, gamma.pdf(lam, a_post, scale=1/b_post), 'r-', 
                   label=f'Posterior: Gamma({a_post}, {b_post})', linewidth=2)
    axes[1, 0].set_xlabel('λ', fontsize=11)
    axes[1, 0].set_ylabel('Probability Density', fontsize=11)
    axes[1, 0].set_title('Gamma-Poisson Conjugacy', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)
    
    # MLE vs MAP
    true_lambda = 5
    n_samples = [5, 10, 20, 50, 100]
    
    mle_estimates = []
    map_estimates = []
    
    np.random.seed(42)
    a, b = 2, 1  # Prior parameters
    
    for n in n_samples:
        # 生成數據
        data = np.random.poisson(true_lambda, n)
        
        # MLE
        mle = np.mean(data)
        mle_estimates.append(mle)
        
        # MAP (Gamma-Poisson)
        map_est = (a + np.sum(data) - 1) / (b + n)
        map_estimates.append(map_est)
    
    axes[1, 1].plot(n_samples, mle_estimates, 'bo-', label='MLE', linewidth=2, markersize=8)
    axes[1, 1].plot(n_samples, map_estimates, 'rs-', label='MAP (with prior)', linewidth=2, markersize=8)
    axes[1, 1].axhline(true_lambda, color='g', linestyle='--', 
                      label=f'True λ = {true_lambda}', linewidth=2)
    axes[1, 1].set_xlabel('Number of Samples', fontsize=11)
    axes[1, 1].set_ylabel('Estimated λ', fontsize=11)
    axes[1, 1].set_title('MLE vs MAP Estimation', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/images/08_poisson_gamma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Poisson-Gamma 共軛關係視覺化完成")


def visualize_all_distributions():
    """視覺化所有基本分佈的比較"""
    print("生成所有分佈比較視覺化...")
    
    fig = plt.figure(figsize=(16, 10))
    
    # Bernoulli
    ax1 = plt.subplot(3, 3, 1)
    x = [0, 1]
    ax1.bar(x, [0.3, 0.7], width=0.4, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_xlabel('x', fontsize=10)
    ax1.set_ylabel('Probability', fontsize=10)
    ax1.set_title('Bernoulli(μ=0.7)', fontsize=11, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Binomial
    ax2 = plt.subplot(3, 3, 2)
    k = np.arange(0, 21)
    ax2.bar(k, binom.pmf(k, 20, 0.6), alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('k', fontsize=10)
    ax2.set_ylabel('Probability', fontsize=10)
    ax2.set_title('Binomial(N=20, μ=0.6)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Beta
    ax3 = plt.subplot(3, 3, 3)
    mu = np.linspace(0, 1, 200)
    ax3.plot(mu, beta.pdf(mu, 3, 3), linewidth=2)
    ax3.fill_between(mu, beta.pdf(mu, 3, 3), alpha=0.3)
    ax3.set_xlabel('μ', fontsize=10)
    ax3.set_ylabel('Density', fontsize=10)
    ax3.set_title('Beta(a=3, b=3)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Poisson
    ax4 = plt.subplot(3, 3, 4)
    k = np.arange(0, 20)
    ax4.bar(k, poisson.pmf(k, 5), alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('k', fontsize=10)
    ax4.set_ylabel('Probability', fontsize=10)
    ax4.set_title('Poisson(λ=5)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Gamma
    ax5 = plt.subplot(3, 3, 5)
    lam = np.linspace(0.01, 15, 200)
    ax5.plot(lam, gamma.pdf(lam, 5, scale=1/1), linewidth=2)
    ax5.fill_between(lam, gamma.pdf(lam, 5, scale=1/1), alpha=0.3)
    ax5.set_xlabel('λ', fontsize=10)
    ax5.set_ylabel('Density', fontsize=10)
    ax5.set_title('Gamma(a=5, b=1)', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Gaussian 1D
    ax6 = plt.subplot(3, 3, 6)
    x = np.linspace(-5, 5, 200)
    ax6.plot(x, norm.pdf(x, 0, 1), linewidth=2)
    ax6.fill_between(x, norm.pdf(x, 0, 1), alpha=0.3)
    ax6.set_xlabel('x', fontsize=10)
    ax6.set_ylabel('Density', fontsize=10)
    ax6.set_title('Gaussian(μ=0, σ²=1)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 共軛對示意圖
    ax7 = plt.subplot(3, 3, 7)
    ax7.text(0.5, 0.7, 'Beta-Binomial\nConjugate Pair', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax7.text(0.5, 0.3, 'Prior: Beta\nLikelihood: Binomial\nPosterior: Beta', 
            ha='center', va='center', fontsize=10)
    ax7.axis('off')
    
    ax8 = plt.subplot(3, 3, 8)
    ax8.text(0.5, 0.7, 'Gamma-Poisson\nConjugate Pair', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    ax8.text(0.5, 0.3, 'Prior: Gamma\nLikelihood: Poisson\nPosterior: Gamma', 
            ha='center', va='center', fontsize=10)
    ax8.axis('off')
    
    ax9 = plt.subplot(3, 3, 9)
    ax9.text(0.5, 0.7, 'Gaussian-Gaussian\nConjugate Pair', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax9.text(0.5, 0.3, 'Prior: Gaussian\nLikelihood: Gaussian\nPosterior: Gaussian', 
            ha='center', va='center', fontsize=10)
    ax9.axis('off')
    
    plt.tight_layout()
    plt.savefig('output/images/09_all_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ 所有分佈比較視覺化完成")


def main():
    """運行所有機率分佈視覺化"""
    print("\n" + "="*60)
    print("開始生成機率分佈視覺化")
    print("="*60 + "\n")
    
    visualize_gaussian()
    visualize_beta_binomial()
    visualize_poisson_gamma()
    visualize_all_distributions()
    
    print("\n" + "="*60)
    print("✓ 所有機率分佈視覺化已完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
