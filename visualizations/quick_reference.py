"""
快速參考卡生成器
生成簡潔的公式參考卡片
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

# 設定中文字體
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs('output/images', exist_ok=True)


def create_quick_reference():
    """生成快速參考卡"""
    print("生成快速參考卡...")
    
    fig = plt.figure(figsize=(16, 20))
    
    # 分成多個區域
    sections = [
        {
            'title': 'LINEAR REGRESSION',
            'content': [
                'LSE: w = (Φᵀ Φ)⁻¹ Φᵀ t',
                'Ridge: w = (λI + Φᵀ Φ)⁻¹ Φᵀ t',
                'GD: w⁽ᵗ⁺¹⁾ = w⁽ᵗ⁾ - η ∇E(w⁽ᵗ⁾)',
                'Newton: w⁽ᵗ⁺¹⁾ = w⁽ᵗ⁾ - H⁻¹ ∇E(w⁽ᵗ⁾)',
            ],
            'color': 'lightblue'
        },
        {
            'title': 'PROBABILITY DISTRIBUTIONS',
            'content': [
                'Bernoulli: P(x|μ) = μˣ (1-μ)¹⁻ˣ',
                'Binomial: P(k|N,μ) = C(N,k) μᵏ (1-μ)ᴺ⁻ᵏ',
                'Beta: Beta(μ|a,b) ∝ μᵃ⁻¹ (1-μ)ᵇ⁻¹',
                'Poisson: P(k|λ) = e⁻ᵏ λᵏ / k!',
                'Gamma: Gam(λ|a,b) ∝ λᵃ⁻¹ e⁻ᵇᵏ',
                'Gaussian: N(x|μ,σ²) = 1/√(2πσ²) e⁻⁽ˣ⁻ᵘ⁾²/²ᵃ²',
            ],
            'color': 'lightgreen'
        },
        {
            'title': 'CONJUGATE PRIORS',
            'content': [
                'Beta-Binomial:',
                '  Prior: Beta(μ|a,b)',
                '  Likelihood: Bin(m|N,μ)',
                '  Posterior: Beta(μ|a+m, b+l)',
                '',
                'Gamma-Poisson:',
                '  Prior: Gam(λ|a,b)',
                '  Likelihood: Poisson(k|λ)',
                '  Posterior: Gam(λ|a+Σkᵢ, b+N)',
            ],
            'color': 'lightyellow'
        },
        {
            'title': 'MLE & MAP',
            'content': [
                'MLE: θ_ML = argmax P(D|θ)',
                'MAP: θ_MAP = argmax P(θ|D)',
                '           = argmax P(D|θ) P(θ)',
                '',
                'Gaussian MLE:',
                '  μ_ML = (1/N) Σ xₙ',
                '  σ²_ML = (1/N) Σ (xₙ - μ_ML)²',
            ],
            'color': 'lightcoral'
        },
        {
            'title': 'CLASSIFICATION',
            'content': [
                'Naive Bayes:',
                '  ŷ = argmax P(Cₖ) Π P(xᵢ|Cₖ)',
                '',
                'Logistic Regression:',
                '  σ(a) = 1 / (1 + e⁻ᵃ)',
                '  P(C₁|x) = σ(wᵀx)',
                '  ∇E(w) = Φᵀ(y - t)',
                '  H = Φᵀ R Φ, R = diag(yₙ(1-yₙ))',
            ],
            'color': 'lavender'
        },
        {
            'title': 'EM ALGORITHM',
            'content': [
                'E-Step: γ(zₙₖ) = πₖ N(xₙ|μₖ,Σₖ) / Σⱼ πⱼ N(xₙ|μⱼ,Σⱼ)',
                '',
                'M-Step:',
                '  Nₖ = Σₙ γ(zₙₖ)',
                '  πₖ = Nₖ / N',
                '  μₖ = (1/Nₖ) Σₙ γ(zₙₖ) xₙ',
                '  Σₖ = (1/Nₖ) Σₙ γ(zₙₖ) (xₙ-μₖ)(xₙ-μₖ)ᵀ',
            ],
            'color': 'lightpink'
        },
        {
            'title': 'PERFORMANCE METRICS',
            'content': [
                'Accuracy = (TP + TN) / Total',
                'Precision = TP / (TP + FP)',
                'Recall = TP / (TP + FN)',
                'Specificity = TN / (TN + FP)',
                'F1 = 2 × Precision × Recall / (Precision + Recall)',
            ],
            'color': 'peachpuff'
        },
        {
            'title': 'KEY CONCEPTS',
            'content': [
                'Bias-Variance Trade-off:',
                '  High Bias → Underfitting',
                '  High Variance → Overfitting',
                '',
                'Regularization:',
                '  L1 (Lasso) → Sparse weights',
                '  L2 (Ridge) → Small weights',
                '',
                'Bayes Theorem:',
                '  Posterior ∝ Likelihood × Prior',
            ],
            'color': 'wheat'
        },
    ]
    
    # 每行2個section
    n_sections = len(sections)
    n_cols = 2
    n_rows = (n_sections + n_cols - 1) // n_cols
    
    for idx, section in enumerate(sections):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        ax.axis('off')
        
        # 標題
        ax.text(0.5, 0.95, section['title'], 
               ha='center', va='top', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=section['color'], 
                        edgecolor='black', linewidth=2))
        
        # 內容
        y_pos = 0.85
        for line in section['content']:
            if line.strip():
                fontsize = 10
                if line.startswith(' '):
                    fontsize = 9
                    x_pos = 0.15
                else:
                    x_pos = 0.1
                
                ax.text(x_pos, y_pos, line, 
                       ha='left', va='top', fontsize=fontsize,
                       family='monospace')
            y_pos -= 0.08
        
        # 邊框
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.9,
                                  fill=False, edgecolor='black', linewidth=1.5))
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    plt.suptitle('Machine Learning Quick Reference Card', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('output/images/00_quick_reference.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ 快速參考卡已生成: output/images/00_quick_reference.png")


def main():
    """運行快速參考卡生成"""
    print("\n" + "="*60)
    print("開始生成快速參考卡")
    print("="*60 + "\n")
    
    create_quick_reference()
    
    print("\n" + "="*60)
    print("✓ 快速參考卡已完成！")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
