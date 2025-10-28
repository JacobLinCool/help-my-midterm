# 📘 Help My Midterm - 機器學習期中考筆記

完整的機器學習核心概念筆記，配合 Python 視覺化幫助你從零開始理解所有內容！

## 🎯 專案簡介

這個專案提供：
- 📝 完整的機器學習核心概念筆記（中文）
- 📊 豐富的視覺化圖片，幫助理解抽象概念
- 🐍 使用 Python 和 uv 管理所有依賴
- 🎨 自動生成所有學習輔助圖片

## 📚 內容涵蓋

### 回歸分析
- Least Squares Error (LSE)
- L1/L2 正規化 (Lasso/Ridge)
- 梯度下降 (Gradient Descent)
- 牛頓法 (Newton's Method)
- Bias-Variance Trade-off

### 機率分佈
- Gaussian Distribution
- Beta-Binomial 共軛
- Gamma-Poisson 共軛
- MLE vs MAP

### 分類方法
- Naive Bayes Classifier
- Logistic Regression
- Confusion Matrix 和性能指標

### 聚類與 EM
- Gaussian Mixture Model (GMM)
- EM Algorithm (E-step & M-step)

## 🚀 快速開始

### 1. 安裝依賴

本專案使用 [uv](https://github.com/astral-sh/uv) 管理 Python 依賴。

```bash
# 安裝 uv (如果還沒安裝)
pip install uv

# 安裝所有依賴
uv sync
```

### 2. 生成所有視覺化

```bash
# 運行主程式，生成所有圖片
uv run python main.py
```

### 3. 查看筆記

開啟 `ML_NOTES.md` 閱讀完整的學習筆記，配合 `output/images/` 中的圖片學習。

## 📂 專案結構

```
help-my-midterm/
├── ML_NOTES.md              # 完整的機器學習筆記
├── README.md                # 本文件
├── main.py                  # 主程式，執行所有視覺化
├── pyproject.toml           # uv 專案配置
├── visualizations/          # 視覺化腳本目錄
│   ├── regression.py        # 回歸相關視覺化
│   ├── distributions.py     # 機率分佈視覺化
│   ├── classification.py    # 分類方法視覺化
│   └── em_algorithm.py      # EM 演算法視覺化
└── output/                  # 輸出目錄（自動生成）
    └── images/              # 所有生成的圖片
```

## 🎨 單獨運行特定視覺化

如果你只想生成特定主題的視覺化：

```bash
# 回歸分析
uv run python visualizations/regression.py

# 機率分佈
uv run python visualizations/distributions.py

# 分類方法
uv run python visualizations/classification.py

# EM 演算法
uv run python visualizations/em_algorithm.py
```

## 📖 學習建議

1. **先閱讀筆記**：從 `ML_NOTES.md` 開始，理解概念
2. **查看視覺化**：對照 `output/images/` 中的圖片加深理解
3. **親手推導**：筆記中標註的重要公式要親手推導一遍
4. **修改參數**：嘗試修改視覺化腳本中的參數，觀察變化

## 🔧 依賴套件

- Python >= 3.12
- numpy - 數值計算
- matplotlib - 繪圖
- scipy - 科學計算
- scikit-learn - 機器學習工具
- seaborn - 統計視覺化
- pandas - 數據處理
- imageio - 圖片/影片處理

## 📝 授權

此專案為教育用途，歡迎自由使用和分享。

## 🤝 貢獻

歡迎提出 Issue 或 Pull Request 來改進這個專案！

---

**祝你期中考試順利！ 📚✨**