# 使用指南 (Usage Guide)

## 📖 如何使用本專案

### 🎯 學習路徑建議

#### 第一步：閱讀筆記
打開 `ML_NOTES.md` 開始學習。筆記分為兩大部分：

1. **第一部分：快速記憶（概念理解）**
   - 回歸分析基礎
   - 機率與分佈
   - 分類與集群

2. **第二部分：A4 雙面筆記（公式與演算法）**
   - 包含所有重要公式
   - 演算法步驟詳解

#### 第二步：生成視覺化
運行主程式生成所有視覺化圖片：

```bash
uv run python main.py
```

這會在 `output/images/` 目錄生成 16 張高品質圖片。

#### 第三步：對照學習
邊看筆記邊查看對應的視覺化圖片，加深理解。

### 📊 視覺化內容對照表

| 圖片檔名 | 對應主題 | 筆記章節 |
|---------|---------|---------|
| `01_lse.png` | 最小平方誤差 | 1️⃣ 基礎與回歸 |
| `02_regularization.png` | L1/L2 正規化 | 1️⃣ 基礎與回歸 |
| `03_gradient_descent.png` | 梯度下降 | 1️⃣ 基礎與回歸 |
| `04_bias_variance.png` | Bias-Variance Trade-off | 1️⃣ 基礎與回歸 |
| `05_newton_vs_gd.png` | Newton's Method vs GD | 1️⃣ 基礎與回歸 |
| `06_gaussian.png` | 高斯分佈 | 2️⃣ 機率與分佈 |
| `07_beta_binomial.png` | Beta-Binomial 共軛 | 2️⃣ 機率與分佈 |
| `08_poisson_gamma.png` | Poisson-Gamma 共軛 | 2️⃣ 機率與分佈 |
| `09_all_distributions.png` | 所有分佈比較 | 2️⃣ 機率與分佈 |
| `10_sigmoid.png` | Sigmoid 函數 | 3️⃣ 分類與集群 |
| `11_logistic_regression.png` | Logistic Regression | 3️⃣ 分類與集群 |
| `12_naive_bayes.png` | Naive Bayes | 3️⃣ 分類與集群 |
| `13_confusion_matrix.png` | 混淆矩陣 | 3️⃣ 分類與集群 |
| `14_gmm_clustering.png` | GMM 聚類 | 3️⃣ 分類與集群 |
| `15_em_steps.png` | EM 演算法步驟 | 3️⃣ 分類與集群 |
| `16_em_concept.png` | EM 演算法概念 | 3️⃣ 分類與集群 |

### 🔧 進階使用

#### 單獨運行特定視覺化

如果只想重新生成特定主題的視覺化：

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

#### 修改視覺化參數

你可以編輯視覺化腳本來：
- 調整圖片大小和解析度
- 修改顏色主題
- 改變數據生成參數
- 增加新的示例

例如，在 `visualizations/regression.py` 中：

```python
# 修改數據量
def generate_data(n_samples=100, noise=0.5, seed=42):
    # 改成 n_samples=200 可以生成更多數據點
```

### 📝 考試準備流程

#### 考前一週
1. **通讀筆記** - 閱讀 ML_NOTES.md 一遍
2. **生成視覺化** - 運行 `uv run python main.py`
3. **理解概念** - 每個主題都配合圖片理解

#### 考前三天
1. **重點複習** - 查看「考試準備建議」章節
2. **親手推導** - 推導必考公式
3. **檢查清單** - 使用「學習檢查清單」自我檢測

#### 考前一天
1. **快速瀏覽** - 看圖片回憶概念
2. **公式記憶** - 複習 A4 雙面筆記中的公式
3. **重點標記** - 標記還不熟悉的部分

### 💡 學習技巧

#### 視覺化記憶法
- 每看到一個公式，就想像對應的圖片
- 例如看到 LSE，就想起誤差線的圖片
- 看到 Gradient Descent，就想起等高線圖上的下降路徑

#### 對比學習法
- 使用對比圖片記憶差異
- 例如：Newton's Method vs Gradient Descent
- 例如：MLE vs MAP

#### 動手實驗法
1. 修改視覺化腳本中的參數
2. 觀察結果變化
3. 理解參數的實際影響

### 🎓 常見問題 (FAQ)

#### Q1: 視覺化生成失敗怎麼辦？
**A:** 確保已正確安裝所有依賴：
```bash
uv sync
```

#### Q2: 圖片顯示亂碼？
**A:** 這是中文字體問題。圖片中的標籤都是英文，不受影響。

#### Q3: 如何列印學習？
**A:** 
1. 列印 `ML_NOTES.md`（建議雙面列印）
2. 選擇性列印重要的視覺化圖片
3. 推薦列印：04, 07, 11, 13, 15

#### Q4: 需要多少時間完成學習？
**A:** 
- 快速瀏覽：2-3 小時
- 深入理解：6-8 小時
- 完全掌握（含推導）：12-15 小時

#### Q5: 如何確認自己學會了？
**A:** 使用筆記末尾的「學習檢查清單」自我檢測。

### 🚀 效率提升技巧

#### 使用標記
在筆記中標記：
- ⭐ 必考重點
- ⚠️ 容易搞混
- ✅ 已經掌握
- 🔄 需要複習

#### 製作小抄
從 A4 雙面筆記部分製作考試小抄（僅供複習）。

#### 組隊學習
1. 與同學分享視覺化圖片
2. 互相講解概念
3. 一起推導公式

### 📚 延伸資源

如果你想深入學習某個主題：

1. **線性回歸** → 閱讀 Bishop PRML Ch3
2. **貝葉斯方法** → 閱讀 Bishop PRML Ch2
3. **分類方法** → 閱讀 Bishop PRML Ch4
4. **EM 演算法** → 閱讀 Bishop PRML Ch9

### 🎯 考試當天建議

1. **考前 30 分鐘**
   - 快速瀏覽所有視覺化圖片
   - 回憶關鍵概念

2. **考試期間**
   - 遇到問題先在腦中「看見」對應的圖片
   - 從圖片推導到公式

3. **時間分配**
   - 簡單題：30%
   - 推導題：50%
   - 證明題：20%

---

**祝你考試成功！記住：理解比記憶更重要 🎓✨**
