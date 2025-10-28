# 📘 Machine Learning 核心筆記整理

**完整機器學習筆記 - 從零開始理解所有概念**

本筆記提供完整的機器學習核心概念，配合 Python 視覺化輔助理解。所有圖片和影片都可以透過執行腳本自動生成。

---

## 🧠 第一部分：快速記憶（概念理解）

### 1️⃣ 基礎與回歸（Regression & Basics）

#### **Least Squares Error (LSE)**
- **為何合理**：假設誤差服從高斯分佈（Gaussian Distribution）
- **目標**：最小化預測值與真實值之間的平方差總和
- **數學表示**：$E_D(w) = \frac{1}{2}\sum_{n=1}^{N} (y_n - w^T\phi(x_n))^2$

**直觀理解**：
LSE 就像在找一條最能代表數據趨勢的線。每個點到線的距離的平方加起來最小的那條線就是我們要的。

#### **Regularization（正規化）**
正規化是防止模型過度擬合的技術：

- **L2 Regularization (Ridge)**
  - 使權重 $w$ 變小但不為零
  - 公式：$E(w) = E_D(w) + \frac{\lambda}{2} w^T w$
  - 效果：讓模型更平滑，減少對訓練數據的過度依賴

- **L1 Regularization (Lasso)**
  - 使權重稀疏（許多權重變成 0）
  - 公式：$E(w) = E_D(w) + \frac{\lambda}{2}|w|$
  - 效果：自動進行特徵選擇

**為什麼需要正規化？**
想像你在記憶一組數字。如果你記住每個數字的每個細節（過度擬合），遇到新的相似數字時反而會搞混。正規化就是教你抓住重點而不是死記硬背。

#### **Gradient Descent (GD)**
- **概念**：沿著梯度最陡峭的相反方向更新參數
- **更新規則**：$w^{(t+1)} = w^{(t)} - \eta \nabla E(w^{(t)})$
- **學習率** $\eta$：控制每次更新的步長

**登山比喻**：
梯度下降就像在濃霧中下山。你看不到山底在哪，但可以感覺到哪個方向最陡。每次都往最陡的方向走一小步，最終會到達山底（最小值）。

#### **Newton's Method**
- **特點**：使用二階導數（Hessian 矩陣）加速收斂
- **更新規則**：$w^{(t+1)} = w^{(t)} - H^{-1}\nabla E(w^{(t)})$
- **優點**：收斂速度快
- **缺點**：計算 Hessian 矩陣和其逆矩陣成本高
- **注意**：若 Hessian 奇異（singular），應改用梯度下降

#### **Bias–Variance Trade-off**
這是機器學習中最重要的概念之一：

- **高偏差（High Bias）**
  - 模型太簡單
  - 無法捕捉數據的真實模式
  - 結果：欠擬合（Underfitting）
  - 例子：用直線擬合曲線數據

- **高變異（High Variance）**
  - 模型太複雜
  - 對訓練數據過度敏感
  - 結果：過度擬合（Overfitting）
  - 例子：用高次多項式擬合少量數據點

**射箭比喻**：
- 高偏差：箭都偏離靶心，但很集中 → 瞄準有問題
- 高變異：箭分散在靶子各處 → 不穩定
- 理想狀態：箭都集中在靶心附近

---

### 2️⃣ 機率與分佈（Probability & Distributions）

#### **Bayes' Theorem（貝葉斯定理）**
$$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

或簡寫為：
$$Posterior \propto Likelihood \times Prior$$

**直觀理解**：
- **Prior（先驗）**：在看到數據之前，我們對參數的信念
- **Likelihood（似然）**：給定參數，觀察到這些數據的可能性
- **Posterior（後驗）**：看到數據後，更新的信念

**例子**：醫療診斷
- Prior: 這個疾病的發病率是 1%
- Likelihood: 如果有病，測試陽性的機率是 99%
- Posterior: 測試陽性後，真的有病的機率是多少？

#### **MLE (Maximum Likelihood Estimation)**
- **目標**：找到參數 $\theta$ 使觀測資料機率 $P(D|\theta)$ 最大
- **步驟**：
  1. 寫出似然函數 $L(\theta) = P(D|\theta)$
  2. 取對數得到 log-likelihood $\log L(\theta)$
  3. 對 $\theta$ 求導並令其為零
  4. 解出 $\theta_{MLE}$

**範例：高斯分佈的 MLE**
$$\mu_{MLE} = \frac{1}{N}\sum_{n=1}^{N} x_n$$

**範例：Poisson 分佈的 MLE**
$$\lambda_{MLE} = \frac{1}{N}\sum_{n=1}^{N} k_n$$

#### **MAP (Maximum A Posteriori)**
- **與 MLE 的不同**：引入先驗分佈 $P(\theta)$
- **目標**：最大化 $P(\theta|D) \propto P(D|\theta)P(\theta)$
- **優點**：可以融入先驗知識，避免過度擬合

**MLE vs MAP**：
- MLE：完全根據數據
- MAP：數據 + 先驗知識
- 當先驗是均勻分佈時，MAP = MLE

#### **Conjugate Prior（共軛先驗）**
當先驗和似然結合後，後驗仍與先驗具有相同的分佈形式。

**Beta–Binomial 共軛對**
- **Prior**: $Beta(\mu|a,b)$
- **Likelihood**: $Bin(m|N,\mu)$（m 次成功，N 次試驗）
- **Posterior**: $Beta(\mu|a+m, b+l)$ 其中 $l = N-m$

**為什麼有用？**
共軛先驗讓貝葉斯推斷變得簡單。每次看到新數據，只需要更新參數，不需要重新計算複雜的積分。

**Gamma–Poisson 共軛對**
- **Prior**: $Gam(\lambda|a,b)$
- **Likelihood**: $Poisson(k|\lambda)$
- **Posterior**: $Gam(\lambda|a+\sum k_i, b+N)$

#### **Gaussian Distribution（高斯分佈/常態分佈）**

**一維高斯分佈**：
$$N(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

**多維高斯分佈**：
$$N(x|\mu,\Sigma) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$$

**重要性質**：

1. **Isotropic Gaussian（各向同性）**
   - 共變異數矩陣 $\Sigma = \sigma^2 I$（對角且值相同）
   - 各維度獨立且有相同的變異數

2. **Affine Property（仿射性質）**
   - 若 $x \sim N(\mu, \Sigma)$ 且 $y = Ax + b$
   - 則 $y \sim N(A\mu + b, A\Sigma A^T)$
   - **意義**：高斯分佈經過線性轉換後仍是高斯分佈

---

### 3️⃣ 分類與集群（Classification & Clustering）

#### **Naive Bayes Classifier（樸素貝葉斯分類器）**
- **核心假設**：各特徵在給定類別下條件獨立
- **分類規則**：
$$\hat{y} = \arg\max_k P(C_k)\prod_{i=1}^{D} P(x_i|C_k)$$

**為什麼叫「樸素」？**
因為假設特徵之間完全獨立，這個假設很「樸素」（通常不成立），但實際效果卻很好。

**處理不同類型特徵**：

1. **離散特徵**（加拉普拉斯平滑）：
$$P(x_i=j|C_k) = \frac{\text{Count}(x_i=j \text{ in } C_k) + 1}{\text{Count}(C_k) + \text{TotalBins}}$$

2. **連續特徵**（假設為高斯分佈）：
$$P(x_i|C_k) = N(x_i|\mu_{ik}, \sigma_{ik}^2)$$

#### **Logistic Regression（邏輯回歸）**
- **名字的誤導**：雖然叫回歸，實際上是分類模型
- **Sigmoid 函數**：$\sigma(a) = \frac{1}{1+e^{-a}}$
- **模型**：$P(C_1|\phi) = \sigma(w^T\phi)$
- **損失函數**（交叉熵）：
$$E(w) = -\sum_{n=1}^{N}[t_n\ln y_n + (1-t_n)\ln(1-y_n)]$$

**為什麼用 Sigmoid？**
1. 輸出範圍在 [0,1]，可解釋為機率
2. 可微分，方便梯度下降
3. 有良好的數學性質

**訓練方法**：
- 梯度下降：$\nabla E(w) = \Phi^T(y-t)$
- Newton's Method / IRLS：使用 Hessian 矩陣加速

#### **EM Algorithm (Expectation–Maximization)**
用於含有潛在變數或缺失數據時的參數估計。

**兩步驟迭代**：

1. **E-step（期望步驟）**
   - 計算 responsibilities
   - 估計潛在變數的後驗機率
   - $\gamma(z_k) = P(Z=k|X,\theta^{old})$

2. **M-step（最大化步驟）**
   - 固定 responsibilities
   - 最大化參數使對數似然最大
   - $\theta^{new} = \arg\max_\theta Q(\theta, \theta^{old})$

**GMM（高斯混合模型）範例**：

E-step:
$$\gamma(z_{nk}) = \frac{\pi_k N(x_n|\mu_k,\Sigma_k)}{\sum_{j=1}^{K}\pi_j N(x_n|\mu_j,\Sigma_j)}$$

M-step:
$$\pi_k = \frac{N_k}{N}, \quad \mu_k = \frac{1}{N_k}\sum_{n=1}^{N}\gamma(z_{nk})x_n$$
$$\Sigma_k = \frac{1}{N_k}\sum_{n=1}^{N}\gamma(z_{nk})(x_n-\mu_k)(x_n-\mu_k)^T$$

其中 $N_k = \sum_{n=1}^{N}\gamma(z_{nk})$

**為什麼叫 EM？**
- E-step: 根據當前參數，**期望**潛在變數是什麼
- M-step: 根據期望的潛在變數，**最大化**參數

#### **Confusion Matrix（混淆矩陣）**

|           | Predict Positive | Predict Negative |
|-----------|------------------|------------------|
| Actually Positive | TP (True Positive)  | FN (False Negative) |
| Actually Negative | FP (False Positive) | TN (True Negative)  |

**重要指標**：

1. **Sensitivity (敏感度/召回率/TPR)**
   $$Sensitivity = \frac{TP}{TP + FN}$$
   - 實際為正的樣本中，有多少被正確預測為正

2. **Specificity (特異度/TNR)**
   $$Specificity = \frac{TN}{TN + FP}$$
   - 實際為負的樣本中，有多少被正確預測為負

3. **Precision (精確率)**
   $$Precision = \frac{TP}{TP + FP}$$
   - 預測為正的樣本中，有多少真的是正

4. **F1 Score**
   $$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$
   - 精確率和召回率的調和平均

**醫療檢測例子**：
- Sensitivity 高：不會漏掉真正的病人（少 False Negative）
- Specificity 高：不會誤診健康的人（少 False Positive）

---

## 📄 第二部分：A4 雙面筆記（公式與演算法）

### 🧮 Side A：Regression, Gaussian & Bayes

#### (1) Linear Regression

**基本模型**：
$$y(x, w) = w^T \phi(x)$$

**損失函數**：
- **LSE**: $E_D(w) = \frac{1}{2}\sum_{n=1}^{N} (y_n - w^T\phi(x_n))^2$
- **Ridge (L2)**: $E(w) = E_D(w) + \frac{\lambda}{2} w^T w$
- **Lasso (L1)**: $E(w) = E_D(w) + \frac{\lambda}{2}|w|$

**解析解**：
- **LSE**: $w_{LSE} = (\Phi^T\Phi)^{-1}\Phi^T t$
- **Ridge**: $w_{Ridge} = (\lambda I + \Phi^T\Phi)^{-1}\Phi^T t$

**優化方法**：
- **GD**: $w^{(t+1)} = w^{(t)} - \eta \nabla E(w^{(t)})$
- **Newton**: $w^{(t+1)} = w^{(t)} - H^{-1}\nabla E(w^{(t)})$
  - Hessian: $H = \Phi^T\Phi$ (for LSE)

---

#### (2) Probability Distributions

**離散分佈**：

1. **Bernoulli（伯努利）**
   $$Bern(x|\mu) = \mu^x (1-\mu)^{1-x}$$
   - 單次試驗，成功機率 $\mu$

2. **Binomial（二項式）**
   $$Bin(k|N,\mu) = \binom{N}{k}\mu^k(1-\mu)^{N-k}$$
   - N 次獨立試驗，k 次成功

3. **Poisson（泊松）**
   $$P(k|\lambda) = e^{-\lambda}\frac{\lambda^k}{k!}$$
   - 單位時間內事件發生 k 次的機率

**連續分佈**：

1. **Beta（貝塔）**
   $$Beta(\mu|a,b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}$$
   - 定義在 [0,1]，是 Binomial 的共軛先驗

2. **Gamma（伽瑪）**
   $$Gam(\lambda|a,b) \propto \lambda^{a-1}e^{-b\lambda}$$
   - 定義在 $(0,\infty)$，是 Poisson 的共軛先驗

3. **Gaussian（高斯）**
   - 1D: $N(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
   - Multi-D: $N(x|\mu,\Sigma) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}e^{-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)}$

**共軛關係**：

- **Beta–Binomial**
  - Prior: $Beta(\mu|a,b)$
  - Likelihood: $Bin(m|N,\mu)$
  - Posterior: $Beta(\mu|a+m, b+l)$ where $l=N-m$

- **Gamma–Poisson**
  - Prior: $Gam(\lambda|a,b)$
  - Likelihood: $\prod_{n=1}^{N} Poisson(k_n|\lambda)$
  - Posterior: $Gam(\lambda|a+\sum k_i, b+N)$

**Affine Property of Gaussian**：
- If $x \sim N(\mu, \Sigma)$ and $y = Ax + b$
- Then $y \sim N(A\mu + b, A\Sigma A^T)$

---

#### (3) MLE, MAP & Bayesian Regression

**Maximum Likelihood Estimation (MLE)**：
$$\theta_{ML} = \arg\max_\theta P(D|\theta) = \arg\max_\theta \log P(D|\theta)$$

**Gaussian MLE**：
$$\mu_{ML} = \frac{1}{N}\sum_{n=1}^{N} x_n, \quad \sigma_{ML}^2 = \frac{1}{N}\sum_{n=1}^{N}(x_n - \mu_{ML})^2$$

**Maximum A Posteriori (MAP)**：
$$\theta_{MAP} = \arg\max_\theta P(\theta|D) = \arg\max_\theta P(D|\theta)P(\theta)$$

**L2 Regularization as MAP**：
- Likelihood: $P(t|\Phi, w, \beta) = N(t|\Phi w, \beta^{-1}I)$
- Prior: $P(w|\alpha) = N(w|0, \alpha^{-1}I)$
- Result: Ridge regression with $\lambda = \alpha/\beta$

**Bayesian Linear Regression**：

Prior:
$$P(w) = N(w|m_0, S_0)$$

Posterior:
$$P(w|t) = N(w|m_N, S_N)$$

where:
$$S_N^{-1} = S_0^{-1} + \beta \Phi^T\Phi$$
$$m_N = S_N(S_0^{-1}m_0 + \beta\Phi^Tt)$$

**Predictive Distribution**：
$$p(t_{new}|t, \phi_{new}) = N(t_{new}|m_N^T\phi_{new}, \sigma_N^2(\phi_{new}))$$
$$\sigma_N^2(\phi_{new}) = \frac{1}{\beta} + \phi_{new}^T S_N \phi_{new}$$

**Sequential Estimation**：
$$\mu_{ML}^{(N)} = \mu_{ML}^{(N-1)} + \frac{1}{N}(x_N - \mu_{ML}^{(N-1)})$$

這個公式顯示：新的估計 = 舊的估計 + 調整量

---

### 🧩 Side B：Classification, Clustering & EM

#### (1) Naive Bayes Classifier

**一般形式**：
$$\hat{y} = \arg\max_k P(C_k) \prod_{i=1}^{D} P(x_i|C_k)$$

**離散特徵**（Laplace Smoothing）：
$$P(x_i=j|C_k) = \frac{\text{Count}(x_i=j \text{ in } C_k) + 1}{\text{Count}(C_k) + \text{TotalBins}}$$

**連續特徵**（Gaussian Assumption）：
$$P(x_i|C_k) = N(x_i|\mu_{ik}, \sigma_{ik}^2)$$

**訓練步驟**：
1. 計算每個類別的先驗 $P(C_k) = \frac{N_k}{N}$
2. 對每個特徵，計算條件機率 $P(x_i|C_k)$
3. 預測時使用貝葉斯規則

---

#### (2) Logistic Regression

**Sigmoid Function**：
$$\sigma(a) = \frac{1}{1+e^{-a}}$$

**Properties**：
- $\sigma(-a) = 1 - \sigma(a)$
- $\frac{d\sigma}{da} = \sigma(a)(1-\sigma(a))$

**Model**：
$$y_n = P(C_1|\phi_n) = \sigma(w^T\phi_n)$$

**Cross-Entropy Error**：
$$E(w) = -\sum_{n=1}^{N}[t_n\ln y_n + (1-t_n)\ln(1-y_n)]$$

**Gradient**：
$$\nabla E(w) = \sum_{n=1}^{N}(y_n - t_n)\phi_n = \Phi^T(y - t)$$

**Hessian**：
$$H = \nabla\nabla E(w) = \Phi^T R \Phi$$
where $R = \text{diag}(y_n(1-y_n))$

**IRLS (Iteratively Reweighted Least Squares)**：
$$w^{(new)} = w^{(old)} - H^{-1}\nabla E(w^{(old)})$$
$$w^{(new)} = (\Phi^T R \Phi)^{-1}\Phi^T R z$$
where $z = \Phi w^{(old)} - R^{-1}(y-t)$

---

#### (3) EM Algorithm

**目標**：最大化 $P(X|\theta)$ 當有潛在變數 $Z$ 時

**E-Step**：
$$Q(\theta, \theta^{old}) = \sum_Z P(Z|X,\theta^{old}) \log P(X,Z|\theta)$$
$$\gamma(z_k) = P(Z=k|X, \theta^{old})$$

**M-Step**：
$$\theta^{new} = \arg\max_\theta Q(\theta, \theta^{old})$$

**GMM (Gaussian Mixture Model) 範例**：

**E-step** (計算 responsibilities)：
$$\gamma(z_{nk}) = \frac{\pi_k N(x_n|\mu_k,\Sigma_k)}{\sum_{j=1}^{K}\pi_j N(x_n|\mu_j,\Sigma_j)}$$

**M-step** (更新參數)：
$$N_k = \sum_{n=1}^{N}\gamma(z_{nk})$$

$$\pi_k^{new} = \frac{N_k}{N}$$

$$\mu_k^{new} = \frac{1}{N_k}\sum_{n=1}^{N}\gamma(z_{nk})x_n$$

$$\Sigma_k^{new} = \frac{1}{N_k}\sum_{n=1}^{N}\gamma(z_{nk})(x_n-\mu_k^{new})(x_n-\mu_k^{new})^T$$

**收斂判斷**：
監控對數似然 $\log P(X|\theta)$ 是否不再顯著增加

---

#### (4) 其他重要概念

**Entropy（熵）**：
$$H(X) = -\sum_{i=1}^{n} P(x_i)\log_2 P(x_i)$$
- 測量不確定性或信息量
- 熵越高，不確定性越大

**KL Divergence（KL 散度）**：
$$KL(P||Q) = \sum_{i=1}^{n} P(x_i)\log\frac{P(x_i)}{Q(x_i)}$$
- 測量兩個分佈的差異
- 非對稱：$KL(P||Q) \neq KL(Q||P)$
- 非負：$KL(P||Q) \geq 0$

**Performance Metrics**：

From Confusion Matrix:

|           | Predict 1 | Predict 0 |
|-----------|-----------|-----------|
| Actual 1  | TP        | FN        |
| Actual 0  | FP        | TN        |

- **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
- **Precision**: $\frac{TP}{TP + FP}$
- **Recall/Sensitivity**: $\frac{TP}{TP + FN}$
- **Specificity**: $\frac{TN}{TN + FP}$
- **F1-Score**: $\frac{2 \times Precision \times Recall}{Precision + Recall}$

---

## 🎯 考試準備建議

### 必須親手推導的公式

1. **Linear Regression**
   - LSE 的解析解推導
   - Ridge Regression 的解析解推導
   - 從 MAP 角度推導 L2 正規化

2. **Bayesian Methods**
   - Beta–Binomial 共軛關係證明
   - Gamma–Poisson 共軛關係證明
   - Bayesian Linear Regression 的 Posterior 推導

3. **Classification**
   - Naive Bayes 的推導和應用
   - Logistic Regression 的梯度計算
   - Logistic Regression 的 Hessian 矩陣推導

4. **EM Algorithm**
   - GMM 的 E-step 推導
   - GMM 的 M-step 推導
   - 理解為什麼 EM 會收斂

5. **Sequential Estimation**
   - 從 MLE 推導 Sequential Update 公式
   - 理解為什麼可以逐步更新

### 重點概念複習

1. **理解數學與直覺的對應**
   - 每個公式背後的物理/幾何意義
   - 參數對模型的影響

2. **掌握機率分佈之間的關係**
   - 共軛關係的意義和應用
   - 各種分佈適用的場景

3. **優化方法的差異**
   - GD vs Newton's Method
   - 何時該用哪種方法

4. **Bias-Variance Trade-off**
   - 在不同問題中如何體現
   - 如何平衡

### 實作練習建議

運行所有視覺化腳本，觀察：
1. 參數變化對結果的影響
2. 不同方法的收斂速度
3. 過度擬合和欠擬合的視覺化差異

---

## 📚 視覺化腳本使用說明

本筆記配備完整的 Python 視覺化腳本，幫助你直觀理解所有概念。

### 安裝依賴

```bash
# 使用 uv 安裝所有依賴
uv sync
```

### 生成所有視覺化

```bash
# 運行主腳本生成所有圖片和影片
uv run python main.py
```

### 單獨運行特定主題

```bash
# 回歸相關視覺化
uv run python visualizations/regression.py

# 機率分佈視覺化
uv run python visualizations/distributions.py

# 分類方法視覺化
uv run python visualizations/classification.py

# EM 演算法視覺化
uv run python visualizations/em_algorithm.py

# Bias-Variance Trade-off
uv run python visualizations/bias_variance.py
```

### 輸出位置

所有生成的圖片和影片會儲存在 `output/` 目錄下：
- `output/images/` - 所有靜態圖片
- `output/videos/` - 所有動畫影片

---

## ✅ 學習檢查清單

使用以下清單檢查你的學習進度：

### 基礎概念
- [ ] 理解 LSE 的合理性（高斯假設）
- [ ] 理解 L1 vs L2 正規化的差異
- [ ] 能解釋 Bias-Variance Trade-off
- [ ] 理解梯度下降的運作原理
- [ ] 知道何時用 Newton's Method

### 機率與統計
- [ ] 能寫出貝葉斯定理並解釋各項
- [ ] 理解 MLE 和 MAP 的差異
- [ ] 掌握 Beta-Binomial 共軛關係
- [ ] 掌握 Gamma-Poisson 共軛關係
- [ ] 理解高斯分佈的 Affine 性質

### 分類方法
- [ ] 能推導 Naive Bayes 分類器
- [ ] 理解為什麼 Logistic Regression 用於分類
- [ ] 能計算 Logistic Regression 的梯度
- [ ] 理解混淆矩陣的各個指標
- [ ] 知道何時用 Precision，何時用 Recall

### 進階方法
- [ ] 理解 EM 演算法的 E-step 和 M-step
- [ ] 能推導 GMM 的 EM 更新公式
- [ ] 理解 Sequential Estimation 的意義
- [ ] 能計算 KL Divergence 和 Entropy

### 實作能力
- [ ] 能用 Python 實現線性回歸
- [ ] 能用 Python 實現 Naive Bayes
- [ ] 能用 Python 實現 Logistic Regression
- [ ] 能視覺化 Bias-Variance Trade-off
- [ ] 能實現簡單的 EM 演算法

---

## 🔗 延伸學習資源

1. **經典教材**
   - Pattern Recognition and Machine Learning (Bishop)
   - The Elements of Statistical Learning (Hastie, Tibshirani, Friedman)

2. **線上課程**
   - Andrew Ng's Machine Learning Course
   - StatQuest 的 YouTube 頻道（概念解釋非常直觀）

3. **實作練習**
   - Kaggle 競賽和教學
   - Scikit-learn 官方教程

---

**祝你期中考試順利！ 📚✨**

記住：機器學習的核心在於理解概念，而非死記公式。多推導、多視覺化、多思考「為什麼」，你一定能掌握這些內容！
