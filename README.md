# 📊 Data Preprocessing & Transformation — ML Reference Repository

A structured, hands-on reference for all major **data preprocessing and transformation techniques** used in Machine Learning pipelines. Each technique is implemented in a clean Jupyter Notebook with real dataset examples, visual comparisons, and best-practice notes.

---

## 🗂️ Repository Structure

```
data-preprocessing-ml/
│
├── 01_Feature_Scaling/
│   ├── min_max_scaler.ipynb
│   ├── standardization_zscore.ipynb
│   ├── robust_scaler.ipynb
│   ├── maxabs_scaler.ipynb
│   └── normalizer_l1_l2.ipynb
│
├── 02_Distribution_Transforms/
│   ├── log_transform.ipynb
│   ├── square_root_transform.ipynb
│   ├── box_cox_transform.ipynb
│   ├── yeo_johnson_transform.ipynb
│   ├── quantile_transform.ipynb
│   ├── power_transform.ipynb
│   └── reciprocal_transform.ipynb
│
├── 03_Outlier_Handling/
│   ├── winsorization_clipping.ipynb
│   ├── zscore_outlier_removal.ipynb
│   └── iqr_method.ipynb
│
├── 04_Encoding_Techniques/
│   ├── one_hot_encoding.ipynb
│   ├── label_encoding.ipynb
│   ├── ordinal_encoding.ipynb
│   └── target_mean_encoding.ipynb
│
├── 05_Missing_Value_Imputation/
│   ├── mean_median_imputation.ipynb
│   ├── knn_imputation.ipynb
│   └── indicator_plus_impute.ipynb
│
├── 06_Full_Pipeline_Example/
│   └── end_to_end_preprocessing.ipynb
│
└── README.md
```

---

## 📚 Techniques Covered

### 1 · Feature Scaling

| Technique | Formula | Best For |
|---|---|---|
| Min-Max Scaler | `(x − min) / (max − min)` | Neural networks, KNN, image data |
| Standardization (Z-score) | `(x − μ) / σ` | SVM, PCA, logistic regression |
| Robust Scaler | `(x − median) / IQR` | Data with outliers |
| MaxAbs Scaler | `x / max(\|x\|)` | Sparse matrices, NLP/TF-IDF |
| Normalizer (L1/L2) | `x / ‖x‖` | Text classification, cosine similarity |

### 2 · Distribution Transforms

| Technique | Skew Direction | Notes |
|---|---|---|
| Log Transform | Right skew ✅ | Most common; requires x > 0 |
| Square Root | Right skew (mild) ✅ | Good for count/Poisson data |
| Box-Cox | Right or Left ✅ | Auto-finds λ; positive values only |
| Yeo-Johnson | Right or Left ✅ | Works with zeros and negatives |
| Quantile Transform | Any distribution ✅ | Maps to uniform or normal; non-parametric |
| Power Transform (x²) | Left skew ✅ | Amplifies outliers — use carefully |
| Reciprocal (1/x) | Heavy right skew ✅ | Reverses order; x ≠ 0 required |

### 3 · Outlier Handling

| Technique | Approach | When to Use |
|---|---|---|
| Winsorization | Clips to percentile (e.g., p1–p99) | Financial data, cannot remove rows |
| Z-score Method | Flag \|z\| > 3 | Normally distributed features |
| IQR Method | Flag beyond Q1 − 1.5·IQR or Q3 + 1.5·IQR | Any distribution; non-parametric |

### 4 · Encoding Techniques

| Technique | Output | When to Use |
|---|---|---|
| One-Hot Encoding | Binary vector | Nominal features, low cardinality |
| Label Encoding | Integer (0, 1, 2…) | Ordinal data or tree models only |
| Ordinal Encoding | Meaningful integer rank | When natural order exists |
| Target Encoding | Mean of target per category | High-cardinality categoricals |

### 5 · Missing Value Imputation

| Technique | Strategy | Best For |
|---|---|---|
| Mean / Median Imputation | Fill with column statistic | Small % missing, MCAR data |
| KNN Imputation | Fill using k nearest neighbors | Preserves feature relationships |
| Indicator + Impute | Add missingness flag, then impute | When missingness itself is informative |

---

## 🔁 Recommended Preprocessing Pipeline

```
Raw Data
    │
    ▼
1. Handle Missing Values
   └── (Indicator flag → Median/KNN impute)
    │
    ▼
2. Handle Outliers
   └── (Winsorize for skewed / IQR for general)
    │
    ▼
3. Fix Skewness
   └── (Log / Sqrt / Yeo-Johnson)
    │
    ▼
4. Scale Features
   └── (Standardize or Min-Max based on algorithm)
    │
    ▼
5. Encode Categoricals
   └── (One-Hot / Ordinal / Target Encoding)
    │
    ▼
Ready for Model Training ✅
```

> ⚠️ **Critical Rule — No Data Leakage:** Always `fit` your scalers and transformers **only on training data**, then `transform` both train and test sets. Never fit on the full dataset.

---

## ⚡ Quick Skew Reference

```
Right Skewed (long tail → right)     Left Skewed (long tail → left)
────────────────────────────────     ──────────────────────────────
✓ Log Transform (most common)        ✓ Square / Cube Transform
✓ Square Root Transform              ✓ Reflect + Log: log(max − x + 1)
✓ Box-Cox (λ < 1)                    ✓ Box-Cox (λ > 1)
✓ Reciprocal (heavy skew)            ✓ Yeo-Johnson
✓ Yeo-Johnson (universal)            ✓ Yeo-Johnson (universal)
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **pandas** — data loading and manipulation
- **NumPy** — numerical operations
- **scikit-learn** — scalers, encoders, imputers
- **SciPy** — Box-Cox, statistical transforms
- **matplotlib / seaborn** — before vs. after distribution plots

Install all dependencies:

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

---

## 📖 How Each Notebook Is Structured

Every notebook in this repository follows a consistent format:

1. **What it is** — brief explanation of the technique
2. **Formula** — mathematical definition
3. **When to use / avoid** — practical decision guide
4. **Code implementation** — using scikit-learn or NumPy
5. **Visual comparison** — before vs. after distribution plot
6. **Effect on ML model** — brief note on which algorithms benefit

---

## 🙋 About

This repository is part of my hands-on ML portfolio, built while completing my B.Tech in Artificial Intelligence & Data Science. Each notebook is written to serve as a reference I (and others) can revisit during real project work — not just theory, but practical implementation with commentary.

**Connect:** [LinkedIn](https://www.linkedin.com/in/yatharth-limje-491551317/) · [GitHub](https://github.com/yatharthgeeta)

---

## ⭐ If this helped you

Give it a star — it keeps the motivation going! Contributions and suggestions are welcome via Issues or Pull Requests.
