# ML Assignment 2 – Obesity Level Classification with Streamlit Deployment

## a. Problem statement

Obesity is usually diagnosed from Body Mass Index (BMI = weight / height²). This project asks a harder question: **can a person's obesity level be predicted from their eating habits, lifestyle and family history alone, without using their height or weight?**

It is a **7-class classification problem**. Five classifiers (Logistic Regression, Decision Tree, K-Nearest Neighbours, Gaussian Naive Bayes and Support Vector Machine) are trained on the same data, tuned with cross-validation, and compared on a held-out test set using Accuracy, AUC, Precision, Recall, F1 and MCC. The trained models are served through an interactive Streamlit app that evaluates them on uploaded test data without retraining.

## b. Dataset description

**Source:** [UCI Machine Learning Repository – Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition) (Palechor & de la Hoz Manotas, 2019). The data comes from individuals in Mexico, Peru and Colombia.

**Size:** 2,111 rows and 16 features plus the target. About 77% of the rows are synthetic: the dataset's authors generated them with SMOTE to balance the classes, as stated on the UCI page.

**Target – `NObeyesdad` (7 classes):**

| Class | Rows | Share |
|---|---|---|
| Insufficient_Weight | 272 | 12.9% |
| Normal_Weight | 287 | 13.6% |
| Overweight_Level_I | 290 | 13.7% |
| Overweight_Level_II | 290 | 13.7% |
| Obesity_Type_I | 351 | 16.6% |
| Obesity_Type_II | 297 | 14.1% |
| Obesity_Type_III | 324 | 15.3% |

The classes are well balanced, so accuracy is not misleading on its own, but macro-averaged metrics are still reported so that every class counts equally.

**Features used (14):**

| Feature | Type | Meaning |
|---|---|---|
| Gender | Binary | Female / Male |
| Age | Numeric | Age in years (14–61) |
| family_history_with_overweight | Binary | A family member is or was overweight |
| FAVC | Binary | Frequently eats high-calorie food |
| FCVC | Numeric | Vegetable consumption frequency (1–3) |
| NCP | Numeric | Number of main meals per day (1–4) |
| CAEC | Ordinal | Eating between meals (no / Sometimes / Frequently / Always) |
| SMOKE | Binary | Smokes |
| CH2O | Numeric | Daily water intake (1–3) |
| SCC | Binary | Monitors calorie intake |
| FAF | Numeric | Physical activity frequency (0–3) |
| TUE | Numeric | Time spent on technology devices (0–2) |
| CALC | Ordinal | Alcohol consumption (no / Sometimes / Frequently / Always) |
| MTRANS | Nominal | Usual transport: Automobile, Bike, Motorbike, Public_Transportation, Walking |

**Dropped features – Height and Weight.** The target classes are defined by BMI, which is calculated from these two columns, so keeping them would let the models recover the label almost directly (typically 90%+ accuracy). They are removed to make this a genuine habits-based prediction task. That leaves 14 features, above the assignment minimum of 12.

**Preprocessing** (`src/data_prep.py`):

1. Dropped Height and Weight.
2. Removed 29 exact duplicate rows, leaving 2,082 rows. None of the duplicates had conflicting labels.
3. Split the data 80/20, stratified on the target with `random_state=2026`: **1,665 training rows** (`data/train_data.csv`) and **417 test rows** (`test_data.csv`). Class shares in the two sets differ by at most 0.2 percentage points.
4. Built a `ColumnTransformer` that is saved inside every model pipeline, producing 18 model inputs:
   - `StandardScaler` on the six numeric features.
   - `OrdinalEncoder` on CAEC and CALC, preserving the order no < Sometimes < Frequently < Always.
   - `OneHotEncoder` on the binary and nominal features; binary columns become a single 0/1 column.

The dataset has no missing values.

## c. GitHub Repository Link

**Repository:** https://github.com/2026ak05299/ML_Assignment2

**Live Streamlit app:** https://2026ak05299.streamlit.app/

## d. Models used

All five models are scikit-learn `Pipeline`s of the shared preprocessor plus a classifier, tuned with `GridSearchCV` (5-fold stratified cross-validation on the training set, scoring on macro F1) and saved to `model/` as `.joblib` files.

| Model | Tuned hyperparameters (best values) |
|---|---|
| Logistic Regression | `C=100` |
| Decision Tree | `criterion="gini"`, `max_depth=15`, `min_samples_leaf=1` |
| KNN | `n_neighbors=3`, `weights="distance"` |
| Naive Bayes (Gaussian) | `var_smoothing=0.1` |
| Support Vector Machine | RBF kernel, `C=1`, `gamma=0.5`, `probability=True` |

### Comparison table

Evaluated on the 417-row held-out test set (`test_data.csv`). Precision, Recall and F1 are macro averages; AUC is one-vs-rest macro.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.5683 | 0.8683 | 0.5546 | 0.5557 | 0.5366 | 0.5004 |
| Decision Tree | 0.7386 | 0.8452 | 0.7277 | 0.7265 | 0.7246 | 0.6950 |
| KNN | 0.7866 | 0.9139 | 0.7865 | 0.7789 | 0.7699 | 0.7532 |
| Naive Bayes | 0.5372 | 0.8536 | 0.5587 | 0.5229 | 0.4907 | 0.4692 |
| Support Vector Machine | 0.8082 | 0.9683 | 0.8227 | 0.8021 | 0.8072 | 0.7779 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Underfits: test accuracy is 0.57, and even training accuracy is only 0.60, although the search chose the weakest regularisation (C=100). A single linear boundary per class cannot separate seven levels using habits alone. Its AUC (0.87) is much higher than its accuracy, meaning it often ranks the correct level highly but not first. It does well only on Obesity_Type_III (F1 0.98) and struggles most with Overweight_Level_II (F1 0.25). |
| Decision Tree | A deep tree (depth 15, leaves of 1 sample) fits the training data almost perfectly (0.998) and drops to 0.74 on test, so it overfits, but it still captures non-linear rules that the linear model misses. It has the lowest AUC of all models (0.85) despite the third-best accuracy, because a tree outputs coarse, mostly 0-or-1 probabilities that rank poorly. |
| KNN | The second-best model (accuracy 0.79, AUC 0.91), using a small neighbourhood (k=3, distance-weighted). Small k works well partly because about 77% of the rows are SMOTE-synthetic, and SMOTE places new points very close to real ones, so near neighbours often share a label. It memorises the training set (training accuracy 1.0), which makes it sensitive to that sampling artefact. |
| Naive Bayes | The weakest model (accuracy 0.54, F1 0.49, MCC 0.47). Its assumptions do not fit this data: the features are not independent (for example, eating habits are correlated), and many are 0/1 or ordinal codes rather than bell-shaped continuous values. It needed heavy smoothing (var_smoothing=0.1) to be stable. Its predictions are skewed towards the obesity classes: it predicts Obesity_Type_II 112 times for 60 actual cases, but Normal_Weight and the two Overweight levels only about 20 times each for 55–58 actual cases. |
| Support Vector Machine | The best model on every metric (accuracy 0.81, F1 0.81, MCC 0.78, AUC 0.97). The RBF kernel models non-linear interactions between habits, and its training accuracy (0.96) is close to its test accuracy, so it generalises better than the tree and KNN. Its weakest class is Normal_Weight (F1 0.62), which sits between the underweight and overweight groups and shares habits with both. |
| Overall winner for this dataset | **Support Vector Machine.** It leads on all six metrics and has the smallest gap between training and test performance among the strong models. Two patterns hold across all models. First, Obesity_Type_III is almost perfectly predicted (F1 0.88–1.00) because in this dataset it is 100% female and always has FCVC = 3, an artefact of the synthetic data rather than a real-world rule. Second, the middle categories (Normal_Weight and Overweight_Level_I/II) are the hardest, since neighbouring BMI bands share similar habits. Without height and weight, about 81% accuracy is a realistic ceiling for habit-based prediction on this data. |

## Repository structure

```
├── app.py                  # Streamlit app: upload test CSV, pick a model, view metrics
├── requirements.txt
├── test_data.csv           # 417-row held-out test set (features + target)
├── .streamlit/config.toml  # App theme
├── data/
│   ├── obesity_levels.csv  # Original UCI dataset
│   └── train_data.csv      # 1,665-row training split
├── model/                  # Saved pipelines (*.joblib), metrics.csv, tuning_summary.csv
└── src/
    ├── data_prep.py        # Loading, cleaning, split, preprocessor
    ├── train_models.py     # GridSearchCV tuning and model saving
    └── evaluate.py         # Test-set metrics and comparison table
```

## How to run

```bash
pip install -r requirements.txt
python src/data_prep.py      # recreate the train/test split
python src/train_models.py   # tune and save the five models
python src/evaluate.py       # print the comparison table
streamlit run app.py         # launch the app locally
```
