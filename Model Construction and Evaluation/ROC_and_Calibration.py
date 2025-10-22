# ============================================================
# Multi-model ROC & Calibration (square 300 dpi)
# + MLP-only Representative Thresholds & Simulated PPV/NPV
# Dataset: Raw_data_MLET.xlsx
# Features: Age, BMI, Diabetes, Gender, Hb, RBC, RHR, Smoking
# Target: exercise_tolerance (0/1)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve, auc,
    accuracy_score, confusion_matrix, f1_score, brier_score_loss
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from statsmodels.nonparametric.smoothers_lowess import lowess

# -----------------------------
# 0) Config
# -----------------------------
RANDOM_STATE = 42
FILE_PATH = "Raw_data_MLET.xlsx"
FEATURES = ["Age", "BMI", "Diabetes", "Gender", "Hb", "RBC", "RHR", "Smoking"]
TARGET = "exercise_tolerance"

np.random.seed(RANDOM_STATE)

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_excel(FILE_PATH)
df = df[FEATURES + [TARGET]].dropna()
X = df[FEATURES]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

# -----------------------------
# 2) Define models (default params)
# -----------------------------
models = {
    # 随机森林（表：ntree=900, mtry=3, nodesize=5, class_weight=balanced, n_jobs=-1）
    "RF": RandomForestClassifier(
        n_estimators=900,
        max_features=3,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ),

    # 逻辑回归（表：alpha=0.1, lambda=3.77081389）
    # 弹性网：alpha→l1_ratio, lambda→C的倒数
    "LR": Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.1,
            C=1.0/3.77081389,  # ≈ 0.265
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ]),

    # KNN（表：k=95, distance=2, kernel=distance）
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=95,
            weights="distance",
            metric="minkowski",
            p=2
        ))
    ]),

    # XGBoost（表：n_estimators=350, lr=0.06, max_depth=8, min_child_weight=5, gamma=3,
    #          subsample=0.7, colsample_bytree=0.4, reg_lambda=4, booster=gbtree）
    "XGB": XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        n_estimators=350,
        learning_rate=0.06,
        max_depth=8,
        min_child_weight=5,
        gamma=3,
        subsample=0.7,
        colsample_bytree=0.4,
        reg_lambda=4.0,
        booster="gbtree",
        random_state=RANDOM_STATE
    ),

    # MLP（表：hidden layer sizes=20, activation=relu, lr=0.001, max_iter=1000, alpha=1e-5）
    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(20,),
            activation="relu",
            learning_rate_init=0.001,
            alpha=1e-5,
            max_iter=1000,
            random_state=42
)
    ]),

    # LightGBM（表：num_leaves=2, max_depth=4, min_child_samples=35, learning_rate=0.12,
    #           n_estimators=200, reg_alpha=0.1, reg_lambda=1, feature_fraction=0.4,
    #           bagging_fraction=0.9, bagging_freq=1, class_weight=balanced）
    "LightGBM": LGBMClassifier(
        objective="binary",
        num_leaves=2,
        max_depth=4,
        min_child_samples=35,
        learning_rate=0.12,
        n_estimators=200,
        reg_alpha=0.1,
        reg_lambda=1.0,
        feature_fraction=0.4,
        bagging_fraction=0.9,
        bagging_freq=1,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
}

# -----------------------------
# 3) Fit, predict prob; gather for plots
# -----------------------------
roc_data = {}     # name -> (y_true, y_prob)
brier_notes = {}  # name -> Brier score
for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    if hasattr(mdl, "predict_proba"):
        y_prob = mdl.predict_proba(X_test)[:, 1]
    else:
        dec = mdl.decision_function(X_test)
        y_prob = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)
    roc_data[name] = (y_test.values, y_prob)
    brier_notes[name] = brier_score_loss(y_test, y_prob)

# -----------------------------
# 4) Plot ROC
# -----------------------------
plt.figure(figsize=(6, 6), dpi=300)
for name, (yt, yp) in roc_data.items():
    fpr, tpr, _ = roc_curve(yt, yp)
    auc_val = roc_auc_score(yt, yp)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01)
plt.xlabel("1 − Specificity (False Positive Rate)")
plt.ylabel("Sensitivity (True Positive Rate)")
plt.title("ROC curves of testing cohort")
plt.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("ROC_AllModels.png", dpi=300)
plt.show()
print("✅ Saved: ROC_AllModels.png")

# -----------------------------
# 5) Plot Calibration
# -----------------------------
def bin_quantile(x, y, n_bins=20):
    dfb = pd.DataFrame({'p': x, 'y': y})
    dfb['bin'] = pd.qcut(dfb['p'], q=n_bins, duplicates="drop")
    g = dfb.groupby('bin', observed=True).mean()
    return g['p'].to_numpy(), g['y'].to_numpy()

plt.figure(figsize=(6, 6), dpi=300)
for name, (yt, yp) in roc_data.items():
    xb, yb = bin_quantile(yp, yt, n_bins=20)
    sm = lowess(yb, xb, frac=0.6, return_sorted=True)
    plt.plot(sm[:, 0], sm[:, 1], lw=2, label=f"{name} (Brier={brier_notes[name]:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Perfectly calibrated")
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(-0.01, 1.01); plt.ylim(-0.01, 1.01)
plt.xlabel("Mean predicted value")
plt.ylabel("Fraction of positives")
plt.title("Calibration plots (reliability curves)")
plt.legend(loc="best", fontsize=9)
plt.tight_layout()
plt.savefig("Calibration_AllModels.png", dpi=300)
plt.show()
print("✅ Saved: Calibration_AllModels.png")

# ============================================================
# 6) MLP — Performance at Representative Thresholds & Simulated PPV/NPV
# ============================================================
def threshold_metrics(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-12)
    spec = tn / (tn + fp + 1e-12)
    ppv  = tp / (tp + fp + 1e-12)
    npv  = tn / (tn + fn + 1e-12)
    f1   = f1_score(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(rec, prec)
    auroc = roc_auc_score(y_true, y_prob)
    return dict(Accuracy=acc, Recall=sens, Specificity=spec, PPV=ppv, NPV=npv,
                F1=f1, AUROC=auroc, AUPRC=auprc, Cutoff=float(thr))

def derive_three_thresholds(y_true, y_prob, min_recall=0.80):
    t_default = 0.5
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if thr.size == 0:
        return {"Default": t_default, "F1-optimal": t_default, "Recall≥0.8 Precision-max": t_default}
    p_t, r_t = prec[1:], rec[1:]
    thr_t = thr
    f1_t = 2 * p_t * r_t / (p_t + r_t + 1e-12)
    t_f1 = float(thr_t[np.nanargmax(f1_t)])
    mask = r_t >= float(min_recall)
    if np.any(mask):
        t_r80 = float(thr_t[mask][np.nanargmax(p_t[mask])])
    else:
        t_r80 = t_f1
    return {"Default": t_default, "F1-optimal": t_f1, "Recall≥0.8 Precision-max": t_r80}

# --- Use MLP only ---
y_true_mlp, y_prob_mlp = roc_data["MLP"]
thr_dict = derive_three_thresholds(y_true_mlp, y_prob_mlp)

mlp_rows = []
for lab, thr in thr_dict.items():
    row = threshold_metrics(y_true_mlp, y_prob_mlp, thr)
    row.update(Model="MLP", Threshold_Label=lab)
    mlp_rows.append(row)
mlp_thr_df = pd.DataFrame(mlp_rows).sort_values("Threshold_Label")
mlp_thr_df.to_excel("MLP_Performance_at_Thresholds.xlsx", index=False)
print("✅ Saved: MLP_Performance_at_Thresholds.xlsx")
print(mlp_thr_df.to_string(index=False))

# --- Simulated PPV/NPV ---
def simulate_ppv_npv(sens, spec, prevalence):
    ppv = (sens * prevalence) / (sens * prevalence + (1 - spec) * (1 - prevalence) + 1e-12)
    npv = (spec * (1 - prevalence)) / ((1 - sens) * prevalence + spec * (1 - prevalence) + 1e-12)
    return ppv, npv

prevalences = [0.05, 0.10, 0.20, 0.30]
sim_rows = []
for _, r in mlp_thr_df.iterrows():
    sens, spec = r["Recall"], r["Specificity"]
    for p in prevalences:
        ppv, npv = simulate_ppv_npv(sens, spec, p)
        sim_rows.append({
            "Model": "MLP",
            "Threshold_Label": r["Threshold_Label"],
            "Cutoff": r["Cutoff"],
            "Prevalence": p,
            "PPV_sim": ppv,
            "NPV_sim": npv
        })
sim_df = pd.DataFrame(sim_rows)
sim_df.to_excel("MLP_PPV_NPV_Simulated_Prevalence.xlsx", index=False)
print("✅ Saved: MLP_PPV_NPV_Simulated_Prevalence.xlsx")

print(sim_df.head())
