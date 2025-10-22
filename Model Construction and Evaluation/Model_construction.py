import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_curve, auc,
    confusion_matrix, f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# -----------------------------
# 1) Load data & keep 8 features
# -----------------------------
FILE_PATH = "Raw_data_MLET.xlsx"
FEATURES = ["Age", "BMI", "Diabetes", "Gender", "Hb", "RBC", "RHR", "Smoking"]
TARGET = "exercise_tolerance"

df = pd.read_excel(FILE_PATH)
df = df[FEATURES + [TARGET]].copy()

X = df[FEATURES]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


# ---------------------------------------
# 2) Define models and full tuning ranges
#    (strict mapping to your S4 ranges)
# ---------------------------------------
# LR: alpha -> l1_ratio；lambda -> C=1/lambda
lambda_grid = np.logspace(np.log10(0.5), np.log10(0.001), num=50)
C_grid_lr = 1.0 / lambda_grid
alpha_grid = np.round(np.arange(0.0, 1.0 + 1e-9, 0.05), 3)

# RF: mtry 不能超过特征数
n_feat = X_train.shape[1]
mtry_list = list(range(1, min(13, n_feat) + 1))

# MLP 隐层备选
hidden_list_mlp = [
    (10,), (20,), (30,), (60,),
    (16, 16), (24, 12), (32, 16), (32, 32), (64, 64), (128, 128)
]

models_and_params = {
    # ---------------------------
    # KNN（表：k=10–200 步长5；distance∈{1,2}；kernel∈{uniform,distance}）
    # ---------------------------
    "KNN": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ]),
        {
            "knn__n_neighbors": list(range(10, 201, 5)),
            "knn__p": [1, 2],                         # 1: 曼哈顿；2: 欧氏
            "knn__weights": ["uniform", "distance"],
        },
    ),

    # ---------------------------
    # 随机森林（表：ntree=100–950 步长50；mtry=1–12；nodesize=2–10）
    # ---------------------------
    "RF": (
        RandomForestClassifier(
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        {
            "n_estimators": list(range(100, 1000, 50)),
            "max_features": mtry_list,                # mtry
            "min_samples_leaf": list(range(2, 11, 1)) # nodesize
        },
    ),

    # ---------------------------
    # 逻辑回归（弹性网）：alpha, lambda（以 C=1/lambda 搜索）
    # ---------------------------
    "LR": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=5000,
                random_state=42
            ))
        ]),
        {
            "lr__l1_ratio": list(alpha_grid),         # alpha
            "lr__C": list(C_grid_lr),                 # 1/lambda，lambda∈[0.5, 0.001]（log 匀布 50 点）
        },
    ),

    # ---------------------------
    # SVM
    # ---------------------------
    "SVM": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        {
            "svm__C": list(2.0 ** np.arange(np.log2(0.1), np.log2(40) + 1e-12, 0.5)),
            "svm__gamma": list(1.0 / (2.0 * (2.0 ** np.arange(-10, -5 + 1e-12, 0.5)) ** 2)),
        },
    ),

    # ---------------------------
    # XGBoost（严格按表格）
    # ---------------------------
    "XGB": (
        XGBClassifier(
            objective="binary:logistic",
            booster="gbtree",
            eval_metric="logloss",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42
        ),
        {
            "n_estimators": list(range(100, 1000, 50)),
            "learning_rate": [0.01] + list(np.round(np.arange(0.06, 0.401, 0.05), 2)),
            "max_depth": list(range(2, 11, 1)),
            "min_child_weight": [1, 2, 3, 4, 5, 6],
            "gamma": [0, 0.2, 0.5, 1, 3],
            "subsample": list(np.round(np.arange(0.05, 1.001, 0.05), 2)),
            "colsample_bytree": list(np.round(np.arange(0.2, 1.01, 0.2), 2)),
            "reg_lambda": list(range(0, 11, 1)),
        },
    ),

    # ---------------------------
    # MLP（严格按表格；epoch 在 sklearn 中无该参数，使用 max_iter 训练轮次）
    # ---------------------------
    "MLP": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(max_iter=1000, random_state=42))
        ]),
        {
            "mlp__hidden_layer_sizes": hidden_list_mlp,
            "mlp__activation": ["relu", "tanh", "logistic", "identity"],
            "mlp__learning_rate_init": [0.001, 0.005, 0.01, 0.05, 0.1],
            "mlp__alpha": [1e-5, 1e-4, 1e-3],
        },
    ),

    # ---------------------------
    # LightGBM（严格按表格）
    # ---------------------------
    "LightGBM": (
        LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),
        {
            "num_leaves": [2, 3, 4],
            "max_depth": [4, 8, 16],
            "min_child_samples": [15, 20, 25, 30, 35, 40],
            "learning_rate": list(np.round(np.arange(0.05, 0.201, 0.01), 2)),
            "n_estimators": list(range(100, 1000, 50)),
            "reg_alpha": [0, 0.1, 0.5, 1.0],
            "reg_lambda": [0, 0.1, 0.5, 1.0],
            "feature_fraction": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "bagging_fraction": [0.6, 0.8, 0.9],
            "bagging_freq": [1],
        },
    ),
}

# ---------------------------------------------------
# 3) Utility: compute metrics at cutoff = 0.5
# ---------------------------------------------------
def compute_metrics(y_true, y_prob, cutoff=0.5):
    y_pred = (y_prob >= cutoff).astype(int)
    auroc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = accuracy_score(y_true, y_pred)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = f1_score(y_true, y_pred)
    return {
        "AUROC": auroc, "AUPRC": auprc, "Cutoff": cutoff,
        "Accuracy": acc, "Sensitivity": sens, "Specificity": spec,
        "PPV": ppv, "NPV": npv, "F1": f1
    }


# ---------------------------------------------------
# 4) Manual grid search with live progress (10-fold CV)
#    — refresh after each parameter combo
# ---------------------------------------------------
results = []
best_models = {}

print("\n=== Manual grid search with live progress (10-fold CV, ROC AUC) ===")
for name, (estimator, param_grid) in models_and_params.items():
    grid = list(ParameterGrid(param_grid))
    print(f"\n>>> {name}: {len(grid)} combinations (live progress)")

    best_auc = -np.inf
    best_params = None

    tbar = tqdm(grid, desc=f"{name} tuning", leave=False)
    for params in tbar:
        # set params (handles Pipeline keys like 'lr__C', 'svm__gamma', etc.)
        model = estimator.set_params(**params)

        # 10-fold CV
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        aucs = []
        for tr_idx, va_idx in cv.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            model.fit(X_tr, y_tr)

            if hasattr(model, "predict_proba"):
                y_val_prob = model.predict_proba(X_va)[:, 1]
            elif hasattr(model, "decision_function"):
                dec = model.decision_function(X_va)
                y_val_prob = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)
            else:
                y_val_prob = np.zeros_like(y_va, dtype=float)

            aucs.append(roc_auc_score(y_va, y_val_prob))

        mean_auc = float(np.mean(aucs))

        # update best
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

        # show running best AUC on the bar
        tbar.set_postfix({"best_auc": round(best_auc, 4)})

    # refit best model on full training set
    best_model = estimator.set_params(**best_params).fit(X_train, y_train)
    best_models[name] = best_model

    # test metrics
    if hasattr(best_model, "predict_proba"):
        y_prob_test = best_model.predict_proba(X_test)[:, 1]
    elif hasattr(best_model, "decision_function"):
        dec = best_model.decision_function(X_test)
        y_prob_test = (dec - dec.min()) / (dec.max() - dec.min() + 1e-12)
    else:
        y_prob_test = np.zeros_like(y_test, dtype=float)

    met = compute_metrics(y_test, y_prob_test, cutoff=0.5)
    results.append({"Model": name, **met})


# -------------------------------------
# 5) Summary & save
# -------------------------------------
res_df = pd.DataFrame(results).sort_values(by="AUROC", ascending=False)
print("\n=== Final Test Performance (sorted by AUROC) ===")
print(res_df.to_string(index=False))
res_df.to_csv("Model_Performance_Metrics.csv", index=False)
print("\nSaved: Model_Performance_Metrics.csv")