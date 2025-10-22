# ============================================================
# Decision Curve Analysis (DCA) for MLP (Best Parameters)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# ---------- 1. Load Data ----------
FILE_PATH = "Raw_data_MLET.xlsx"
FEATURES = ["Age", "BMI", "Diabetes", "Gender", "Hb", "RBC", "RHR", "Smoking"]
TARGET = "exercise_tolerance"

df = pd.read_excel(FILE_PATH)
df = df[FEATURES + [TARGET]].dropna()
X = df[FEATURES]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ---------- 2. Define MLP (best parameters) ----------
mlp_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
            hidden_layer_sizes=(20,),
            activation="relu",
            learning_rate_init=0.001,
            alpha=1e-5,
            max_iter=1000,
            random_state=42
    ))
])

# ---------- 3. Fit Model ----------
mlp_model.fit(X_train, y_train)
y_prob = mlp_model.predict_proba(X_test)[:, 1]
y_true = y_test.values

# ---------- 4. Compute Net Benefit ----------
def compute_net_benefit_model(thresholds, y_prob, y_true):
    n = len(y_true)
    nb = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        net_benefit = (tp / n) - (fp / n) * (t / (1 - t))
        nb.append(net_benefit)
    return np.array(nb)

def compute_net_benefit_all(thresholds, y_true):
    event_rate = np.mean(y_true)
    return event_rate - (1 - event_rate) * (thresholds / (1 - thresholds))

# Threshold grid and F1-optimal cutoff
thresholds = np.linspace(0.01, 0.99, 100)
best_cutoff = 0.30  # F1-optimal

nb_model = compute_net_benefit_model(thresholds, y_prob, y_true)
nb_all = compute_net_benefit_all(thresholds, y_true)
nb_none = np.zeros_like(thresholds)

# ---------- 5. Plot Publication-Quality DCA (Square Format) ----------
fig, ax = plt.subplots(figsize=(6, 6), dpi=300)  # ✅ 完全正方形输出
plt.rcParams.update({
    "font.size": 11,
    "axes.linewidth": 1.3,
    "axes.labelsize": 12,
    "legend.frameon": False,
    "font.family": "Arial"
})

# DCA curves
ax.plot(thresholds, nb_model, color="#0072B2", lw=2.5, label="MLP model")
ax.plot(thresholds, nb_all, color="black", linestyle="--", lw=1.5, label="Treat all")
ax.plot(thresholds, nb_none, color="gray", linestyle="--", lw=1.5, label="Treat none")
ax.axvline(best_cutoff, color="red", linestyle=":", lw=1.6, label=f"F1-optimal = {best_cutoff:.2f}")

# Highlight net benefit region
ax.fill_between(thresholds, nb_model, nb_none, where=nb_model > nb_none,
                color="#0072B2", alpha=0.25)

# ✅ 强制保持正方形比例
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, max(nb_model) + 0.05)
ax.set_xlabel("Threshold probability", fontsize=12)
ax.set_ylabel("Net benefit", fontsize=12)
ax.set_title("Decision Curve Analysis for MLP Model", pad=10)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("DCA_MLP_BestParam_Square.png", dpi=300, bbox_inches="tight")
plt.show()

print("✅ Saved: DCA_MLP_BestParam_Square.png (square, 300 dpi)")
