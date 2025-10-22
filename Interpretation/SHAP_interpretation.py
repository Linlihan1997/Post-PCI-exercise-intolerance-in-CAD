# -*- coding: utf-8 -*-
# SHAP for MLP (bar + radial inset), single-run script
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.cm import ScalarMappable
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

# -----------------------------
# 1) Load data
# -----------------------------
FILE_PATH = "Raw_data_MLET.xlsx"
FEATURES = ["Age", "BMI", "Diabetes", "Gender", "Hb", "RBC", "RHR", "Smoking"]
TARGET = "exercise_tolerance"

df = pd.read_excel(FILE_PATH)
df = df[FEATURES + [TARGET]].dropna()
X = df[FEATURES]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# -----------------------------
# 2) Train MLP (your hyperparams)
# -----------------------------
mlp = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(10,),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42))
])
mlp.fit(X_train, y_train)

# -----------------------------
# 3) SHAP values (test set)
# -----------------------------
# 对非树模型，使用通用 Explainer；背景样本做子集以提速
bg_idx = np.random.RandomState(42).choice(len(X_train), size=min(200, len(X_train)), replace=False)
background = X_train.iloc[bg_idx]

# 让 SHAP 调用 predict_proba 的阳性概率列
def model_fn(data):
    return mlp.predict_proba(pd.DataFrame(data, columns=FEATURES))[:, 1]

explainer = shap.Explainer(model_fn, background, feature_names=FEATURES)
shap_values = explainer(X_test)  # (n_samples, n_features)

# 取各特征 |SHAP| 的平均作为重要性
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
# 按重要性从大到小排序
order = np.argsort(-mean_abs_shap)
feat_sorted = [FEATURES[i] for i in order]
shap_sorted = mean_abs_shap[order]

# -----------------------------
# 4) Figure: bar + radial inset
# -----------------------------
# 颜色映射（红蓝渐变，与重要性大小对应）
cmap = plt.cm.RdBu_r
norm = mcolors.Normalize(vmin=float(shap_sorted.min()), vmax=float(shap_sorted.max()))
colors = [cmap(norm(v)) for v in shap_sorted]

# 画布与布局（正方形、300 dpi）
fig = plt.figure(figsize=(10, 10), dpi=300)
left_margin, right_margin, bottom_margin, top_margin = 0.08, 0.08, 0.10, 0.10
colorbar_w = 0.025
space = 0.04

plot_bottom = bottom_margin
plot_h = 1.0 - bottom_margin - top_margin
cbar_left = left_margin
main_left = cbar_left + colorbar_w + space
main_w = 1.0 - main_left - right_margin

ax_cbar = fig.add_axes([cbar_left, plot_bottom, colorbar_w, plot_h])
ax_bar  = fig.add_axes([main_left, plot_bottom, main_w, plot_h])

# ---- colorbar
sm = ScalarMappable(cmap=cmap, norm=norm)
cbar = fig.colorbar(sm, cax=ax_cbar, orientation='vertical')
cbar.set_ticks([])
ax_cbar.text(0.5, 1.01, 'High', transform=ax_cbar.transAxes, ha='center', va='bottom', fontsize=13)
ax_cbar.text(0.5, -0.02, 'Low',  transform=ax_cbar.transAxes, ha='center', va='top',    fontsize=13)
ax_cbar.text(-1.25, 0.5, 'Feature contribution (|SHAP|)', transform=ax_cbar.transAxes,
             fontsize=14, rotation=90, va='center')

# ---- bar chart (0 在右侧)
ypos = np.arange(len(feat_sorted))
ax_bar.barh(y=ypos, width=shap_sorted, color=colors, height=0.6)
ax_bar.invert_yaxis()     # 重要的在上
ax_bar.invert_xaxis()     # 让 0 在右侧

# 关闭默认的 ytick 标签，改为手动画在柱条左端外侧且不重叠
ax_bar.set_yticks([])
max_w = float(shap_sorted.max())
x_margin = max_w * 0.14                    # 预留左侧空白，放标签不遮挡
ax_bar.set_xlim(max_w + x_margin, 0.0)     # 扩展左侧坐标范围
label_pad = max_w * 0.1                 # 与柱条左端的距离

for i, (f, w) in enumerate(zip(feat_sorted, shap_sorted)):
    # 在柱条“左端外侧”放置标签：坐标为 (w + label_pad)
    ax_bar.text(w + label_pad, i, f, ha='left', va='center', fontsize=13, color='black')

ax_bar.set_xlabel('Mean |SHAP|', fontsize=14)
ax_bar.spines[['left', 'top']].set_visible(False)
ax_bar.tick_params(axis='x', which='major', direction='in', length=6, labelsize=12)
ax_bar.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
ax_bar.tick_params(axis='x', which='minor', direction='in', length=4)

# ---- radial inset（右移 + 缩小，避免遮挡主轴）
inset_left   = main_left -0.05  # 更靠右
inset_bottom = plot_bottom
inset_size   = 0.4               # 稍小
ax_rad = fig.add_axes([inset_left, inset_bottom, inset_size, inset_size], projection='polar')
ax_rad.patch.set_alpha(0)

# 径向图数据
percent = (shap_sorted / shap_sorted.sum()) * 100
widths  = (shap_sorted / shap_sorted.sum()) * 2 * np.pi
n = len(feat_sorted)
base_len, step, ring_w = 3.0, 0.45, 1.8
tot_len = [base_len + i * step for i in range(n)]
inner_h = [max(0, tl - ring_w) for tl in tot_len]
inner_cols = ['#f7f7f7', '#eeeeee'] * (n // 2 + 1)
inner_cols = inner_cols[:n]
offset = np.pi / 21
thetas = np.cumsum([0] + list(widths[:-1])) - offset

# 画内环+外彩环
ax_rad.bar(thetas, inner_h, width=widths, color=inner_cols, align='edge',
           edgecolor='white', linewidth=1.1)
ax_rad.bar(thetas, [ring_w]*n, width=widths, bottom=inner_h, color=colors,
           align='edge', edgecolor='white', linewidth=1.1)

# 百分比标签
for i in range(n):
    ax_rad.text(thetas[i] + widths[i]/2, tot_len[i] + 0.3, f"{percent[i]:.1f}%",
                ha='center', va='center', fontsize=10)

ax_rad.set_yticklabels([]); ax_rad.set_xticklabels([])
ax_rad.spines['polar'].set_visible(False); ax_rad.grid(False)
ax_rad.set_theta_zero_location('N'); ax_rad.set_theta_direction(-1)
ax_rad.set_ylim(0, max(tot_len) + 1.5)

# -----------------------------
# 5) Save & show
# -----------------------------
plt.savefig("SHAP_MLP_Importance.png", dpi=300, bbox_inches='tight')
plt.show()
print("✅ Saved: SHAP_MLP_Importance.png")
# =============================
# 6) SHAP beeswarm (同配色、正方形、300 dpi)
# =============================


# 与条形图一致的顺序和数据
X_test_sorted = X_test[feat_sorted]
shap_values_sorted = shap_values[:, order]   # 特征维度重排成与 feat_sorted 一致

# 1) 让 SHAP 自己按指定尺寸创建图像（关键！）
shap.summary_plot(
    shap_values_sorted.values,        # (n_samples, n_features)
    X_test_sorted,
    feature_names=feat_sorted,
    plot_type="dot",
    show=False,
    plot_size=(8, 8),                 # <-- 这一步保证方形
    cmap=plt.cm.RdBu_r,               # 与条形图相同配色
    max_display=len(feat_sorted)
)

# 2) 保险起见，再把当前图像尺寸强制设为正方形
fig = plt.gcf()
fig.set_size_inches(8, 8)

# 细节微调 & 保存
plt.xlabel("SHAP value (impact on model output)", fontsize=12)
plt.ylabel("")                # 不重复大标题
plt.tight_layout()
plt.savefig("SHAP_MLP_Beeswarm.png", dpi=300, bbox_inches="tight")
plt.show()
print("✅ Saved: SHAP_MLP_Beeswarm.png")
