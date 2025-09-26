import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cm import get_cmap

# -------------------- 0) 工具函数 --------------------
def mixture_truncated_norm(n, mus, sigmas, weights=None, low=0.0, high=1.0, rng=None):
    """
    从混合高斯分布采样，并截断到 [low, high]。
    n: 样本数
    mus/sigmas/weights: 各成分的均值/标准差/权重
    rng: numpy Generator
    """
    rng = np.random.default_rng() if rng is None else rng
    k = len(mus)
    if weights is None:
        weights = np.ones(k) / k
    weights = np.array(weights) / np.sum(weights)
    comps = rng.choice(k, size=n, p=weights)
    x = np.array([rng.normal(mus[c], sigmas[c]) for c in comps])
    return np.clip(x, low, high)

# -------------------- 1) 基础设置 --------------------
rng = np.random.default_rng(42)   # 可复现随机数种子
N = 120                           # 每组样本数（越大 violin 越平滑）
dataset = "CAR" #CAR #SHAPE
# -------------------- 2) 生成 1–6 组合成数据 --------------------
if dataset== "SAHPE":
    #SHAPE
    data = {
        1: mixture_truncated_norm(N, [0.12, 0.24, 0.38], [0.05, 0.05, 0.06],
                                [0.35, 0.45, 0.20], rng=rng),
        2: mixture_truncated_norm(N, [0.05, 0.10, 0.20], [0.02, 0.04, 0.05],
                                [0.45, 0.40, 0.15], rng=rng),
        3: mixture_truncated_norm(N, [0.04, 0.09, 0.15], [0.02, 0.035, 0.04],
                                [0.35, 0.50, 0.15], rng=rng),
        4: mixture_truncated_norm(N, [0.02, 0.05, 0.09], [0.015, 0.02, 0.025],
                                [0.35, 0.45, 0.20], rng=rng),
        5: mixture_truncated_norm(N, [0.08, 0.13, 0.16], [0.03, 0.03, 0.02],
                                [0.25, 0.55, 0.20], rng=rng),
        6: mixture_truncated_norm(N, [0.02, 0.04, 0.07], [0.012, 0.015, 0.02],
                                [0.35, 0.50, 0.15], rng=rng),
    }
    g7_obs = np.array([
        0.10242686571804786, 0.1507256824255085, 0.13707159746546196,
        0.13224043380045689, 0.17013604556749898, 0.056239113191369684,
        0.05484289490646849, 0.15222498958118577, 0.12948232461567968,
        0.138576322920161
    ], dtype=float)
    std7 = g7_obs.std(ddof=1)
    h7 = max(0.01, 1.06 * std7 * (len(g7_obs) ** (-1/5)))  # Silverman 带宽
    base7 = rng.choice(g7_obs, size=N, replace=True)
    noise7 = rng.normal(0, h7, size=N)
    g7_syn = np.clip(base7 + noise7, 0.02, 1.0)            # ⭐ 下限抬到 0.05
    data[7] = g7_syn

    # -------------------- 4) 第8组：来自你的样本，核自助扩展（保留低端尾部） --------------------
    g8_obs = np.array([
        0.06490023942960146, 0.038629540931518803, 0.06333703225821427,
        0.02370193039031325, 0.024733092180275654, 0.09198318749288703,
        0.08012033639882878, 0.029294682246758755, 0.05770183906186868,
        0.09252098742884686
    ], dtype=float)
    std8 = g8_obs.std(ddof=1)
    h8 = max(0.006, 1.06 * std8 * (len(g8_obs) ** (-1/5))) # 对小样本给更小下限带宽
    base8 = rng.choice(g8_obs, size=N, replace=True)
    noise8 = rng.normal(0, h8, size=N)
    g8_syn = np.clip(base8 + noise8, 0.01, 1.0)             # 保留低端尾部（含 ~0.02）
    data[8] = g8_syn

else:
    #CAR
    data = {
        # 1: 高而瘦、上侧有明显长尾，整体中位数 ~0.19
        1: mixture_truncated_norm(
            N,
            mus=[0.10, 0.19, 0.28],        # 上端从 0.36 改到 0.28
            sigmas=[0.030, 0.020, 0.025],  # 保持略窄
            weights=[0.20, 0.60, 0.20],    # 主体集中在中间
            rng=rng
        ),

        # 2: 比较紧、中心 ~0.095
        2: mixture_truncated_norm(
            N,
            mus=[0.075, 0.095, 0.120],
            sigmas=[0.015, 0.010, 0.012],
            weights=[0.25, 0.50, 0.25],
            rng=rng
        ),

        # 3: 比 2 略高、略有上尾，中心 ~0.11
        3: mixture_truncated_norm(
            N,
            mus=[0.085, 0.105, 0.130],
            sigmas=[0.015, 0.012, 0.015],
            weights=[0.20, 0.55, 0.25],
            rng=rng
        ),

        # 4: 更低更紧，中心 ~0.08
        4: mixture_truncated_norm(
            N,
            mus=[0.060, 0.080, 0.100],
            sigmas=[0.012, 0.010, 0.012],
            weights=[0.20, 0.60, 0.20],
            rng=rng
        ),

        # 5: 稍高，分布略宽，上尾更可见，中心 ~0.10
        5: mixture_truncated_norm(
            N,
            mus=[0.085, 0.105, 0.135],
            sigmas=[0.015, 0.012, 0.015],
            weights=[0.20, 0.50, 0.30],
            rng=rng
        ),

        # 6: 与 5 类似但整体更低更窄，中心 ~0.085
        6: mixture_truncated_norm(
            N,
            mus=[0.070, 0.085, 0.095],
            sigmas=[0.012, 0.010, 0.010],
            weights=[0.25, 0.50, 0.25],
            rng=rng
        ),
    }

    g7_obs = np.array([
        0.05849248380721478, 0.06674381642893959, 0.039056133097995764,
        0.057722645501780835, 0.026293892924095934, 0.04434943824830435,
        0.04279867179473155, 0.025883680270084915, 0.07387578499794031,
        0.015006054831411196
    ], dtype=float)

    std7 = g7_obs.std(ddof=1)
    h7 = max(0.006, 1.06 * std7 * (len(g7_obs) ** (-1/5)))   # Silverman 带宽，给个小下限
    base7 = rng.choice(g7_obs, size=N, replace=True)
    noise7 = rng.normal(0, h7, size=N)

    # 7号整体很低，这里仅把下限抬到 0.01，避免出现 0
    g7_syn = np.clip(base7 + noise7, 0.01, 0.9)
    data[7] = g7_syn

    # ---------- 第8组：维持“比7更低、可贴近0、稍微更散”的关系 ----------
    # 做法：以 7 号为参考，稍微下移一个小量 delta，并让方差略大 (alpha*h7)，下限允许到 0
    delta = 0.008   # 8号平均比7号低 ~0.008（可按需要微调 0.005~0.012）
    alpha = 1.10    # 8号比7号略更“散”，>1 稍宽；若想更紧，设为 0.9

    base8 = rng.choice(g7_obs, size=N, replace=True)        # 也可以改用 g7_syn 做基底，差异不大
    noise8 = rng.normal(0, alpha * h7, size=N)
    g8_syn = np.clip(base8 + noise8, 0.0, 0.9)       # 允许更低尾部到 0
    data[8] = g8_syn
    shift_low=0.05
    shift_4=0.01
    data[4] = np.clip(data[4] + shift_4, 0.0, 1.0)
    data[7] = np.clip(data[7] + shift_low, 0.0, 1.0)
    data[8] = np.clip(data[8] + shift_low, 0.0, 1.0)

# -------------------- 3) 第7组：来自你的样本，核自助扩展（下限≥0.05） --------------------









# -------------------- 5) 保存合并数据 --------------------
rows = [{"Model": k, "MIG": float(v)} for k, vals in data.items() for v in vals]
df = pd.DataFrame(rows)
df.to_csv("synthetic_mig_violin_with_g7_g8.csv", index=False)
print("CSV saved to: synthetic_mig_violin_with_g7_g8.csv")

# -------------------- 6) 绘制 1–8 组小提琴图 --------------------
sns.set_theme(style="whitegrid", rc={"figure.dpi": 300})
viridis = get_cmap("viridis")
palette = [viridis(i/8) for i in range(8)]

fig, ax = plt.subplots(figsize=(6, 4.5))  # 单图；dpi 由 seaborn rc 控制为 300
parts = ax.violinplot([data[i] for i in range(1, 9)],
                      showmeans=False, showmedians=False, showextrema=False)

# 给每个 violin 上 viridis 颜色
for i, b in enumerate(parts['bodies'], start=1):
    b.set_facecolor(palette[i-1])
    b.set_edgecolor("black")
    b.set_alpha(0.9)
    b.set_linewidth(1.0)

# 中位数（白点 + 白线）
medians = [np.median(data[i]) for i in range(1, 9)]
for i, m in enumerate(medians, start=1):
    ax.scatter(i, m, s=20, c="white", zorder=3)
    ax.plot([i-0.22, i+0.22], [m, m], lw=2, c="white", zorder=2)

# 坐标轴与图例

ax.set_xticks(list(range(1, 9)))
ax.set_xticklabels([str(i) for i in range(1, 9)], fontsize=16)
ax.tick_params(axis="y", labelsize=14)

labels = [
    "β-STCVAE", "β-TCVAE", "β-TCVAE", "βVAE",
    "FactorVAE", "DIP-I", "VQVAE", "QLVAE"
]
handles = [plt.Line2D([], [], marker="s", color=palette[i], linestyle="") for i in range(8)]
legend = ax.legend(handles, labels, loc="upper right", frameon=True,
                   fontsize=10, title="Methods")
legend.get_title().set_fontsize(11)

plt.tight_layout()
plt.savefig("violin_mig_viridis_with_"+dataset+".svg", format="svg", dpi=300, bbox_inches="tight")
plt.show()
#print("SVG saved to: violin_mig_viridis_with_g7_g8.svg")