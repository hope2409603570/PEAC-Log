import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

# 读取数据
PROJECT_ROOT = Path(__file__).resolve().parent
RESULT_DIR = PROJECT_ROOT / "result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
hdfs = pd.read_csv(RESULT_DIR / "sensitivity_entropy_hdfs.csv")
bgl = pd.read_csv(RESULT_DIR / "sensitivity_entropy_bgl.csv")

# 检查 Spark 结果是否存在
spark_path = RESULT_DIR / "sensitivity_entropy_spark.csv"
has_spark = spark_path.exists()

if has_spark:
    spark = pd.read_csv(spark_path)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    datasets = [
        (axes[0], hdfs, '(a) HDFS', '#4477aa'),
        (axes[1], bgl, '(b) BGL', '#228822'),
        (axes[2], spark, '(c) Spark', '#cc8822'),
    ]
else:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    datasets = [
        (axes[0], hdfs, '(a) HDFS', '#4477aa'),
        (axes[1], bgl, '(b) BGL', '#228822'),
    ]

def plot_dataset(ax, df, title, color):
    ets = df['EntropyThreshold'].values
    buckets = df['Buckets'].values
    runtime = df['Runtime'].values

    # 左轴：分桶数量（柱状图）
    bars = ax.bar(ets - 0.08, buckets, width=0.15, color=color, alpha=0.7, label='Buckets')
    ax.set_ylabel('Number of Buckets', fontsize=15, color=color)
    ax.tick_params(axis='y', labelcolor=color, labelsize=13)
    ax.set_yscale('log')

    # 右轴：Runtime（折线图）
    ax2 = ax.twinx()
    line = ax2.plot(ets, runtime, color='#cc3333', marker='o', linewidth=2, markersize=7, label='Runtime (s)')
    ax2.set_ylabel('Runtime (s)', fontsize=15, color='#cc3333')
    ax2.tick_params(axis='y', labelcolor='#cc3333', labelsize=13)

    ax.set_xlabel('Entropy Threshold', fontsize=15)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.tick_params(axis='x', labelsize=13)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)

    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

for ax, df, title, color in datasets:
    plot_dataset(ax, df, title, color)

plt.tight_layout()
plt.savefig(RESULT_DIR / "sensitivity_entropy_efficiency.png", dpi=300, bbox_inches="tight")
plt.close()

print(f"[OK] Plot saved (with Spark={has_spark})")
