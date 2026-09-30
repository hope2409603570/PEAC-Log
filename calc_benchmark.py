import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
df = pd.read_csv(PROJECT_ROOT / 'result' / 'all_benchmark_results.csv')

print('PEAC-Log (完整方法) Benchmark 结果统计')
print('=' * 50)
print(f'平均 PA: {df["PA"].mean():.4f}')
print(f'平均 GA: {df["GA"].mean():.4f}')
print(f'平均 FGA: {df["FGA"].mean():.4f}')
if "FTA" in df.columns and df["FTA"].notna().any():
    valid_fta = df["FTA"].dropna()
    print(f'平均 FTA: {valid_fta.mean():.4f}（有效数据集 {len(valid_fta)}/{len(df)}）')
else:
    print('平均 FTA: 未评估，请使用预测文件重新评估；历史汇总不能推算 FTA。')
print(f'平均 EPS: {df["EPS(Logs/s)"].mean():.2f}')
print('=' * 50)
print()
print('各数据集详细结果:')
for _, row in df.iterrows():
    print(f'{row["Dataset"]:15s} PA={row["PA"]:.4f} GA={row["GA"]:.4f} FGA={row["FGA"]:.4f} FTA={row.get("FTA", float("nan")):.4f} EPS={row["EPS(Logs/s)"]:,.2f}')
