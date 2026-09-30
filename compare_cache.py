import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
df_hash = pd.read_csv(PROJECT_ROOT / 'result' / 'all_benchmark_results.csv')
df_tree = pd.read_csv(PROJECT_ROOT / 'result_ablation' / 'ablation_treecache_results.csv')

# 提取数据集名称（去掉后缀）
df_hash['Dataset_clean'] = df_hash['Dataset'].str.replace('_TreeCache', '', regex=False)
df_tree['Dataset_clean'] = df_tree['Dataset'].str.replace('_TreeCache', '', regex=False)

# 合并两个 DataFrame
merged = pd.merge(df_hash, df_tree, on='Dataset_clean', suffixes=('_Hash', '_Tree'))

print('Hash vs Tree 缓存逐数据集对比')
print('=' * 80)
print(f'{"Dataset":15s} {"PA_Hash":>8s} {"PA_Tree":>8s} {"PA_Diff":>8s} {"GA_Hash":>8s} {"GA_Tree":>8s} {"GA_Diff":>8s}')
print('-' * 80)

for _, row in merged.iterrows():
    pa_diff = row['PA_Hash'] - row['PA_Tree']
    ga_diff = row['GA_Hash'] - row['GA_Tree']
    print(f'{row["Dataset_clean"]:15s} {row["PA_Hash"]:8.4f} {row["PA_Tree"]:8.4f} {pa_diff:+8.4f} {row["GA_Hash"]:8.4f} {row["GA_Tree"]:8.4f} {ga_diff:+8.4f}')

print('=' * 80)
print()

# 统计 Hash 更优的数据集数量
pa_hash_win = (merged['PA_Hash'] > merged['PA_Tree']).sum()
ga_hash_win = (merged['GA_Hash'] > merged['GA_Tree']).sum()
print(f'PA Hash 更优的数据集: {pa_hash_win}/14')
print(f'GA Hash 更优的数据集: {ga_hash_win}/14')
print()

# 平均差异
print(f'平均 PA 差异: {(merged["PA_Hash"] - merged["PA_Tree"]).mean():+.4f} (Hash 更优)')
print(f'平均 GA 差异: {(merged["GA_Hash"] - merged["GA_Tree"]).mean():+.4f} (Hash 更优)')
