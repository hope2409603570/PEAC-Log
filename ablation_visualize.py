import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_ablation_results():
    """加载消融实验结果（从各变体的汇总 CSV 文件中读取）"""
    result_ablation_dir = os.path.join(current_dir, "result_ablation")
    
    if not os.path.exists(result_ablation_dir):
        print(f"[ERROR] Ablation results directory not found: {result_ablation_dir}")
        return None
    
    all_results = []
    
    # 查找 result_ablation 目录下的所有变体汇总 CSV 文件
    for filename in os.listdir(result_ablation_dir):
        if filename.startswith('ablation_') and filename.endswith('_results.csv'):
            csv_path = os.path.join(result_ablation_dir, filename)
            try:
                df = pd.read_csv(csv_path)
                all_results.append(df)
            except Exception as e:
                print(f"[WARN] Failed to read file {csv_path}: {e}")
    
    if not all_results:
        print("[WARN] No ablation results found. Please run ablation_study.py first.")
        return None
    
    # 合并所有结果
    ablation_df = pd.concat(all_results, ignore_index=True)
    
    # 分离数据集名称和变体名称（Dataset 字段格式: DatasetName_VariantName）
    ablation_df['DatasetName'] = ablation_df['Dataset'].str.split('_').str[0]
    ablation_df['Variant'] = ablation_df['Dataset'].str.split('_').str[1]
    
    return ablation_df


def plot_ablation_comparison(ablation_df, metric='FGA', save_path=None):
    """
    绘制消融实验对比柱状图
    
    Args:
        ablation_df: 消融实验结果 DataFrame
        metric: 要对比的指标 (PA, GA, FGA)
        save_path: 图片保存路径
    """
    variants = ablation_df['Variant'].unique()
    datasets = ablation_df['DatasetName'].unique()
    
    # 设置柱状图位置
    x = np.arange(len(datasets))
    width = 0.12
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = {
        'Baseline': '#2E86AB',
        'NoBucket': '#A23B72',
        'NoAdaptiveCluster': '#F18F01',
        'NoHierarchicalCluster': '#C73E1D',
        'NoSequenceAlign': '#95C623'
    }
    
    for i, variant in enumerate(variants):
        variant_data = []
        for dataset in datasets:
            value = ablation_df[
                (ablation_df['DatasetName'] == dataset) & 
                (ablation_df['Variant'] == variant)
            ][metric]
            variant_data.append(value.values[0] if not value.empty else 0)
        
        offset = (i - len(variants)/2) * width + width/2
        bars = ax.bar(x + offset, variant_data, width, 
                     label=variant, color=colors.get(variant, '#666666'),
                     edgecolor='white', linewidth=0.5)
        
        # 在柱子上标注数值
        for bar, val in zip(bars, variant_data):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{metric} Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Ablation Study: {metric} Comparison Across Variants', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend(title='Variant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Chart saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_ablation_heatmap(ablation_df, metric='FGA', save_path=None):
    """
    绘制消融实验热力图
    
    Args:
        ablation_df: 消融实验结果 DataFrame
        metric: 要对比的指标
        save_path: 图片保存路径
    """
    pivot_df = ablation_df.pivot_table(
        index='Variant', 
        columns='DatasetName', 
        values=metric,
        aggfunc='first'
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(pivot_df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # 设置坐标轴
    ax.set_xticks(np.arange(len(pivot_df.columns)))
    ax.set_yticks(np.arange(len(pivot_df.index)))
    ax.set_xticklabels(pivot_df.columns, rotation=45, ha='right')
    ax.set_yticklabels(pivot_df.index)
    
    # 在每个格子中显示数值
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            if not pd.isna(val):
                text = ax.text(j, i, f'{val:.3f}',
                             ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(f'Ablation Study Heatmap: {metric}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variant', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f'{metric} Score', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Heatmap saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_runtime_comparison(ablation_df, save_path=None):
    """绘制运行时间对比图"""
    pivot_df = ablation_df.pivot_table(
        index='Variant',
        columns='DatasetName',
        values='Runtime(s)',
        aggfunc='first'
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    pivot_df.T.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Runtime Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.legend(title='Variant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK] Runtime comparison chart saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def generate_ablation_report(ablation_df, save_path=None):
    """生成消融实验文字报告"""
    report = []
    report.append("=" * 80)
    report.append("PEAC-Log Ablation Study Analysis Report")
    report.append("=" * 80)
    report.append("")
    
    datasets = ablation_df['DatasetName'].unique()
    variants = ablation_df['Variant'].unique()
    
    for dataset in datasets:
        ds_df = ablation_df[ablation_df['DatasetName'] == dataset]
        report.append(f"\n{'='*80}")
        report.append(f"Dataset: {dataset}")
        report.append(f"{'='*80}")
        
        # 找到 baseline
        baseline = ds_df[ds_df['Variant'] == 'Baseline']
        if baseline.empty:
            continue
        
        baseline_fga = baseline['FGA'].values[0]
        baseline_pa = baseline['PA'].values[0]
        baseline_ga = baseline['GA'].values[0]
        
        report.append(f"\nBaseline Performance:")
        report.append(f"   PA: {baseline_pa:.4f} | GA: {baseline_ga:.4f} | FGA: {baseline_fga:.4f}")
        report.append(f"   Runtime: {baseline['Runtime(s)'].values[0]:.2f}s")
        report.append(f"   Clusters: GT={baseline['GT_Clusters'].values[0]}, Pred={baseline['Pred_Clusters'].values[0]}")
        
        report.append(f"\nFGA Drop Relative to Baseline:")
        
        for variant in variants:
            if variant == 'Baseline':
                continue
            var_df = ds_df[ds_df['Variant'] == variant]
            if var_df.empty:
                continue
            
            var_fga = var_df['FGA'].values[0]
            drop = baseline_fga - var_fga
            drop_pct = (drop / baseline_fga * 100) if baseline_fga > 0 else 0
            
            report.append(f"   {variant:15s}: FGA={var_fga:.4f} (drop={drop:.4f}, {drop_pct:5.1f}%)")
    
    report.append("\n" + "=" * 80)
    report.append("Overall Conclusion")
    report.append("=" * 80)
    
    # 计算每个变体在所有数据集上的平均表现
    report.append("\nAverage FGA Across All Datasets:")
    avg_scores = ablation_df.groupby('Variant')['FGA'].mean().sort_values(ascending=False)
    for variant, score in avg_scores.items():
        marker = "*" if variant == 'Baseline' else " "
        report.append(f"   [{marker}] {variant:15s}: {score:.4f}")
    
    report_text = "\n".join(report)
    print(report_text)
    
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n[OK] Report saved to: {save_path}")
    
    return report_text


def main():
    """主函数：生成所有可视化图表"""
    print("[INFO] Loading ablation results...")
    ablation_df = load_ablation_results()
    
    if ablation_df is None:
        return
    
    output_dir = os.path.join(current_dir, "result_ablation", "visualization")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[INFO] Found {len(ablation_df)} ablation records")
    print(f"   Datasets: {', '.join(ablation_df['DatasetName'].unique())}")
    print(f"   Variants: {', '.join(ablation_df['Variant'].unique())}")
    
    # 生成各种图表
    print("\n[INFO] Generating visualization charts...")
    
    plot_ablation_comparison(
        ablation_df, metric='FGA',
        save_path=os.path.join(output_dir, "ablation_fga_comparison.png")
    )
    
    plot_ablation_comparison(
        ablation_df, metric='PA',
        save_path=os.path.join(output_dir, "ablation_pa_comparison.png")
    )
    
    plot_ablation_comparison(
        ablation_df, metric='GA',
        save_path=os.path.join(output_dir, "ablation_ga_comparison.png")
    )
    
    plot_ablation_heatmap(
        ablation_df, metric='FGA',
        save_path=os.path.join(output_dir, "ablation_fga_heatmap.png")
    )
    
    plot_runtime_comparison(
        ablation_df,
        save_path=os.path.join(output_dir, "ablation_runtime_comparison.png")
    )
    
    # 生成文字报告
    generate_ablation_report(
        ablation_df,
        save_path=os.path.join(output_dir, "ablation_report.txt")
    )
    
    print(f"\n[OK] All visualization results saved to: {output_dir}")


if __name__ == "__main__":
    main()
