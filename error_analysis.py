import pandas as pd
import re
import os

def clean_template(template_str):
    """和 evaluate.py 保持完全一致的清洗逻辑，确保对比公平"""
    if not isinstance(template_str, str): return ""
    t = re.sub(r'<(?:NUM|IP|HEX|.*?|\*)>', '*', template_str)
    t = re.sub(r'[^a-zA-Z0-9*]', '', t)
    return t.lower()

def run_error_analysis(dataset_name="Linux"):
    print("="*60)
    print(f"🔍 启动 {dataset_name} 数据集 Bad Case 错误分析...")
    print("="*60)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    gt_path = os.path.join(base_dir, "data", dataset_name, f"{dataset_name}_full.log_structured.csv")
    pred_path = os.path.join(base_dir, "result", dataset_name.lower(), f"{dataset_name.lower()}_predictions.csv")
    out_err_path = os.path.join(base_dir, "result", dataset_name.lower(), f"{dataset_name.lower()}_bad_cases.csv")

    if not os.path.exists(gt_path) or not os.path.exists(pred_path):
        print(f"❌ 找不到输入文件或预测结果文件，请确保 {dataset_name} 的 benchmark 已运行！")
        return

    # 1. 读取数据并按 LineId 对齐 (尝试读取 EventId，如果 GT 里没有就用 Template 兜底)
    try:
        df_gt = pd.read_csv(gt_path)[['LineId', 'Content', 'EventId', 'EventTemplate']]
    except KeyError:
        df_gt = pd.read_csv(gt_path)[['LineId', 'Content', 'EventTemplate']]
        df_gt['EventId'] = df_gt['EventTemplate'] 
        
    df_pred = pd.read_csv(pred_path)[['LineId', 'EventId', 'EventTemplate']]
    
    # 重命名列以便区分
    df_gt.rename(columns={'EventTemplate': 'GT_Template', 'EventId': 'GT_EventId'}, inplace=True)
    df_pred.rename(columns={'EventTemplate': 'Pred_Template', 'EventId': 'Pred_EventId'}, inplace=True)

    # 合并数据
    df_merged = pd.merge(df_gt, df_pred, on='LineId')

    # ==========================================
    # 模块 A: PA (解析准确率) 错误分析
    # ==========================================
    df_merged['GT_Norm'] = df_merged['GT_Template'].apply(clean_template)
    df_merged['Pred_Norm'] = df_merged['Pred_Template'].apply(clean_template)

    errors_df = df_merged[df_merged['GT_Norm'] != df_merged['Pred_Norm']].copy()
    
    error_count = len(errors_df)
    total_count = len(df_merged)
    error_rate = error_count / total_count if total_count > 0 else 0

    print(f"\n📊 [PA 分析] 模板生成准确率概览:")
    print(f"   - 总日志数: {total_count}")
    print(f"   - 错误日志数: {error_count}")
    print(f"   - 当前错误率: {error_rate:.2%} (对应 PA: {1-error_rate:.4f})")

    if error_count > 0:
        error_summary = errors_df.groupby(['GT_Template', 'Pred_Template']).agg(
            ErrorCount=('LineId', 'count'),
            SampleContent=('Content', 'first') 
        ).reset_index().sort_values(by='ErrorCount', ascending=False)
        
        error_summary.to_csv(out_err_path, index=False, encoding='utf-8-sig')
        print(f"📂 完整的 PA 错误分析报告已保存至: {out_err_path}")

        print("\n🔥 Top 3 最严重的模板生成错误 🔥")
        for idx, row in error_summary.head(3).iterrows():
            print(f"🔴 错误频次: {row['ErrorCount']} 次")
            print(f"   📝 原始日志 : {row['SampleContent']}")
            print(f"   ✅ 官方答案 : {row['GT_Template']}")
            print(f"   ❌ 你的预测 : {row['Pred_Template']}")
            print("-" * 60)
    else:
        print("🎉 恭喜！没有任何 PA 解析错误！")

    # ==========================================
    # 模块 B: GA (聚类准确率) 错误分析
    # ==========================================
    print(f"\n📊 [GA 分析] 聚类逻辑错误诊断:")
    
    # 1. 查找过度合并 (Over-merge): 一个 Pred_EventId 包含了多个 GT_EventId
    over_merge_df = df_merged.groupby('Pred_EventId')['GT_EventId'].nunique().reset_index()
    over_merged_preds = over_merge_df[over_merge_df['GT_EventId'] > 1]
    
    # 2. 查找过度拆分 (Over-split): 一个 GT_EventId 被拆成了多个 Pred_EventId
    over_split_df = df_merged.groupby('GT_EventId')['Pred_EventId'].nunique().reset_index()
    over_split_gts = over_split_df[over_split_df['Pred_EventId'] > 1]
    
    print(f"   - 发现过度合并 (Over-merge) 的类簇数量: {len(over_merged_preds)}")
    print(f"   - 发现过度拆分 (Over-split) 的类簇数量: {len(over_split_gts)}")

    if len(over_merged_preds) > 0:
        print("\n🚨 最严重的【过度合并】案例 (多个不同模板被强行塞进了一个桶):")
        worst_merge = over_merged_preds.sort_values(by='GT_EventId', ascending=False).iloc[0]
        worst_pred_id = worst_merge['Pred_EventId']
        involved_gts = df_merged[df_merged['Pred_EventId'] == worst_pred_id]['GT_Template'].unique()
        pred_temp = df_merged[df_merged['Pred_EventId'] == worst_pred_id]['Pred_Template'].iloc[0]
        
        print(f"   ❌ 你的类簇 [{worst_pred_id}] 模板: {pred_temp}")
        print(f"   ⚠️ 它错误地吞噬了以下 {len(involved_gts)} 种官方模板:")
        for gt in involved_gts[:3]: 
            print(f"      - {gt}")
        if len(involved_gts) > 3: print("      - ... (更多)")

    if len(over_split_gts) > 0:
        print("\n🔪 最严重的【过度拆分】案例 (本该是一类的日志被你拆碎了):")
        worst_split = over_split_gts.sort_values(by='Pred_EventId', ascending=False).iloc[0]
        worst_gt_id = worst_split['GT_EventId']
        involved_preds = df_merged[df_merged['GT_EventId'] == worst_gt_id]['Pred_Template'].unique()
        gt_temp = df_merged[df_merged['GT_EventId'] == worst_gt_id]['GT_Template'].iloc[0]
        
        print(f"   ✅ 官方类簇 [{worst_gt_id}] 模板: {gt_temp}")
        print(f"   ⚠️ 被你错误地切碎成了以下 {len(involved_preds)} 种模板:")
        for pt in involved_preds[:3]:
            print(f"      - {pt}")
        if len(involved_preds) > 3: print("      - ... (更多)")
        print("="*60)

if __name__ == "__main__":
    run_error_analysis("Spark")
