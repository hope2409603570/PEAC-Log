"""Log parsing metrics using the project's normalized-template convention.

FTA follows the counting rule in LogHub-2.0's template_level_analysis.py:
a predicted template counts when every assigned message has that same oracle
template. Exact equality of complete message groups is required only by GA/FGA.
Normalization is project-specific, so scores are not strict raw-token scores.
"""
import argparse
from datetime import datetime
from pathlib import Path
import re

import pandas as pd

FTA_DEFINITION = "loghub2_template_purity_f1_over_normalized_templates"


def clean_template(template_str):
    if not isinstance(template_str, str):
        return ""
    t = re.sub(r'<(?:NUM|IP|HEX|.*?|\*)>', '*', template_str)
    t = re.sub(r'[^a-zA-Z0-9*]', '', t)
    return t.lower()


def _read_rows(path):
    df = pd.read_csv(path, usecols=['LineId', 'EventTemplate'],
                     dtype={'LineId': str, 'EventTemplate': str}, keep_default_na=False)
    if df.empty:
        raise ValueError(f"{path}: cannot evaluate empty input")
    if df['LineId'].str.strip().eq('').any():
        raise ValueError(f"{path}: empty LineId")
    if df['LineId'].duplicated().any():
        raise ValueError(f"{path}: duplicate LineId")
    if df['EventTemplate'].str.strip().eq('').any():
        raise ValueError(f"{path}: empty EventTemplate")
    return df.set_index('LineId')['EventTemplate'].apply(clean_template)


def evaluate_metrics(gt_path, pred_path, dataset_name="Proxifier", runtime=0.0, output_csv_path=None):
    """Align by LineId, persist a summary row, and return (PA, GA, FGA, FTA)."""
    gt = _read_rows(gt_path)
    pred = _read_rows(pred_path)
    missing = gt.index.difference(pred.index)
    extra = pred.index.difference(gt.index)
    if len(missing) or len(extra):
        raise ValueError(f"LineId mismatch: missing={len(missing)}, extra={len(extra)}")
    df_eval = pd.DataFrame({'GT_Norm': gt, 'Pred_Norm': pred.reindex(gt.index)})
    total_logs = len(df_eval)
    PA = float((df_eval['GT_Norm'] == df_eval['Pred_Norm']).mean())
    gt_groups = df_eval.groupby('GT_Norm').groups
    pred_groups = df_eval.groupby('Pred_Norm').groups
    gt_sets = {frozenset(ids) for ids in gt_groups.values()}
    matched = [ids for ids in pred_groups.values() if frozenset(ids) in gt_sets]
    correct_g_temps = len(matched)
    GA = sum(len(ids) for ids in matched) / total_logs
    num_pred, num_gt = len(pred_groups), len(gt_groups)
    FGA = 2 * correct_g_temps / (num_pred + num_gt)

    pair_counts = df_eval.groupby(['GT_Norm', 'Pred_Norm']).size()
    correct_t_temps = sum(
        pair_counts.get((label, label), 0) == len(ids)
        for label, ids in pred_groups.items()
    )
    PTA = correct_t_temps / num_pred
    RTA = correct_t_temps / num_gt
    FTA = 2 * correct_t_temps / (num_pred + num_gt)
    new_result = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Dataset": dataset_name, "TotalLogs": total_logs,
        "GT_Clusters": num_gt, "Pred_Clusters": num_pred,
        "Matched_Clusters": correct_g_temps, "Matched_Templates": correct_t_temps,
        "PA": round(PA, 4), "GA": round(GA, 4), "FGA": round(FGA, 4),
        "PTA": round(PTA, 4), "RTA": round(RTA, 4), "FTA": round(FTA, 4),
        "FTA_definition": FTA_DEFINITION,
        "Runtime(s)": round(runtime, 4),
        "EPS(Logs/s)": round(total_logs / runtime, 2) if runtime > 0 else 0
    }
    summary_path = Path(output_csv_path) if output_csv_path else Path(__file__).resolve().parents[1] / 'result' / 'all_benchmark_results.csv'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([new_result])
    if summary_path.exists():
        df_old = pd.read_csv(summary_path)
        df_old = df_old[df_old['Dataset'] != dataset_name]
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    # Missing FTA in historical rows stays blank; aggregate scores cannot recover it.
    df_new.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"\n{dataset_name}: PA={PA:.4f} | GA={GA:.4f} | FGA={FGA:.4f} | FTA={FTA:.4f}")
    print(f"Runtime: {runtime:.4f}s | EPS: {new_result['EPS(Logs/s)']}")
    return PA, GA, FGA, FTA


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate existing predictions without rerunning parsing")
    parser.add_argument('--gt', default=root / 'data/Proxifier/Proxifier_full.log_structured.csv')
    parser.add_argument('--pred', default=root / 'result/proxifier/proxifier_predictions.csv')
    parser.add_argument('--dataset', default='Proxifier')
    parser.add_argument('--runtime', type=float, default=0.0)
    parser.add_argument('--output')
    args = parser.parse_args()
    evaluate_metrics(args.gt, args.pred, args.dataset, args.runtime, args.output)
