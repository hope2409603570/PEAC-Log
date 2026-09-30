import sys
import os
import csv
import time
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from main import PEACLogPipeline, preprocess_logs
from src.pipeline_config import get_pipeline_parameters
from src.stage1_preprocessing import SmartPreprocessor
from src.stage2_lsh import AdaptiveLSHClusterer
from src.stage3_4_extraction import FastTemplateExtractor
from utils.evaluate import evaluate_metrics


def run_sensitivity(dataset_name, input_path, entropy_values, output_csv):
    """
    对指定数据集，扫描不同的 entropy_threshold，记录 PA 和 GA
    """
    print(f"\n{'='*60}")
    print(f" Sensitivity Analysis: {dataset_name}")
    print(f" entropy_threshold range: {entropy_values}")
    print(f"{'='*60}")

    # 加载数据
    line_ids, contents = [], []
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_ids.append(row["LineId"])
            contents.append(row["Content"].strip())
    total_logs = len(contents)
    print(f"[OK] Loaded {total_logs} logs.")

    # Preprocessing rules and dataset parameters are both configuration-driven.
    supported_datasets = {"hdfs", "bgl", "spark"}
    if dataset_name.lower() not in supported_datasets:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    preprocessed = preprocess_logs(dataset_name, contents)
    parameters = get_pipeline_parameters(dataset_name)
    min_sim = parameters.min_similarity
    max_pos = parameters.max_check_positions

    # 断点续跑：读取已有结果
    completed_ets = set()
    results = []
    if os.path.exists(output_csv):
        with open(output_csv, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["Dataset"] == dataset_name and row.get("FTA", "").strip():
                    completed_ets.add(float(row["EntropyThreshold"]))
                    results.append({key: row[key] if key == "Dataset" else float(row[key]) for key in row})
        print(f"[INFO] Found existing results, skipping {len(completed_ets)} completed thresholds.")

    for et in entropy_values:
        if et in completed_ets:
            print(f"\n>>> entropy_threshold={et} already done, skipping.")
            continue

        print(f"\n>>> Testing entropy_threshold={et}")
        start_time = time.time()

        # 构造各阶段组件
        preprocessor = SmartPreprocessor(entropy_threshold=et, max_check_positions=max_pos)
        clusterer = AdaptiveLSHClusterer(min_similarity=min_sim)
        extractor = FastTemplateExtractor(sample_size=50)

        # 阶段一：分桶（这是我们要测试的变量）
        initial_buckets = preprocessor.fit_and_bucket(preprocessed)
        print(f"  Buckets: {len(initial_buckets)}")

        # ==========================================================
        # 阶段二~四：聚类 + 模板提取（大数据集启用缓存加速）
        # ==========================================================
        if total_logs > 50000:
            print(f"  [INFO] Large dataset ({total_logs}), activating cache accelerator...")

            # 将分桶后的数据扁平化，准备采样
            all_bucket_items = []
            for bucket_key, log_items in initial_buckets.items():
                all_bucket_items.extend(log_items)

            # Top-K 高频采样（基于分桶后的 tokens）
            freq_counter = Counter(item['raw_log'] for item in all_bucket_items)
            unique_count = len(freq_counter)
            top_k = min(8000, unique_count)
            train_logs = [item[0] for item in freq_counter.most_common(top_k)]
            print(f"  Training on top {top_k} patterns...")

            # 对采样数据执行完整聚类+模板提取
            train_tokens = [log.strip().split() for log in train_logs]
            train_bucket = [{
                "original_index": idx,
                "raw_log": train_logs[idx],
                "tokens": tokens
            } for idx, tokens in enumerate(train_tokens)]

            train_clusters = clusterer.cluster_bucket(train_bucket)
            train_results = extractor.extract_templates(train_clusters)

            # 后处理：生成 EventId
            unique_templates_map = {}
            cluster_to_event = {}
            event_to_template = {}
            event_counter = 1

            sorted_res = sorted(train_results, key=lambda x: x['log_count'], reverse=True)
            for res in sorted_res:
                norm_key = res['template'].replace(" ", "").lower()
                if norm_key not in unique_templates_map:
                    e_id = f"E{event_counter}"
                    unique_templates_map[norm_key] = e_id
                    event_to_template[e_id] = res['template']
                    event_counter += 1
                cluster_to_event[res['cluster_id']] = unique_templates_map[norm_key]

            # 构建 Hash 缓存
            inference_cache = {}
            for c_idx, cluster in enumerate(train_clusters):
                cid = c_idx + 1
                e_id = cluster_to_event.get(cid, "E99")
                for item in cluster:
                    inference_cache[item['raw_log']] = e_id

            # 预计算模板信息（用于未命中回退）
            template_info = {}
            for temp_e_id, temp_str in event_to_template.items():
                t_set = set(temp_str.split())
                template_info[temp_e_id] = (t_set, len(t_set))

            # 全量映射
            output_rows = []
            for i in range(total_logs):
                p_str = preprocessed[i]

                # 缓存命中
                if p_str in inference_cache:
                    e_id = inference_cache[p_str]
                else:
                    # Jaccard 回退匹配
                    best_e, best_score = "E99", 0
                    p_set = set(p_str.split())
                    p_len = len(p_set)
                    for temp_e_id, (t_set, t_len) in template_info.items():
                        if abs(p_len - t_len) > 12:
                            continue
                        intersect_len = len(p_set & t_set)
                        union_len = p_len + t_len - intersect_len
                        score = intersect_len / union_len if union_len > 0 else 0
                        if score > best_score:
                            best_score = score
                            best_e = temp_e_id
                            if score > 0.85:
                                break
                    e_id = best_e

                output_rows.append({
                    "LineId": line_ids[i],
                    "Content": contents[i],
                    "EventId": e_id,
                    "EventTemplate": event_to_template.get(e_id, "<*>")
                })
            output_rows.sort(key=lambda x: int(x["LineId"]))

            all_refined_clusters = train_clusters  # 用于统计聚类数

        else:
            # 小数据集：直接全量聚类
            all_refined_clusters = []
            for log_items in initial_buckets.values():
                if not log_items:
                    continue
                refined_clusters = clusterer.cluster_bucket(log_items)
                all_refined_clusters.extend(refined_clusters)
            print(f"  Clusters: {len(all_refined_clusters)}")

            template_results = extractor.extract_templates(all_refined_clusters)

            unique_templates_map = {}
            cluster_to_event = {}
            event_to_template = {}
            event_counter = 1

            sorted_res = sorted(template_results, key=lambda x: x['log_count'], reverse=True)
            for res in sorted_res:
                norm_key = res['template'].replace(" ", "").lower()
                if norm_key not in unique_templates_map:
                    e_id = f"E{event_counter}"
                    unique_templates_map[norm_key] = e_id
                    event_to_template[e_id] = res['template']
                    event_counter += 1
                cluster_to_event[res['cluster_id']] = unique_templates_map[norm_key]

            output_rows = []
            for c_idx, cluster in enumerate(all_refined_clusters):
                cid = c_idx + 1
                e_id = cluster_to_event.get(cid, "E99")
                for item in cluster:
                    orig_idx = item['original_index']
                    output_rows.append({
                        "LineId": line_ids[orig_idx],
                        "Content": contents[orig_idx],
                        "EventId": e_id,
                        "EventTemplate": event_to_template.get(e_id, "<*>")
                    })
            output_rows.sort(key=lambda x: int(x["LineId"]))

        # 保存临时预测文件用于评估
        tmp_pred = os.path.join(current_dir, "result", f"sensitivity_{dataset_name.lower()}_et{et}.csv")
        os.makedirs(os.path.dirname(tmp_pred), exist_ok=True)
        with open(tmp_pred, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["LineId", "Content", "EventId", "EventTemplate"])
            writer.writeheader()
            writer.writerows(output_rows)

        # 评估
        gt_path = input_path
        pa, ga, fga, fta = evaluate_metrics(gt_path, tmp_pred, dataset_name, runtime=0.0, output_csv_path=tmp_pred.replace(".csv", "_metrics.csv"))

        runtime = time.time() - start_time
        eps = total_logs / runtime if runtime > 0 else 0

        result_row = {
            "Dataset": dataset_name,
            "EntropyThreshold": et,
            "Buckets": len(initial_buckets),
            "Clusters": len(all_refined_clusters),
            "PA": pa,
            "GA": ga,
            "FGA": fga,
            "FTA": fta,
            "Runtime": runtime,
            "EPS": eps
        }
        results.append(result_row)
        print(f"  PA={pa:.4f} GA={ga:.4f} FGA={fga:.4f} FTA={fta:.4f} Runtime={runtime:.2f}s")

        # 每完成一个点立即追加保存（断点续跑）
        fields = ["Dataset", "EntropyThreshold", "Buckets", "Clusters", "PA", "GA", "FGA", "FTA", "Runtime", "EPS"]
        old_rows = []
        if os.path.exists(output_csv):
            with open(output_csv, "r", newline="", encoding="utf-8-sig") as f:
                old_rows = [row for row in csv.DictReader(f)
                            if not (row["Dataset"] == dataset_name and float(row["EntropyThreshold"]) == et)]
        with open(output_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerow(result_row)

    print(f"\n[OK] Results saved to {output_csv}")
    return results


def plot_sensitivity(all_results, output_path):
    """绘制敏感性曲线图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"HDFS": "#4477aa", "BGL": "#cc3333", "Spark": "#228822"}
    markers = {"HDFS": "o", "BGL": "s", "Spark": "^"}

    for ds_name, results in all_results.items():
        ets = [r["EntropyThreshold"] for r in results]
        pas = [r["PA"] for r in results]
        gas = [r["GA"] for r in results]

        ax1.plot(ets, pas, marker=markers[ds_name], color=colors[ds_name], label=ds_name, linewidth=2, markersize=8)
        ax2.plot(ets, gas, marker=markers[ds_name], color=colors[ds_name], label=ds_name, linewidth=2, markersize=8)

    for ax, title in [(ax1, "PA (Parsing Accuracy)"), (ax2, "GA (Grouping Accuracy)")]:
        ax.set_xlabel("Entropy Threshold", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"[OK] Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    base_dir = current_dir
    entropy_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    datasets = [
        ("HDFS", os.path.join(base_dir, "data", "HDFS", "HDFS_full.log_structured.csv")),
        ("BGL", os.path.join(base_dir, "data", "BGL", "BGL_full.log_structured.csv")),
        ("Spark", os.path.join(base_dir, "data", "Spark", "Spark_full.log_structured.csv")),
    ]

    all_results = {}
    for ds_name, input_path in datasets:
        output_csv = os.path.join(base_dir, "result", f"sensitivity_entropy_{ds_name.lower()}.csv")
        results = run_sensitivity(ds_name, input_path, entropy_values, output_csv)
        all_results[ds_name] = results

    # 汇总 CSV
    summary_csv = os.path.join(base_dir, "result", "sensitivity_entropy_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["Dataset", "EntropyThreshold", "Buckets", "Clusters", "PA", "GA", "FGA", "FTA", "Runtime", "EPS"])
        writer.writeheader()
        for ds_name, results in all_results.items():
            writer.writerows(results)
    print(f"\n[OK] Summary saved to {summary_csv}")

    # 绘图
    plot_path = os.path.join(base_dir, "result", "sensitivity_entropy_plot.png")
    plot_sensitivity(all_results, plot_path)
