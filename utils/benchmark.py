import sys
import os
import csv
import time
from collections import Counter

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Dataset rules and dispatch are defined in config/preprocessing_rules.json.
from main import PEACLogPipeline, preprocess_logs
from utils.evaluate import evaluate_metrics

def jaccard_similarity(s1, s2):
    """简单的 Jaccard 相似度，用于千万级数据推理时的兜底匹配"""
    set1, set2 = set(s1.split()), set(s2.split())
    if not set1 or not set2: return 0.0
    return len(set1 & set2) / len(set1 | set2)

def run_benchmark():
    base_dir = project_root
    
    # 包含了 Loghub 的所有 12 个数据集
    datasets = [
        {
            "name": "Proxifier",
            "input_path": os.path.join(base_dir, "data", "Proxifier", "Proxifier_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "proxifier")
        },
        {
            "name": "Linux",
            "input_path": os.path.join(base_dir, "data", "Linux", "Linux_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "linux")
        },
        {
            "name": "HDFS",
            "input_path": os.path.join(base_dir, "data", "HDFS", "HDFS_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "hdfs")
        },
        {
            "name": "Mac",
            "input_path": os.path.join(base_dir, "data", "Mac", "Mac_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "mac")
        },
        {
            "name": "Apache",
            "input_path": os.path.join(base_dir, "data", "Apache", "Apache_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "apache")
        },
        {
            "name": "Hadoop",
            "input_path": os.path.join(base_dir, "data", "Hadoop", "Hadoop_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "hadoop")
        },
        {   
            "name": "BGL",
            "input_path": os.path.join(base_dir, "data", "BGL", "BGL_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "bgl")
        },
        {   
            "name": "HealthApp",
            "input_path": os.path.join(base_dir, "data", "HealthApp", "HealthApp_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "healthapp")
        },
        {   
            "name": "HPC",
            "input_path": os.path.join(base_dir, "data", "HPC", "HPC_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "hpc")
        },
        {   
            "name": "OpenSSH",
            "input_path": os.path.join(base_dir, "data", "OpenSSH", "OpenSSH_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "openssh")
        },
        {
            "name": "OpenStack",
            "input_path": os.path.join(base_dir, "data", "OpenStack", "OpenStack_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "openstack")
        },
        {   # Thunderbird dataset
            "name": "Thunderbird",
            "input_path": os.path.join(base_dir, "data", "Thunderbird", "Thunderbird_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "thunderbird")
        },
        {   
            "name": "Zookeeper",
            "input_path": os.path.join(base_dir, "data", "Zookeeper", "Zookeeper_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "zookeeper")
        },
        {   # Spark dataset
            "name": "Spark",
            "input_path": os.path.join(base_dir, "data", "Spark", "Spark_full.log_structured.csv"),
            "output_dir": os.path.join(base_dir, "result", "spark")
        }
    ]

    print("="*60)
    print("Starting Automated Benchmark Pipeline")
    print("="*60)

    for ds in datasets:
        dataset_name = ds["name"]
        input_path = ds["input_path"]
        output_dir = ds["output_dir"]
        output_path = os.path.join(output_dir, f"{dataset_name.lower()}_predictions.csv")

        print(f"\n" + "="*50)
        print(f" Processing dataset: {dataset_name}")
        print("="*50)

        if not os.path.exists(input_path):
            print(f"[ERROR] Input file not found: {input_path}")
            continue

        os.makedirs(output_dir, exist_ok=True)

        line_ids, contents = [], []
        if total_logs_hint := (dataset_name.lower() in ["hdfs", "mac", "hadoop", "bgl", "healthapp", "hpc", "openssh", "openstack", "spark", "thunderbird", "zookeeper"]):
            print(f"[INFO] Loading {dataset_name} full dataset into memory, please wait...")

        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Load full dataset without truncation
            for row in reader:
                line_ids.append(row["LineId"])
                contents.append(row["Content"].strip())
        
        total_logs = len(contents)
        print(f"[OK] Data loaded, total {total_logs} logs.")

        print("[INFO] Preprocessing and cleaning...")
        preprocessed = preprocess_logs(dataset_name, contents)

        pipeline = PEACLogPipeline(dataset_name=dataset_name)
        
        # ==================================================
        # 缓存加速分支 (长度剪枝 + O(1) 短路)
        # ==================================================
        if total_logs > 50000:
            print(f"\n{'='*60}")
            print(f"PEAC-Log Benchmark - Cache Acceleration ENABLED")
            print(f"{'='*60}")

            print(f"[INFO] Large dataset detected ({total_logs:,} logs).")
            print(f"[INFO] Activating cache acceleration...")

            # 全局扫描 + Top-K 采样
            print("  Step 1: Global frequency analysis...")
            
            freq_counter = Counter(preprocessed)
            unique_count = len(freq_counter)
            print(f"  Global scan complete: {unique_count} unique patterns found.")
            
            top_k = min(8000, unique_count)
            train_data = [item[0] for item in freq_counter.most_common(top_k)]
            print(f"  Training on top {top_k} core patterns...")
            
            c_to_e, e_to_t, all_clusters, train_duration = pipeline.parse(train_data)
            
            print("  Building O(1) inference cache and precomputing indices...")
            fast_cache = {}
            for c_idx, cluster in enumerate(all_clusters):
                cid = c_idx + 1
                e_id = c_to_e.get(cid, "E99")
                for item in cluster:
                    orig_idx = item['original_index']
                    p_str = train_data[orig_idx]
                    fast_cache[p_str] = e_id
            
            # 预计算核心：提前算好所有模板的 set 和长度
            template_info = {}
            for temp_e_id, temp_str in e_to_t.items():
                t_set = set(temp_str.split())
                template_info[temp_e_id] = (t_set, len(t_set))
            
            print(f"  Mapping {total_logs} records and writing to output...")
            infer_start = time.time()
            
            with open(output_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["LineId", "Content", "EventId", "EventTemplate"])
                writer.writeheader()
                for i in range(total_logs):
                    p_str = preprocessed[i]
                    if p_str in fast_cache:
                        e_id = fast_cache[p_str]
                    else:
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
                        fast_cache[p_str] = e_id 
                        
                    writer.writerow({
                        "LineId": line_ids[i],
                        "Content": contents[i],
                        "EventId": e_id,
                        "EventTemplate": e_to_t.get(e_id, "<*>")
                    })
            
            duration = train_duration + (time.time() - infer_start)
            print(f"[OK] Processing complete! Total time: {duration:.4f}s")

        else:
            # 传统模式
            c_to_e, e_to_t, all_clusters, duration = pipeline.parse(preprocessed)
            final_rows = []
            for c_idx, cluster in enumerate(all_clusters):
                cid = c_idx + 1
                e_id = c_to_e.get(cid, "E99")
                e_temp = e_to_t.get(e_id, "<*>")
                for item in cluster:
                    orig_idx = item['original_index']
                    final_rows.append({
                        "LineId": line_ids[orig_idx],
                        "Content": contents[orig_idx],
                        "EventId": e_id,
                        "EventTemplate": e_temp
                    })
                    
            final_rows.sort(key=lambda x: int(x['LineId']))
            
            with open(output_path, mode="w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["LineId", "Content", "EventId", "EventTemplate"])
                writer.writeheader()
                writer.writerows(final_rows)
                
            print(f"[OK] Results saved to: {output_path}")

        # 计算得分
        try:
            evaluate_metrics(
                gt_path=input_path,
                pred_path=output_path,
                dataset_name=dataset_name,
                runtime=duration
            )
        except Exception as e:
            print(f"[WARN] Evaluation error: {e}")

    print("\n" + "="*60)
    print("Benchmark task completed!")
    print("="*60)

if __name__ == "__main__":
    run_benchmark()
