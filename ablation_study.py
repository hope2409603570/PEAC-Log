import sys
import os
import csv
import time
import re
import json
from collections import Counter
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入所有高精度清洗函数和流水线 (与 benchmark.py 完全一致)
from main import (
    loghub_standard_preprocess,
    linux_standard_preprocess,
    hdfs_standard_preprocess,
    mac_standard_preprocess,
    apache_standard_preprocess,
    hadoop_standard_preprocess,
    bgl_standard_preprocess,
    healthapp_standard_preprocess,
    hpc_standard_preprocess,
    openssh_standard_preprocess,
    openstack_standard_preprocess,
    thunderbird_standard_preprocess,
    zookeeper_standard_preprocess,
    spark_standard_preprocess
)
from utils.evaluate import evaluate_metrics

# ==========================================
# 0. 数据集配置（与 benchmark.py 完全一致）
# ==========================================

def get_dataset_config(base_dir):
    return [
        {"name": "Proxifier", "input_path": os.path.join(base_dir, "data", "Proxifier", "Proxifier_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "proxifier")},
        {"name": "Linux", "input_path": os.path.join(base_dir, "data", "Linux", "Linux_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "linux")},
        {"name": "HDFS", "input_path": os.path.join(base_dir, "data", "HDFS", "HDFS_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "hdfs")},
        {"name": "Mac", "input_path": os.path.join(base_dir, "data", "Mac", "Mac_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "mac")},
        {"name": "Apache", "input_path": os.path.join(base_dir, "data", "Apache", "Apache_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "apache")},
        {"name": "Hadoop", "input_path": os.path.join(base_dir, "data", "Hadoop", "Hadoop_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "hadoop")},
        {"name": "BGL", "input_path": os.path.join(base_dir, "data", "BGL", "BGL_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "bgl")},
        {"name": "HealthApp", "input_path": os.path.join(base_dir, "data", "HealthApp", "HealthApp_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "healthapp")},
        {"name": "HPC", "input_path": os.path.join(base_dir, "data", "HPC", "HPC_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "hpc")},
        {"name": "OpenSSH", "input_path": os.path.join(base_dir, "data", "OpenSSH", "OpenSSH_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "openssh")},
        {"name": "OpenStack", "input_path": os.path.join(base_dir, "data", "OpenStack", "OpenStack_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "openstack")},
        {"name": "Thunderbird", "input_path": os.path.join(base_dir, "data", "Thunderbird", "Thunderbird_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "thunderbird")},
        {"name": "Zookeeper", "input_path": os.path.join(base_dir, "data", "Zookeeper", "Zookeeper_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "zookeeper")},
        {"name": "Spark", "input_path": os.path.join(base_dir, "data", "Spark", "Spark_full.log_structured.csv"), "output_dir": os.path.join(base_dir, "result_ablation", "spark")},
    ]


# ==========================================
# 1. 数据预处理函数（与 benchmark.py 完全一致）
# ==========================================

def get_preprocessor(dataset_name):
    """根据数据集名称返回对应的预处理函数"""
    preprocessors = {
        "linux": linux_standard_preprocess,
        "hdfs": hdfs_standard_preprocess,
        "hadoop": hadoop_standard_preprocess,
        "bgl": bgl_standard_preprocess,
        "healthapp": healthapp_standard_preprocess,
        "hpc": hpc_standard_preprocess,
        "openssh": openssh_standard_preprocess,
        "openstack": openstack_standard_preprocess,
        "thunderbird": thunderbird_standard_preprocess,
        "zookeeper": zookeeper_standard_preprocess,
        "mac": mac_standard_preprocess,
        "apache": apache_standard_preprocess,
        "spark": spark_standard_preprocess,
    }
    return preprocessors.get(dataset_name.lower(), loghub_standard_preprocess)


# ==========================================
# 2. 消融实验变体定义
# ==========================================

class AblationVariant:
    """消融实验变体基类"""
    def __init__(self, name):
        self.name = name

    def run(self, dataset_name, preprocessed, contents, line_ids, total_logs):
        """
        执行变体解析流程

        Args:
            dataset_name: 数据集名称
            preprocessed: 预处理后的日志列表
            contents: 原始日志内容列表
            line_ids: 行ID列表
            total_logs: 总日志数

        Returns:
            output_rows: 预测结果行列表
            duration: 运行时间
        """
        raise NotImplementedError


def _post_process_clusters(all_refined_clusters):
    """统一后处理：从聚类结果生成 EventId 和 Template"""
    from src.stage3_4_extraction import FastTemplateExtractor
    
    extractor = FastTemplateExtractor(sample_size=50)
    results = extractor.extract_templates(all_refined_clusters)
    
    unique_templates_map = {}
    cluster_to_event = {}
    event_to_template = {}
    event_counter = 1
    
    sorted_res = sorted(results, key=lambda x: x['log_count'], reverse=True)
    for res in sorted_res:
        norm_key = re.sub(r'[^a-zA-Z0-9*]', '', res['template']).lower()
        if norm_key not in unique_templates_map:
            e_id = f"E{event_counter}"
            unique_templates_map[norm_key] = e_id
            event_to_template[e_id] = res['template']
            event_counter += 1
        cluster_to_event[res['cluster_id']] = unique_templates_map[norm_key]
    
    return cluster_to_event, event_to_template


def _generate_output_rows(all_refined_clusters, cluster_to_event, event_to_template, 
                          contents, line_ids):
    """根据聚类结果生成输出行"""
    output_rows = []
    for c_idx, cluster in enumerate(all_refined_clusters):
        cid = c_idx + 1
        e_id = cluster_to_event.get(cid, "E99")
        e_temp = event_to_template.get(e_id, "<*>")
        for item in cluster:
            orig_idx = item['original_index']
            output_rows.append({
                "LineId": line_ids[orig_idx],
                "Content": contents[orig_idx],
                "EventId": e_id,
                "EventTemplate": e_temp
            })
    
    output_rows.sort(key=lambda x: int(x['LineId']))
    return output_rows


def _cache_accelerator(preprocessed, contents, line_ids, total_logs,
                          process_train_data_func):
    """
    缓存加速引擎

    Args:
        preprocessed: 预处理后的日志列表
        contents: 原始日志内容列表
        line_ids: 行ID列表
        total_logs: 总日志数
        process_train_data_func: 处理函数，接收 train_data 返回 (clusters, event_to_template, cluster_to_event)

    Returns:
        output_rows: 预测结果行列表
        duration: 运行时间
    """
    print(f"[INFO] Large dataset detected ({total_logs} logs), activating cache accelerator...")

    freq_counter = Counter(preprocessed)
    unique_count = len(freq_counter)
    print(f"[INFO] Global scan complete: {unique_count} unique patterns found.")

    top_k = min(8000, unique_count)
    train_data = [item[0] for item in freq_counter.most_common(top_k)]
    print(f"[INFO] Training on top {top_k} core patterns...")

    # 对采样数据执行聚类和模板提取
    start_time = time.time()
    train_clusters, event_to_template, cluster_to_event = process_train_data_func(train_data)
    train_duration = time.time() - start_time

    # 构建 O(1) 推理缓存
    print("[INFO] Building O(1) inference cache...")
    fast_cache = {}
    for c_idx, cluster in enumerate(train_clusters):
        cid = c_idx + 1
        e_id = cluster_to_event.get(cid, "E99")
        for item in cluster:
            orig_idx = item['original_index']
            p_str = train_data[orig_idx]
            fast_cache[p_str] = e_id

    # 预计算模板信息
    template_info = {}
    for temp_e_id, temp_str in event_to_template.items():
        t_set = set(temp_str.split())
        template_info[temp_e_id] = (t_set, len(t_set))

    # 快速映射全量数据
    print(f"[INFO] Fast mapping {total_logs} logs...")
    infer_start = time.time()
    output_rows = []

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

        output_rows.append({
            "LineId": line_ids[i],
            "Content": contents[i],
            "EventId": e_id,
            "EventTemplate": event_to_template.get(e_id, "<*>")
        })

    duration = train_duration + (time.time() - infer_start)
    print(f"[OK] Data processing complete! Total time: {duration:.4f}s")
    return output_rows, duration


# ==========================================
# Trie 树缓存数据结构（用于 TreeCache 变体）
# ==========================================

class TrieNode:
    """Trie 树节点"""
    def __init__(self):
        self.children = {}
        self.event_id = None
        self.is_end = False


class LogPatternTrie:
    """
    基于 Trie 前缀树的日志模式缓存
    将日志的 token 序列按层次组织，支持最长前缀匹配
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, tokens, event_id):
        """将一条日志的 token 序列插入 Trie 树"""
        node = self.root
        for token in tokens:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        node.is_end = True
        node.event_id = event_id

    def search(self, tokens):
        """
        最长前缀匹配查询
        返回: (匹配的 event_id, 匹配的 token 长度)
        """
        node = self.root
        last_match_event = None
        match_length = 0
        for i, token in enumerate(tokens):
            if token not in node.children:
                break
            node = node.children[token]
            if node.is_end:
                last_match_event = node.event_id
                match_length = i + 1
        return last_match_event, match_length


class NoBucketVariant(AblationVariant):
    """
    Abl-1: 无强锚点分桶
    保留领域预处理和缓存加速，但跳过阶段一的强锚点分桶
    所有日志放入单一桶进行聚类
    """
    def __init__(self):
        super().__init__("NoBucket")

    def run(self, dataset_name, preprocessed, contents, line_ids, total_logs):
        from src.stage2_lsh import AdaptiveLSHClusterer
        from src.stage3_4_extraction import FastTemplateExtractor

        # 跳过阶段一：所有日志放入一个桶
        tokenized_logs = [log.strip().split() for log in preprocessed]
        single_bucket = [{
            "original_index": idx,
            "raw_log": contents[idx],
            "tokens": tokens
        } for idx, tokens in enumerate(tokenized_logs)]

        # 阶段二：对整个数据集聚类（使用与 main 相同的参数）
        if dataset_name.lower() == "hdfs":
            clusterer = AdaptiveLSHClusterer(min_similarity=0.85)
        elif dataset_name.lower() in ["hpc", "thunderbird", "hadoop", "openssh", "spark", "zookeeper"]:
            clusterer = AdaptiveLSHClusterer(min_similarity=0.65)
        elif dataset_name.lower() == "bgl":
            clusterer = AdaptiveLSHClusterer(min_similarity=0.88)
        elif dataset_name.lower() == "healthapp":
            clusterer = AdaptiveLSHClusterer(min_similarity=0.85)
        elif dataset_name.lower() == "openstack":
            clusterer = AdaptiveLSHClusterer(min_similarity=0.75)
        elif dataset_name.lower() == "mac":
            clusterer = AdaptiveLSHClusterer(min_similarity=0.5)
        else:
            clusterer = AdaptiveLSHClusterer(min_similarity=0.6)

        # 缓存加速分支（大规模数据集）
        if total_logs > 50000:
            print(f"[INFO] Large dataset detected ({total_logs} logs), activating cache accelerator...")

            freq_counter = Counter(preprocessed)
            unique_count = len(freq_counter)
            print(f"[INFO] Global scan complete: {unique_count} unique patterns found.")

            top_k = min(8000, unique_count)
            train_data = [item[0] for item in freq_counter.most_common(top_k)]
            print(f"[INFO] Training on top {top_k} core patterns...")

            # 对采样数据执行聚类
            train_tokens = [log.strip().split() for log in train_data]
            train_bucket = [{
                "original_index": idx,
                "raw_log": train_data[idx],
                "tokens": tokens
            } for idx, tokens in enumerate(train_tokens)]

            start_time = time.time()
            refined_clusters = clusterer.cluster_bucket(train_bucket)
            train_duration = time.time() - start_time

            # 模板提取
            extractor = FastTemplateExtractor(sample_size=50)
            results = extractor.extract_templates(refined_clusters)

            # 后处理
            unique_templates_map = {}
            cluster_to_event = {}
            event_to_template = {}
            event_counter = 1

            sorted_res = sorted(results, key=lambda x: x['log_count'], reverse=True)
            for res in sorted_res:
                norm_key = re.sub(r'[^a-zA-Z0-9*]', '', res['template']).lower()
                if norm_key not in unique_templates_map:
                    e_id = f"E{event_counter}"
                    unique_templates_map[norm_key] = e_id
                    event_to_template[e_id] = res['template']
                    event_counter += 1
                cluster_to_event[res['cluster_id']] = unique_templates_map[norm_key]

            # 构建 O(1) 推理缓存
            print("[INFO] Building O(1) inference cache...")
            fast_cache = {}
            for c_idx, cluster in enumerate(refined_clusters):
                cid = c_idx + 1
                e_id = cluster_to_event.get(cid, "E99")
                for item in cluster:
                    orig_idx = item['original_index']
                    p_str = train_data[orig_idx]
                    fast_cache[p_str] = e_id

            # 预计算模板信息
            template_info = {}
            for temp_e_id, temp_str in event_to_template.items():
                t_set = set(temp_str.split())
                template_info[temp_e_id] = (t_set, len(t_set))

            # 快速映射全量数据
            print(f"[INFO] Fast mapping {total_logs} logs...")
            infer_start = time.time()
            output_rows = []

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

                output_rows.append({
                    "LineId": line_ids[i],
                    "Content": contents[i],
                    "EventId": e_id,
                    "EventTemplate": event_to_template.get(e_id, "<*>")
                })

            duration = train_duration + (time.time() - infer_start)
            print(f"[OK] Data processing complete! Total time: {duration:.4f}s")
            return output_rows, duration

        else:
            # 传统模式（小数据集）
            start_time = time.time()
            refined_clusters = clusterer.cluster_bucket(single_bucket)
            duration = time.time() - start_time

            # 阶段三/四：模板提取
            extractor = FastTemplateExtractor(sample_size=50)
            results = extractor.extract_templates(refined_clusters)

            # 后处理：生成 EventId
            unique_templates_map = {}
            cluster_to_event = {}
            event_to_template = {}
            event_counter = 1

            sorted_res = sorted(results, key=lambda x: x['log_count'], reverse=True)
            for res in sorted_res:
                norm_key = re.sub(r'[^a-zA-Z0-9*]', '', res['template']).lower()
                if norm_key not in unique_templates_map:
                    e_id = f"E{event_counter}"
                    unique_templates_map[norm_key] = e_id
                    event_to_template[e_id] = res['template']
                    event_counter += 1
                cluster_to_event[res['cluster_id']] = unique_templates_map[norm_key]

            # 生成输出
            output_rows = []
            for c_idx, cluster in enumerate(refined_clusters):
                cid = c_idx + 1
                e_id = cluster_to_event.get(cid, "E99")
                e_temp = event_to_template.get(e_id, "<*>")
                for item in cluster:
                    orig_idx = item['original_index']
                    output_rows.append({
                        "LineId": line_ids[orig_idx],
                        "Content": contents[orig_idx],
                        "EventId": e_id,
                        "EventTemplate": e_temp
                    })

            output_rows.sort(key=lambda x: int(x['LineId']))
            return output_rows, duration


class FixedThresholdVariant(AblationVariant):
    """基类：固定阈值聚类"""
    def __init__(self, name, fixed_threshold):
        super().__init__(name)
        self.fixed_threshold = fixed_threshold

    def _process_buckets(self, initial_buckets, fixed_threshold):
        """处理所有桶的聚类逻辑"""
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        import numpy as np

        def jaccard_distance(tokens1, tokens2):
            set1, set2 = set(tokens1), set(tokens2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            if union == 0:
                return 1.0
            return 1.0 - (intersection / union)

        all_refined_clusters = []
        for log_items in initial_buckets.values():
            if not log_items:
                continue

            n_logs = len(log_items)
            if n_logs == 1:
                all_refined_clusters.append([log_items[0]])
                continue

            # 大桶内存保护：超过 5000 条日志时，使用采样 + 近似聚类
            max_bucket_size = 5000
            if n_logs > max_bucket_size:
                import random
                random.seed(42)
                sample_indices = random.sample(range(n_logs), max_bucket_size)
                sample_items = [log_items[i] for i in sample_indices]
                sample_tokens = [item["tokens"] for item in sample_items]

                s_logs = len(sample_items)
                dist_matrix = np.zeros((s_logs, s_logs))
                for i in range(s_logs):
                    for j in range(i + 1, s_logs):
                        d = jaccard_distance(sample_tokens[i], sample_tokens[j])
                        dist_matrix[i, j] = d
                        dist_matrix[j, i] = d

                condensed_dist = squareform(dist_matrix)
                threshold = 1.0 - fixed_threshold
                Z = linkage(condensed_dist, method='complete')
                cluster_labels = fcluster(Z, t=threshold, criterion='distance')

                sample_clusters = {}
                for idx, label in enumerate(cluster_labels):
                    if label not in sample_clusters:
                        sample_clusters[label] = []
                    sample_clusters[label].append(sample_items[idx])

                cluster_reps = {}
                for label, items in sample_clusters.items():
                    rep_tokens = set(items[0]["tokens"])
                    cluster_reps[label] = rep_tokens

                final_clusters = {label: [] for label in sample_clusters.keys()}
                for item in log_items:
                    item_tokens = set(item["tokens"])
                    best_label = None
                    best_score = -1
                    for label, rep in cluster_reps.items():
                        intersect = len(item_tokens & rep)
                        union = len(item_tokens | rep)
                        score = intersect / union if union > 0 else 0
                        if score > best_score:
                            best_score = score
                            best_label = label
                    if best_label is not None and best_score >= fixed_threshold:
                        final_clusters[best_label].append(item)
                    else:
                        new_label = max(final_clusters.keys()) + 1 if final_clusters else 1
                        final_clusters[new_label] = [item]
                        cluster_reps[new_label] = item_tokens

                all_refined_clusters.extend(list(final_clusters.values()))
            else:
                tokens_list = [item["tokens"] for item in log_items]
                dist_matrix = np.zeros((n_logs, n_logs))

                for i in range(n_logs):
                    for j in range(i + 1, n_logs):
                        d = jaccard_distance(tokens_list[i], tokens_list[j])
                        dist_matrix[i, j] = d
                        dist_matrix[j, i] = d

                condensed_dist = squareform(dist_matrix)
                threshold = 1.0 - fixed_threshold

                Z = linkage(condensed_dist, method='complete')
                cluster_labels = fcluster(Z, t=threshold, criterion='distance')

                clusters = {}
                for idx, label in enumerate(cluster_labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(log_items[idx])

                all_refined_clusters.extend(list(clusters.values()))
        
        return all_refined_clusters

    def run(self, dataset_name, preprocessed, contents, line_ids, total_logs):
        from src.stage1_preprocessing import SmartPreprocessor

        # 阶段一：分桶（与 main 相同参数）
        if dataset_name.lower() == "hdfs":
            preprocessor = SmartPreprocessor(entropy_threshold=1.0, max_check_positions=15)
        elif dataset_name.lower() == "mac":
            preprocessor = SmartPreprocessor(entropy_threshold=2.5, max_check_positions=12)
        else:
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=12)

        # 缓存加速分支（大规模数据集）
        if total_logs > 50000:
            def process_train_data(train_data):
                train_tokens = [log.strip().split() for log in train_data]
                train_bucket = [{
                    "original_index": idx,
                    "raw_log": train_data[idx],
                    "tokens": tokens
                } for idx, tokens in enumerate(train_tokens)]

                initial_buckets = {"all": train_bucket}
                all_refined_clusters = self._process_buckets(initial_buckets, self.fixed_threshold)
                cluster_to_event, event_to_template = _post_process_clusters(all_refined_clusters)
                return all_refined_clusters, event_to_template, cluster_to_event

            return _cache_accelerator(preprocessed, contents, line_ids, total_logs, process_train_data)

        # 传统模式（小数据集）
        start_time = time.time()
        initial_buckets = preprocessor.fit_and_bucket(preprocessed)
        all_refined_clusters = self._process_buckets(initial_buckets, self.fixed_threshold)
        duration = time.time() - start_time

        cluster_to_event, event_to_template = _post_process_clusters(all_refined_clusters)
        output_rows = _generate_output_rows(all_refined_clusters, cluster_to_event, event_to_template, contents, line_ids)
        return output_rows, duration


class FixedThresholdLowVariant(FixedThresholdVariant):
    """
    Abl-2a: 固定低阈值聚类
    保留领域预处理和分桶，但使用固定低阈值 0.3 替代肘部法则
    效果：聚类更严格，产生的簇更多
    """
    def __init__(self):
        super().__init__("FixedThresholdLow", fixed_threshold=0.3)


class FixedThresholdMid1Variant(FixedThresholdVariant):
    """
    Abl-2b: 固定阈值 0.4 聚类
    保留领域预处理和分桶，但使用固定阈值 0.4 替代肘部法则
    """
    def __init__(self):
        super().__init__("FixedThreshold04", fixed_threshold=0.4)


class FixedThresholdMid2Variant(FixedThresholdVariant):
    """
    Abl-2c: 固定阈值 0.5 聚类
    保留领域预处理和分桶，但使用固定阈值 0.5 替代肘部法则
    """
    def __init__(self):
        super().__init__("FixedThreshold05", fixed_threshold=0.5)


class FixedThresholdMid3Variant(FixedThresholdVariant):
    """
    Abl-2d: 固定阈值 0.6 聚类
    保留领域预处理和分桶，但使用固定阈值 0.6 替代肘部法则
    """
    def __init__(self):
        super().__init__("FixedThreshold06", fixed_threshold=0.6)


class FixedThresholdHighVariant(FixedThresholdVariant):
    """
    Abl-2e: 固定高阈值聚类
    保留领域预处理和分桶，但使用固定高阈值 0.7 替代肘部法则
    效果：聚类更宽松，产生的簇更少
    """
    def __init__(self):
        super().__init__("FixedThresholdHigh", fixed_threshold=0.7)


class NoSequenceAlignVariant(AblationVariant):
    """
    Abl-4: 无序列对齐
    保留完整分桶和聚类，但模板提取时使用基于正则表达式模式挖掘的方法替代 DP 对齐
    核心思想：通过分析 token 的语义类型（数字、IP、路径等）自动推断变量位置
    """
    def __init__(self):
        super().__init__("NoSequenceAlign")

    def _get_preprocessor_clusterer(self, dataset_name):
        """获取数据集特定的预处理器和聚类器参数"""
        from src.stage1_preprocessing import SmartPreprocessor
        from src.stage2_lsh import AdaptiveLSHClusterer
        
        if dataset_name.lower() == "hdfs":
            preprocessor = SmartPreprocessor(entropy_threshold=1.0, max_check_positions=15)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.85)
        elif dataset_name.lower() in ["hpc", "thunderbird", "hadoop", "openssh", "spark", "zookeeper"]:
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=12)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.65)
        elif dataset_name.lower() == "bgl":
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=12)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.88)
        elif dataset_name.lower() == "healthapp":
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=10)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.85)
        elif dataset_name.lower() == "openstack":
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=12)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.75)
        elif dataset_name.lower() == "mac":
            preprocessor = SmartPreprocessor(entropy_threshold=2.5, max_check_positions=12)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.5)
        else:
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=10)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.6)
        
        return preprocessor, clusterer

    def _get_token_type(self, token):
        """
        判断 token 的语义类型
        返回: 'number', 'ip', 'path', 'url', 'hex', 'date', 'time', 'literal'
        """
        # 纯数字（包括小数、负数）
        if re.match(r'^-?\d+(\.\d+)?$', token):
            return 'number'
        
        # IP 地址（IPv4）
        if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', token):
            return 'ip'
        
        # 文件路径
        if re.match(r'^(/[\w\-./]+)|([A-Za-z]:\\[\w\-\\.]+)$', token):
            return 'path'
        
        # URL
        if re.match(r'^(http|https|ftp)://', token):
            return 'url'
        
        # 十六进制
        if re.match(r'^0x[0-9a-fA-F]+$', token):
            return 'hex'
        
        # 日期格式
        if re.match(r'^\d{4}-\d{2}-\d{2}$', token):
            return 'date'
        
        # 时间格式
        if re.match(r'^\d{2}:\d{2}:\d{2}$', token):
            return 'time'
        
        # 默认字面量
        return 'literal'

    def _extract_template_by_pattern(self, cluster):
        """
        基于正则表达式模式挖掘的模板提取
        1. 分析每个位置上的 token 类型分布
        2. 如果某个位置上的 token 类型不一致，或者都是可变类型（数字、IP等），则标记为 <*>
        3. 如果类型一致且是字面量，则保留该值
        """
        if not cluster:
            return ""
        
        all_tokens = [item["tokens"] for item in cluster]
        n_logs = len(all_tokens)
        
        if n_logs == 1:
            return " ".join(all_tokens[0])
        
        # 找到最短的日志长度作为基准
        min_len = min(len(t) for t in all_tokens)
        
        template_tokens = []
        
        for pos in range(min_len):
            # 收集该位置上的所有 token 及其类型
            pos_tokens = [tokens[pos] for tokens in all_tokens if pos < len(tokens)]
            pos_types = [self._get_token_type(t) for t in pos_tokens]
            
            if not pos_tokens:
                continue
            
            # 统计类型分布
            type_counter = Counter(pos_types)
            most_common_type, type_count = type_counter.most_common(1)[0]
            
            # 判断是否应该泛化
            should_generalize = False
            
            # 规则1：如果类型不一致（多样性高）
            if len(type_counter) > 1:
                should_generalize = True
            
            # 规则2：如果都是可变类型（数字、IP、路径等），即使值不同也泛化
            if most_common_type in ['number', 'ip', 'path', 'url', 'hex']:
                # 检查值是否完全相同
                unique_values = set(pos_tokens)
                if len(unique_values) > 1:
                    should_generalize = True
            
            # 规则3：字面量但值不完全相同，且出现频率低于阈值
            if most_common_type == 'literal':
                unique_values = set(pos_tokens)
                diversity = len(unique_values) / len(pos_tokens)
                if diversity > 0.3:  # 30% 以上不同则泛化
                    should_generalize = True
            
            if should_generalize:
                template_tokens.append("<*>")
            else:
                # 保留最常见的值
                value_counter = Counter(pos_tokens)
                most_common_value = value_counter.most_common(1)[0][0]
                template_tokens.append(most_common_value)
        
        # 处理长度超过 min_len 的日志（它们有额外的 token）
        # 如果超过 20% 的日志在该位置有值，则添加 <*>
        extra_positions = 0
        for tokens in all_tokens:
            if len(tokens) > min_len:
                extra_positions = max(extra_positions, len(tokens) - min_len)
        
        for offset in range(extra_positions):
            pos = min_len + offset
            pos_tokens = [tokens[pos] for tokens in all_tokens if pos < len(tokens)]
            if len(pos_tokens) > n_logs * 0.2:  # 超过 20% 的日志有该位置
                template_tokens.append("<*>")
        
        # 清理连续的 <*>
        cleaned_template = []
        for token in template_tokens:
            if token == "<*>" and cleaned_template and cleaned_template[-1] == "<*>":
                continue
            cleaned_template.append(token)
        
        return " ".join(cleaned_template)

    def _cluster_and_extract(self, preprocessed_data, dataset_name):
        """执行分桶、聚类和基于模式挖掘的模板提取"""
        preprocessor, clusterer = self._get_preprocessor_clusterer(dataset_name)

        initial_buckets = preprocessor.fit_and_bucket(preprocessed_data)

        all_refined_clusters = []
        for log_items in initial_buckets.values():
            if not log_items:
                continue
            refined_clusters = clusterer.cluster_bucket(log_items)
            all_refined_clusters.extend(refined_clusters)

        # 基于模式挖掘的模板提取
        results = []
        for idx, cluster in enumerate(all_refined_clusters):
            if not cluster:
                continue

            template_str = self._extract_template_by_pattern(cluster)

            results.append({
                "cluster_id": idx + 1,
                "log_count": len(cluster),
                "template": template_str
            })

        # 后处理
        unique_templates_map = {}
        cluster_to_event = {}
        event_to_template = {}
        event_counter = 1

        sorted_res = sorted(results, key=lambda x: x['log_count'], reverse=True)
        for res in sorted_res:
            norm_key = re.sub(r'[^a-zA-Z0-9*]', '', res['template']).lower()
            if norm_key not in unique_templates_map:
                e_id = f"E{event_counter}"
                unique_templates_map[norm_key] = e_id
                event_to_template[e_id] = res['template']
                event_counter += 1
            cluster_to_event[res['cluster_id']] = unique_templates_map[norm_key]

        return all_refined_clusters, event_to_template, cluster_to_event

    def run(self, dataset_name, preprocessed, contents, line_ids, total_logs):
        # 缓存加速分支（大规模数据集）
        if total_logs > 50000:
            def process_train_data(train_data):
                return self._cluster_and_extract(train_data, dataset_name)

            return _cache_accelerator(preprocessed, contents, line_ids, total_logs, process_train_data)

        # 传统模式（小数据集）
        preprocessor, clusterer = self._get_preprocessor_clusterer(dataset_name)
        
        start_time = time.time()
        initial_buckets = preprocessor.fit_and_bucket(preprocessed)

        all_refined_clusters = []
        for log_items in initial_buckets.values():
            if not log_items:
                continue
            refined_clusters = clusterer.cluster_bucket(log_items)
            all_refined_clusters.extend(refined_clusters)

        # 基于模式挖掘的模板提取
        results = []
        for idx, cluster in enumerate(all_refined_clusters):
            if not cluster:
                continue

            template_str = self._extract_template_by_pattern(cluster)

            results.append({
                "cluster_id": idx + 1,
                "log_count": len(cluster),
                "template": template_str
            })

        duration = time.time() - start_time

        # 后处理
        unique_templates_map = {}
        cluster_to_event = {}
        event_to_template = {}
        event_counter = 1

        sorted_res = sorted(results, key=lambda x: x['log_count'], reverse=True)
        for res in sorted_res:
            norm_key = re.sub(r'[^a-zA-Z0-9*]', '', res['template']).lower()
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
            e_temp = event_to_template.get(e_id, "<*>")
            for item in cluster:
                orig_idx = item['original_index']
                output_rows.append({
                    "LineId": line_ids[orig_idx],
                    "Content": contents[orig_idx],
                    "EventId": e_id,
                    "EventTemplate": e_temp
                })

        output_rows.sort(key=lambda x: int(x['LineId']))
        return output_rows, duration


class TreeCacheVariant(AblationVariant):
    """
    Abl-5: 基于 Trie 前缀树的缓存变体
    基本流程不变，只将基于 hash 构建的 O(1) 缓存改为基于 Trie 前缀树的层次化缓存
    利用 Token 序列的前缀匹配实现高效的日志模式查找，对相似但不完全相同的模式具有更好的泛化能力
    """
    def __init__(self):
        super().__init__("TreeCache")

    def run(self, dataset_name, preprocessed, contents, line_ids, total_logs):
        from src.stage1_preprocessing import SmartPreprocessor
        from src.stage2_lsh import AdaptiveLSHClusterer
        from src.stage3_4_extraction import FastTemplateExtractor

        # 阶段一：分桶（与 main 相同的参数配置）
        if dataset_name.lower() == "hdfs":
            preprocessor = SmartPreprocessor(entropy_threshold=1.0, max_check_positions=15)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.85)
        elif dataset_name.lower() in ["hpc", "thunderbird", "hadoop", "openssh", "spark", "zookeeper"]:
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=12)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.65)
        elif dataset_name.lower() == "bgl":
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=12)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.88)
        elif dataset_name.lower() == "healthapp":
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=10)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.85)
        elif dataset_name.lower() == "openstack":
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=12)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.75)
        elif dataset_name.lower() == "mac":
            preprocessor = SmartPreprocessor(entropy_threshold=2.5, max_check_positions=12)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.5)
        else:
            preprocessor = SmartPreprocessor(entropy_threshold=1.5, max_check_positions=10)
            clusterer = AdaptiveLSHClusterer(min_similarity=0.6)

        # 缓存加速分支（大规模数据集）
        if total_logs > 50000:
            print(f"[INFO] Large dataset detected ({total_logs} logs), activating tree cache accelerator...")

            freq_counter = Counter(preprocessed)
            unique_count = len(freq_counter)
            print(f"[INFO] Global scan complete: {unique_count} unique patterns found.")

            top_k = min(8000, unique_count)
            train_data = [item[0] for item in freq_counter.most_common(top_k)]
            print(f"[INFO] Training on top {top_k} core patterns...")

            # 对采样数据执行完整的分桶、聚类和模板提取
            start_time = time.time()
            train_tokens = [log.strip().split() for log in train_data]
            train_bucket = [{
                "original_index": idx,
                "raw_log": train_data[idx],
                "tokens": tokens
            } for idx, tokens in enumerate(train_tokens)]

            refined_clusters = clusterer.cluster_bucket(train_bucket)
            train_duration = time.time() - start_time

            # 模板提取
            extractor = FastTemplateExtractor(sample_size=50)
            results = extractor.extract_templates(refined_clusters)

            # 后处理：生成 EventId
            unique_templates_map = {}
            cluster_to_event = {}
            event_to_template = {}
            event_counter = 1

            sorted_res = sorted(results, key=lambda x: x['log_count'], reverse=True)
            for res in sorted_res:
                norm_key = re.sub(r'[^a-zA-Z0-9*]', '', res['template']).lower()
                if norm_key not in unique_templates_map:
                    e_id = f"E{event_counter}"
                    unique_templates_map[norm_key] = e_id
                    event_to_template[e_id] = res['template']
                    event_counter += 1
                cluster_to_event[res['cluster_id']] = unique_templates_map[norm_key]

            # 构建 Trie 树缓存（替代原有的 hash 缓存）
            print("[INFO] Building Trie-based inference cache...")
            trie_cache = LogPatternTrie()
            for c_idx, cluster in enumerate(refined_clusters):
                cid = c_idx + 1
                e_id = cluster_to_event.get(cid, "E99")
                for item in cluster:
                    orig_idx = item['original_index']
                    tokens = train_data[orig_idx].split()
                    trie_cache.insert(tokens, e_id)

            # 预计算模板信息（用于 Trie 未命中时的回退匹配）
            template_info = {}
            for temp_e_id, temp_str in event_to_template.items():
                t_set = set(temp_str.split())
                template_info[temp_e_id] = (t_set, len(t_set))

            # 使用 Trie 树快速映射全量数据
            print(f"[INFO] Fast mapping {total_logs} logs with Trie cache...")
            infer_start = time.time()
            output_rows = []

            for i in range(total_logs):
                p_str = preprocessed[i]
                tokens = p_str.split()

                # 第一步：Trie 树最长前缀匹配
                e_id, match_len = trie_cache.search(tokens)

                # 第二步：如果 Trie 未命中或匹配长度不足 50%，回退到 Jaccard 相似度匹配
                if e_id is None or match_len < len(tokens) * 0.5:
                    best_e, best_score = "E99", 0
                    p_set = set(tokens)
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

            duration = train_duration + (time.time() - infer_start)
            print(f"[OK] Data processing complete! Total time: {duration:.4f}s")
            return output_rows, duration

        # 传统模式（小数据集）- 与 main 流程一致
        start_time = time.time()
        initial_buckets = preprocessor.fit_and_bucket(preprocessed)

        all_refined_clusters = []
        for log_items in initial_buckets.values():
            if not log_items:
                continue
            refined_clusters = clusterer.cluster_bucket(log_items)
            all_refined_clusters.extend(refined_clusters)

        duration = time.time() - start_time

        cluster_to_event, event_to_template = _post_process_clusters(all_refined_clusters)
        output_rows = _generate_output_rows(all_refined_clusters, cluster_to_event, event_to_template, contents, line_ids)
        return output_rows, duration


# ==========================================
# 3. 消融实验主控程序
# ==========================================

ABLACTION_VARIANTS = [
    NoBucketVariant(),
    FixedThresholdLowVariant(),
    FixedThresholdMid1Variant(),
    FixedThresholdMid2Variant(),
    FixedThresholdMid3Variant(),
    FixedThresholdHighVariant(),
    NoSequenceAlignVariant(),
    TreeCacheVariant(),
]


def run_single_experiment(variant, dataset_config, base_dir):
    """运行单个消融实验"""
    dataset_name = dataset_config["name"]
    input_path = dataset_config["input_path"]
    output_dir = dataset_config["output_dir"]
    variant_name = variant.name

    output_path = os.path.join(
        output_dir,
        f"{dataset_name.lower()}_{variant_name.lower()}_predictions.csv"
    )

    print(f"\n  [{variant_name}] Processing {dataset_name}...")

    if not os.path.exists(input_path):
        print(f"  [ERROR] Input file not found: {input_path}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # 读取数据（与 benchmark.py 完全一致）
    line_ids, contents = [], []
    if dataset_name.lower() in ["hdfs", "mac", "hadoop", "bgl", "healthapp", "hpc", "openssh", "openstack", "spark", "thunderbird", "zookeeper"]:
        print(f"  [INFO] Loading {dataset_name} full dataset into memory, please wait...")

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line_ids.append(row["LineId"])
            contents.append(row["Content"].strip())

    total_logs = len(contents)
    print(f"  [OK] Data loaded, total {total_logs} logs.")

    # 数据预处理（与 benchmark.py 完全一致）
    print("  [INFO] Preprocessing...")
    preprocessor_func = get_preprocessor(dataset_name)
    preprocessed = preprocessor_func(contents)

    # 执行变体解析
    try:
        output_rows, duration = variant.run(dataset_name, preprocessed, contents, line_ids, total_logs)
    except Exception as e:
        print(f"  [ERROR] Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 保存结果
    with open(output_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["LineId", "Content", "EventId", "EventTemplate"])
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"  [OK] Results saved to: {output_path}")

    # 评估指标 - 按变体汇总到各自的 CSV 文件
    eval_dataset_name = f"{dataset_name}_{variant_name}"
    eval_output_path = os.path.join(
        base_dir,
        "result_ablation",
        f"ablation_{variant_name.lower()}_results.csv"
    )
    os.makedirs(os.path.dirname(eval_output_path), exist_ok=True)
    try:
        evaluate_metrics(
            gt_path=input_path,
            pred_path=output_path,
            dataset_name=eval_dataset_name,
            runtime=duration,
            output_csv_path=eval_output_path
        )
        return {
            "variant": variant_name,
            "dataset": dataset_name,
            "duration": duration,
            "status": "success",
            "metrics_path": eval_output_path
        }
    except Exception as e:
        print(f"  [WARN] Evaluation failed: {e}")
        return {
            "variant": variant_name,
            "dataset": dataset_name,
            "duration": duration,
            "status": "eval_failed"
        }


def run_ablation_study(target_datasets=None, target_variants=None):
    """
    运行完整的消融实验

    Args:
        target_datasets: 指定要测试的数据集名称列表，如 ["HDFS", "BGL"]
        target_variants: 指定要测试的变体名称列表，如 ["NoBucket", "NoCluster"]
    """
    base_dir = current_dir
    datasets = get_dataset_config(base_dir)

    # 过滤数据集
    if target_datasets:
        datasets = [d for d in datasets if d["name"] in target_datasets]

    # 过滤变体
    variants = ABLACTION_VARIANTS
    if target_variants:
        variants = [v for v in variants if v.name in target_variants]

    print("=" * 70)
    print("PEAC-Log Ablation Study")
    print("=" * 70)
    print(f"Datasets: {len(datasets)}")
    print(f"Variants: {len(variants)}")
    print(f"Total experiments: {len(datasets) * len(variants)}")
    print("=" * 70)

    all_results = []

    for ds in datasets:
        dataset_name = ds["name"]
        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")

        for variant in variants:
            result = run_single_experiment(variant, ds, base_dir)
            if result:
                all_results.append(result)

    # 汇总报告
    print("\n" + "=" * 70)
    print("Ablation Study Summary")
    print("=" * 70)

    summary_path = os.path.join(base_dir, "result_ablation", "ablation_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Experiment completed! Summary saved to: {summary_path}")
    print(f"[INFO] Detailed results see result_ablation directory")

    return all_results


# ==========================================
# 4. 主入口
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ablation Study")
    parser.add_argument("--datasets", nargs="+", help="Specify datasets, e.g. HDFS BGL")
    parser.add_argument("--variants", nargs="+", help="Specify variants, e.g. NoBucket NoCluster")
    parser.add_argument("--compare", action="store_true", help="Compare results only")
    args = parser.parse_args()

    if args.compare:
        from ablation_visualize import compare_ablation_results
        compare_ablation_results()
    else:
        run_ablation_study(
            target_datasets=args.datasets,
            target_variants=args.variants
        )
