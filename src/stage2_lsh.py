import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm

class AdaptiveLSHClusterer:
    def __init__(self, min_similarity=0.3):
        """
        :param min_similarity: Lower-bound similarity that prevents unrelated logs from merging.
        """
        self.min_distance = 1.0 - min_similarity

    def _jaccard_distance(self, tokens1, tokens2):
        """
        Calculate Jaccard distance between two token sets (1 - similarity).
        """
        set1, set2 = set(tokens1), set(tokens2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 1.0
        return 1.0 - (intersection / union)

    def _find_elbow_point(self, distances):
        """
        Select the distance cutoff dynamically with the maximum-curvature elbow.
        """
        if len(distances) <= 1:
            return np.max(distances) if len(distances) == 1 else 0.5

        sorted_dist = np.sort(distances)
        horizontal_span = len(sorted_dist) - 1
        vertical_span = sorted_dist[-1] - sorted_dist[0]
        if horizontal_span == 0 and vertical_span == 0:
            return sorted_dist[0]
        
        # The perpendicular distance has a constant denominator for every
        # point.  Vectorizing the numerator preserves the same argmax as the
        # original point-by-point projection while avoiding millions of tiny
        # NumPy allocations for large Top-K buckets.
        indices = np.arange(len(sorted_dist), dtype=float)
        vertical = sorted_dist - sorted_dist[0]
        numerator = np.abs(indices * vertical_span - vertical * horizontal_span)
        elbow_index = int(np.argmax(numerator))
                
        dynamic_threshold = sorted_dist[elbow_index]
        return min(dynamic_threshold, self.min_distance)

    def cluster_bucket(self, log_items):
        """
        Adaptively cluster messages within one physical bucket.
        """
        n_logs = len(log_items)
        if n_logs == 0:
            return []
        if n_logs == 1:
            return [[log_items[0]]]

        tokens_list = [item["tokens"] for item in log_items]
        dist_matrix = np.zeros((n_logs, n_logs))
        
        # Number of pairwise comparisons: N*(N-1)/2.
        total_pairs = n_logs * (n_logs - 1) // 2
        
        # Display pairwise-distance progress for large buckets.
        with tqdm(total=total_pairs, desc=f"Distance matrix (N={n_logs})", unit="pair", leave=False) as pbar:
            for i in range(n_logs):
                for j in range(i + 1, n_logs):
                    d = self._jaccard_distance(tokens_list[i], tokens_list[j])
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
                    pbar.update(1)
                
        condensed_dist = squareform(dist_matrix)
        
        dynamic_threshold = self._find_elbow_point(condensed_dist)
        print(
            f"  -> adaptive distance cutoff: {dynamic_threshold:.3f} "
            f"(similarity: {1 - dynamic_threshold:.3f})"
        )

        Z = linkage(condensed_dist, method='complete')
        cluster_labels = fcluster(Z, t=dynamic_threshold, criterion='distance')
        
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(log_items[idx])
            
        return list(clusters.values())
