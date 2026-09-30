import numpy as np

class FastTemplateExtractor:
    def __init__(self, sample_size=10):
        self.sample_size = sample_size

    def _diverse_sample(self, cluster):
        n_logs = len(cluster)
        if n_logs <= self.sample_size:
            return cluster

        cluster.sort(key=lambda x: len(x["tokens"]))
        sampled = [cluster[0], cluster[-1]]
        
        step = (n_logs - 2) / (self.sample_size - 2)
        for i in range(1, self.sample_size - 1):
            idx = int(1 + i * step)
            if cluster[idx] not in sampled:
                sampled.append(cluster[idx])
                
        return sampled

    def _align_and_merge(self, template_tokens, log_tokens):
        len_t = len(template_tokens)
        len_l = len(log_tokens)
        
        dp = np.zeros((len_t + 1, len_l + 1), dtype=int)
        
        for i in range(1, len_t + 1):
            for j in range(1, len_l + 1):
                if template_tokens[i-1] == log_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        i, j = len_t, len_l
        new_template = []
        
        while i > 0 and j > 0:
            if template_tokens[i-1] == log_tokens[j-1]:
                new_template.append(template_tokens[i-1])
                i -= 1
                j -= 1
            elif dp[i-1][j] > dp[i][j-1]:
                if not new_template or new_template[-1] != "<*>":
                    new_template.append("<*>")
                i -= 1
            else:
                if not new_template or new_template[-1] != "<*>":
                    new_template.append("<*>")
                j -= 1
                
        if i > 0 or j > 0:
            if not new_template or new_template[-1] != "<*>":
                new_template.append("<*>")

        new_template.reverse()
        return new_template

    def extract_templates(self, clusters):
        final_templates = []
        
        for idx, cluster in enumerate(clusters):
            if not cluster:
                continue
                
            sampled_logs = self._diverse_sample(cluster)
            current_template = sampled_logs[0]["tokens"]
            
            for log in sampled_logs[1:]:
                current_template = self._align_and_merge(current_template, log["tokens"])
                
            final_templates.append({
                "cluster_id": idx + 1,
                "log_count": len(cluster),
                "template": " ".join(current_template)
            })
            
        return final_templates