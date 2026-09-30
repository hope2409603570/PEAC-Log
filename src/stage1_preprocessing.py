import math
from collections import defaultdict, Counter

class SmartPreprocessor:
    def __init__(self, entropy_threshold=0.5, max_check_positions=10):
        self.entropy_threshold = entropy_threshold
        self.max_check_positions = max_check_positions
        self.strong_anchor_positions = [] 

    def _tokenize(self, log_message):
        # Preserve all characters and digits during whitespace tokenization.
        return log_message.strip().split()

    def _calculate_positional_entropy(self, tokenized_logs):
        position_counters = defaultdict(Counter)

        for tokens in tokenized_logs:
            for i in range(min(len(tokens), self.max_check_positions)):
                position_counters[i][tokens[i]] += 1

        entropies = {}
        for pos, counter in position_counters.items():
            # M_pos: number of messages that reach position pos; p_i = c_i / M_pos sums to 1.
            m_pos = sum(counter.values())
            entropy = 0.0
            for token, count in counter.items():
                probability = count / m_pos
                entropy -= probability * math.log2(probability)
            entropies[pos] = entropy
            
        return entropies

    def fit_and_bucket(self, raw_logs):
        print(f"Stage 1: processing {len(raw_logs)} log messages...")
        tokenized_logs = [self._tokenize(log) for log in raw_logs]

        # 1. Calculate positional entropy.
        entropies = self._calculate_positional_entropy(tokenized_logs)
        
        # 2. Extract strong anchors.
        self.strong_anchor_positions = [
            pos for pos, ent in entropies.items() if ent < self.entropy_threshold
        ]
        self.strong_anchor_positions.sort()
        
        print(f"Detected strong-anchor positions: {self.strong_anchor_positions}")
        for pos in self.strong_anchor_positions:
            print(f" - position {pos}: entropy={entropies[pos]:.4f}")

        # 3. Build initial physical buckets without a length constraint.
        buckets = defaultdict(list)
        
        for idx, tokens in enumerate(tokenized_logs):
            length = len(tokens)
            
            # The hash key depends only on strong features, not message length.
            key_parts = []
            for pos in self.strong_anchor_positions:
                if pos < length:
                    key_parts.append(f"P{pos}:{tokens[pos]}")
            
            # Use a deterministic fallback bucket when no anchor is present.
            hash_key = "_".join(key_parts) if key_parts else "NO_ANCHOR"
            
            buckets[hash_key].append({
                "original_index": idx,
                "raw_log": raw_logs[idx],
                "tokens": tokens
            })

        print(f"Stage 1 complete: created {len(buckets)} initial buckets.")
        return buckets
