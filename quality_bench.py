#!/usr/bin/env python3
"""
AVM Quality Benchmarks

Measures effectiveness, not just speed:
1. Recall Precision - Are relevant memories returned?
2. Token Efficiency - How much context compression?
3. Cache Hit Rate - Provider cache effectiveness
4. Relevance Ranking - Is ranking quality good?
"""

import os
import sys
import tempfile
import random
from dataclasses import dataclass
from typing import List, Dict, Set

sys.path.insert(0, os.path.expanduser("~/.openclaw/workspace/vfs"))
os.environ["XDG_DATA_HOME"] = tempfile.mkdtemp()

from avm import AVM


@dataclass
class QualityResult:
    name: str
    score: float
    details: Dict


class QualityBenchmark:
    def __init__(self):
        self.results: List[QualityResult] = []
    
    def run_all(self):
        print("\n" + "="*60)
        print("AVM Quality Benchmarks")
        print("="*60 + "\n")
        
        self.bench_recall_precision()
        self.bench_token_efficiency()
        self.bench_relevance_ranking()
        self.bench_importance_filtering()
        
        self.print_report()
    
    def bench_recall_precision(self):
        """
        Test: Can recall find the right memories?
        
        Method:
        1. Create memories with known topics
        2. Query for specific topic
        3. Measure precision (correct / returned) and recall (found / total relevant)
        """
        print("[1/4] Recall Precision...")
        
        avm = AVM()
        mem = avm.agent_memory("precision_test")
        
        # Create labeled memories
        topics = {
            "trading": [
                "RSI indicates overbought at 75",
                "MACD crossover signals bullish momentum", 
                "Support level at 150, resistance at 165",
                "Volume spike suggests accumulation",
                "Moving average golden cross forming",
            ],
            "macro": [
                "Fed rate decision next week",
                "Inflation data came in hot",
                "GDP growth slowing in Q3",
                "Unemployment claims rising",
                "Treasury yields inverting",
            ],
            "risk": [
                "Position size limited to 2% per trade",
                "Stop loss at 5% below entry",
                "Max drawdown threshold reached",
                "Correlation risk between positions",
                "Volatility regime changing",
            ]
        }
        
        # Store with tags
        memory_map: Dict[str, str] = {}  # path -> topic
        for topic, contents in topics.items():
            for i, content in enumerate(contents):
                node = mem.remember(content, title=f"{topic}_{i}", tags=[topic])
                memory_map[node.path] = topic
        
        # Test queries
        test_cases = [
            ("RSI MACD technical", "trading", 3),
            ("Fed inflation macro", "macro", 3),
            ("risk stop loss position", "risk", 3),
        ]
        
        total_precision = 0
        total_recall = 0
        
        for query, expected_topic, expected_count in test_cases:
            result = mem.recall(query, max_tokens=2000)
            
            # Parse returned paths from result
            returned_topics = []
            for line in result.split('\n'):
                if line.startswith('## ') or line.startswith('### '):
                    # Extract topic from title
                    for path, topic in memory_map.items():
                        if topic in line.lower():
                            returned_topics.append(topic)
                            break
            
            # Calculate precision and recall
            correct = sum(1 for t in returned_topics if t == expected_topic)
            precision = correct / len(returned_topics) if returned_topics else 0
            recall = correct / expected_count
            
            total_precision += precision
            total_recall += recall
        
        avg_precision = total_precision / len(test_cases)
        avg_recall = total_recall / len(test_cases)
        f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        self.results.append(QualityResult(
            name="Recall Precision",
            score=f1,
            details={
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": f1,
            }
        ))
        print(f"      F1 Score: {f1:.2%}\n")
    
    def bench_token_efficiency(self):
        """
        Test: How much token savings does AVM provide?
        
        Method:
        1. Create N memories totaling X tokens
        2. Recall with budget B
        3. Measure: returned_tokens / total_tokens
        4. Compare: if no AVM, would need full context
        """
        print("[2/4] Token Efficiency...")
        
        avm = AVM()
        mem = avm.agent_memory("token_test")
        
        # Create memories of varying size
        total_chars = 0
        for i in range(100):
            # Varying content sizes
            size = random.randint(100, 500)
            content = f"Memory {i}: " + " ".join(["word"] * (size // 5))
            total_chars += len(content)
            mem.remember(content, title=f"mem_{i}", importance=random.uniform(0.3, 1.0))
        
        total_tokens = total_chars // 4  # Rough estimate
        
        # Test different budgets
        budgets = [500, 1000, 2000, 4000]
        efficiency_data = []
        
        for budget in budgets:
            result = mem.recall("memory content", max_tokens=budget)
            returned_tokens = len(result) // 4
            
            # Compression ratio: how much we reduced
            compression = 1 - (returned_tokens / total_tokens)
            # Efficiency: got useful content within budget
            utilization = returned_tokens / budget
            
            efficiency_data.append({
                "budget": budget,
                "returned": returned_tokens,
                "compression": compression,
                "utilization": utilization,
            })
        
        avg_compression = sum(e["compression"] for e in efficiency_data) / len(efficiency_data)
        
        self.results.append(QualityResult(
            name="Token Efficiency",
            score=avg_compression,
            details={
                "total_tokens": total_tokens,
                "budgets": efficiency_data,
                "avg_compression": avg_compression,
            }
        ))
        print(f"      Compression: {avg_compression:.1%} (tokens saved)\n")
    
    def bench_relevance_ranking(self):
        """
        Test: Are more relevant memories ranked higher?
        
        Method:
        1. Create memories with known relevance to query
        2. Check if high-relevance items appear first
        3. Measure NDCG (Normalized Discounted Cumulative Gain)
        """
        print("[3/4] Relevance Ranking...")
        
        avm = AVM()
        mem = avm.agent_memory("ranking_test")
        
        # Create memories with explicit relevance
        # Query will be "NVIDIA stock analysis"
        memories = [
            ("NVIDIA Q4 earnings beat expectations, stock up 5%", 1.0),  # Highly relevant
            ("NVIDIA announces new AI chip architecture", 0.9),
            ("Tech stocks rally on AI optimism", 0.6),
            ("AMD reports strong GPU sales", 0.4),  # Related but different
            ("Apple iPhone sales decline", 0.1),  # Irrelevant
            ("Weather forecast for tomorrow", 0.0),  # Completely irrelevant
        ]
        
        path_relevance = {}
        for content, relevance in memories:
            node = mem.remember(content, title=f"doc_{len(path_relevance)}", importance=0.5)
            path_relevance[content[:30]] = relevance  # Use content prefix as key
        
        # Recall and check order
        result = mem.recall("NVIDIA stock analysis", max_tokens=2000)
        
        # Extract order from result
        found_relevances = []
        for content_prefix, relevance in path_relevance.items():
            if content_prefix in result:
                pos = result.find(content_prefix)
                found_relevances.append((pos, relevance))
        
        # Sort by position and get relevance scores
        found_relevances.sort(key=lambda x: x[0])
        ranking = [r for _, r in found_relevances]
        
        # Calculate NDCG
        def dcg(relevances):
            return sum(rel / (i + 2) for i, rel in enumerate(relevances))  # log2(i+2)
        
        ideal_ranking = sorted(ranking, reverse=True)
        actual_dcg = dcg(ranking)
        ideal_dcg = dcg(ideal_ranking)
        ndcg = actual_dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        self.results.append(QualityResult(
            name="Relevance Ranking",
            score=ndcg,
            details={
                "ndcg": ndcg,
                "ranking": ranking,
                "ideal": ideal_ranking,
            }
        ))
        print(f"      NDCG: {ndcg:.2%}\n")
    
    def bench_importance_filtering(self):
        """
        Test: Does importance scoring affect recall correctly?
        
        Method:
        1. Create memories with different importance
        2. With limited budget, high-importance should be prioritized
        """
        print("[4/4] Importance Filtering...")
        
        avm = AVM()
        mem = avm.agent_memory("importance_test")
        
        # Create memories with varying importance
        high_importance = []
        low_importance = []
        
        for i in range(20):
            if i < 10:
                node = mem.remember(f"High priority item {i}", 
                                   title=f"high_{i}", importance=0.9)
                high_importance.append(f"High priority item {i}")
            else:
                node = mem.remember(f"Low priority item {i}", 
                                   title=f"low_{i}", importance=0.2)
                low_importance.append(f"Low priority item {i}")
        
        # Recall with tight budget - should prioritize high importance
        result = mem.recall("priority item", max_tokens=500)
        
        high_found = sum(1 for h in high_importance if h in result)
        low_found = sum(1 for l in low_importance if l in result)
        
        # Score: ratio of high vs low importance returned
        total_found = high_found + low_found
        importance_ratio = high_found / total_found if total_found > 0 else 0
        
        self.results.append(QualityResult(
            name="Importance Filtering",
            score=importance_ratio,
            details={
                "high_found": high_found,
                "low_found": low_found,
                "ratio": importance_ratio,
            }
        ))
        print(f"      High Priority Ratio: {importance_ratio:.1%}\n")
    
    def print_report(self):
        print("="*60)
        print("QUALITY REPORT")
        print("="*60 + "\n")
        
        print(f"{'Metric':<25} {'Score':>10} {'Details'}")
        print("-"*60)
        
        for r in self.results:
            print(f"{r.name:<25} {r.score:>9.1%}")
            for k, v in r.details.items():
                if isinstance(v, float):
                    print(f"  └─ {k}: {v:.2%}")
                elif isinstance(v, list) and len(v) <= 5:
                    print(f"  └─ {k}: {v}")
                elif isinstance(v, dict):
                    continue  # Skip nested dicts
                else:
                    print(f"  └─ {k}: {v}")
        
        # Overall score
        overall = sum(r.score for r in self.results) / len(self.results)
        print("\n" + "="*60)
        print(f"Overall Quality Score: {overall:.1%}")
        print("="*60 + "\n")


if __name__ == "__main__":
    bench = QualityBenchmark()
    bench.run_all()
