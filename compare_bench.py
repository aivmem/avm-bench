#!/usr/bin/env python3
"""
AVM Comparison Benchmark

Compares AVM vs naive baseline across multiple rounds.

Key insight: AVM trades CPU time for TOKEN SAVINGS.
- Baseline is faster (pure memory scan)
- AVM is smarter (returns less, more relevant content)
- TOKEN SAVINGS is the key metric for LLM cost reduction
"""

import os
import sys
import time
import tempfile
import random
import statistics
import argparse
from dataclasses import dataclass, field
from typing import List, Dict

sys.path.insert(0, os.path.expanduser("~/.openclaw/workspace/vfs"))
os.environ["XDG_DATA_HOME"] = tempfile.mkdtemp()

from avm import AVM


@dataclass
class RoundResult:
    avm_time_ms: float
    baseline_time_ms: float
    avm_tokens: int
    baseline_tokens: int


@dataclass
class CompareResult:
    name: str
    rounds: int
    avm_avg_ms: float
    baseline_avg_ms: float
    speedup: float  # baseline / avm
    avm_avg_tokens: int
    baseline_avg_tokens: int
    token_savings: float  # 1 - (avm / baseline)
    details: Dict = field(default_factory=dict)


class ComparisonBenchmark:
    def __init__(self, rounds: int = 5, scale: int = 100, verbose: bool = False):
        self.rounds = rounds
        self.scale = scale
        self.verbose = verbose
        self.results: List[CompareResult] = []
    
    def log(self, msg: str):
        if self.verbose:
            print(f"    {msg}")
    
    def run_all(self):
        print(f"\n{'='*70}")
        print(f"AVM vs Baseline Comparison (rounds={self.rounds}, scale={self.scale})")
        print(f"{'='*70}\n")
        
        self.compare_recall()
        self.compare_context_building()
        self.compare_knowledge_lookup()
        
        self.print_report()
    
    def compare_recall(self):
        """
        Compare: AVM recall vs naive full-text search
        
        Baseline: Load all memories, filter by keyword, return all matches
        AVM: Smart recall with token budget and ranking
        """
        print(f"[1/3] Recall Performance ({self.rounds} rounds)...")
        
        round_results: List[RoundResult] = []
        
        for r in range(self.rounds):
            self.log(f"Round {r+1}/{self.rounds}")
            
            # Fresh DB each round
            os.environ["XDG_DATA_HOME"] = tempfile.mkdtemp()
            avm = AVM()
            mem = avm.agent_memory("recall_test")
            
            # Create memories
            all_content = []
            for i in range(self.scale):
                content = f"Memory {i}: " + " ".join(
                    random.choices(["trading", "signal", "market", "risk", "alpha"], k=20)
                )
                mem.remember(content, title=f"mem_{i}", importance=random.uniform(0.3, 1.0))
                all_content.append(content)
            
            query = "trading signal alpha"
            
            token_budget = 1000
            
            # Baseline: naive full scan (returns ALL matches, no budget)
            t0 = time.perf_counter()
            baseline_results = []
            for content in all_content:
                if any(word in content.lower() for word in query.split()):
                    baseline_results.append(content)
            baseline_context = "\n\n".join(baseline_results)
            baseline_time = (time.perf_counter() - t0) * 1000
            baseline_tokens = len(baseline_context) // 4  # All matches, no limit
            
            # AVM: smart recall with budget (returns top relevant within budget)
            t0 = time.perf_counter()
            avm_context = mem.recall(query, max_tokens=token_budget)
            avm_time = (time.perf_counter() - t0) * 1000
            avm_tokens = len(avm_context) // 4
            
            round_results.append(RoundResult(
                avm_time_ms=avm_time,
                baseline_time_ms=baseline_time,
                avm_tokens=avm_tokens,
                baseline_tokens=baseline_tokens,
            ))
        
        self._record_result("Recall", round_results)
    
    def compare_context_building(self):
        """
        Compare: Building context for LLM prompt
        
        Baseline: Concatenate all recent memories
        AVM: Token-budget aware context with importance ranking
        """
        print(f"[2/3] Context Building ({self.rounds} rounds)...")
        
        round_results: List[RoundResult] = []
        
        for r in range(self.rounds):
            self.log(f"Round {r+1}/{self.rounds}")
            
            os.environ["XDG_DATA_HOME"] = tempfile.mkdtemp()
            avm = AVM()
            mem = avm.agent_memory("context_test")
            
            # Create memories with varying importance
            all_memories = []
            for i in range(self.scale):
                importance = 0.9 if i % 10 == 0 else 0.3  # 10% are important
                content = f"{'IMPORTANT ' if importance > 0.5 else ''}Note {i}: details here " * 10
                mem.remember(content, title=f"note_{i}", importance=importance)
                all_memories.append((content, importance))
            
            token_budget = 2000
            
            # Baseline: just take most recent until budget
            t0 = time.perf_counter()
            baseline_context = ""
            for content, _ in reversed(all_memories):
                if len(baseline_context) // 4 + len(content) // 4 > token_budget:
                    break
                baseline_context += content + "\n"
            baseline_time = (time.perf_counter() - t0) * 1000
            baseline_tokens = len(baseline_context) // 4
            
            # Count important items in baseline
            baseline_important = baseline_context.count("IMPORTANT")
            
            # AVM: importance-aware recall
            t0 = time.perf_counter()
            avm_context = mem.recall("notes details", max_tokens=token_budget, strategy="importance")
            avm_time = (time.perf_counter() - t0) * 1000
            avm_tokens = len(avm_context) // 4
            avm_important = avm_context.count("IMPORTANT")
            
            round_results.append(RoundResult(
                avm_time_ms=avm_time,
                baseline_time_ms=baseline_time,
                avm_tokens=avm_tokens,
                baseline_tokens=baseline_tokens,
            ))
        
        self._record_result("Context Building", round_results)
    
    def compare_knowledge_lookup(self):
        """
        Compare: Finding specific knowledge
        
        Baseline: Linear scan through all memories
        AVM: Graph-based navigation + semantic search
        """
        print(f"[3/3] Knowledge Lookup ({self.rounds} rounds)...")
        
        round_results: List[RoundResult] = []
        
        for r in range(self.rounds):
            self.log(f"Round {r+1}/{self.rounds}")
            
            os.environ["XDG_DATA_HOME"] = tempfile.mkdtemp()
            avm = AVM()
            mem = avm.agent_memory("lookup_test")
            
            # Create memories with one target
            target_idx = random.randint(0, self.scale - 1)
            all_content = []
            
            for i in range(self.scale):
                if i == target_idx:
                    content = "TARGET_FOUND: This is the specific answer we need"
                else:
                    content = f"Filler content {i} with random words " * 5
                mem.remember(content, title=f"item_{i}")
                all_content.append(content)
            
            # Baseline: linear scan
            t0 = time.perf_counter()
            baseline_result = None
            baseline_scanned = 0
            for content in all_content:
                baseline_scanned += 1
                if "TARGET_FOUND" in content:
                    baseline_result = content
                    break
            baseline_time = (time.perf_counter() - t0) * 1000
            baseline_tokens = sum(len(c) // 4 for c in all_content[:baseline_scanned])
            
            # AVM: semantic search
            t0 = time.perf_counter()
            avm_result = mem.recall("specific answer target", max_tokens=500)
            avm_time = (time.perf_counter() - t0) * 1000
            avm_tokens = len(avm_result) // 4
            
            round_results.append(RoundResult(
                avm_time_ms=avm_time,
                baseline_time_ms=baseline_time,
                avm_tokens=avm_tokens,
                baseline_tokens=baseline_tokens,
            ))
        
        self._record_result("Knowledge Lookup", round_results)
    
    def _record_result(self, name: str, rounds: List[RoundResult]):
        avm_times = [r.avm_time_ms for r in rounds]
        baseline_times = [r.baseline_time_ms for r in rounds]
        avm_tokens = [r.avm_tokens for r in rounds]
        baseline_tokens = [r.baseline_tokens for r in rounds]
        
        avm_avg_ms = statistics.mean(avm_times)
        baseline_avg_ms = statistics.mean(baseline_times)
        # Speedup: positive = AVM faster, negative = baseline faster
        if avm_avg_ms > 0 and baseline_avg_ms > 0:
            speedup = baseline_avg_ms / avm_avg_ms
        else:
            speedup = 1.0
        
        avm_avg_tokens = int(statistics.mean(avm_tokens))
        baseline_avg_tokens = int(statistics.mean(baseline_tokens))
        token_savings = 1 - (avm_avg_tokens / baseline_avg_tokens) if baseline_avg_tokens > 0 else 0
        
        self.results.append(CompareResult(
            name=name,
            rounds=len(rounds),
            avm_avg_ms=avm_avg_ms,
            baseline_avg_ms=baseline_avg_ms,
            speedup=speedup,
            avm_avg_tokens=avm_avg_tokens,
            baseline_avg_tokens=baseline_avg_tokens,
            token_savings=token_savings,
            details={
                "avm_p50_ms": statistics.median(avm_times),
                "baseline_p50_ms": statistics.median(baseline_times),
                "avm_stddev": statistics.stdev(avm_times) if len(avm_times) > 1 else 0,
            }
        ))
        
        print(f"      Speedup: {speedup:.1f}x, Token savings: {token_savings:.0%}\n")
    
    def print_report(self):
        print("="*70)
        print("COMPARISON REPORT")
        print("="*70)
        print("\n⚡ AVM trades compute time for TOKEN SAVINGS (= LLM cost reduction)\n")
        
        # Token comparison is the key metric
        print(f"{'Benchmark':<20} {'AVM Tokens':>12} {'Baseline':>12} {'Saved':>10} {'Time':>10}")
        print("-"*70)
        
        for r in self.results:
            saved_pct = f"{r.token_savings:.0%}" if r.token_savings > 0 else f"{r.token_savings:.0%}"
            print(f"{r.name:<20} {r.avm_avg_tokens:>12} {r.baseline_avg_tokens:>12} "
                  f"{saved_pct:>10} {r.avm_avg_ms:>8.1f}ms")
        
        print("-"*70)
        
        avg_token_savings = statistics.mean(r.token_savings for r in self.results)
        
        print("\n" + "="*70)
        print("Token Usage Comparison:")
        print("-"*70)
        print(f"{'Benchmark':<20} {'AVM Tokens':>12} {'Baseline':>12} {'Reduction':>12}")
        print("-"*70)
        
        for r in self.results:
            reduction = r.baseline_avg_tokens - r.avm_avg_tokens
            print(f"{r.name:<20} {r.avm_avg_tokens:>12} {r.baseline_avg_tokens:>12} {reduction:>+12}")
        
        total_avm = sum(r.avm_avg_tokens for r in self.results)
        total_baseline = sum(r.baseline_avg_tokens for r in self.results)
        total_reduction = total_baseline - total_avm
        
        print("-"*70)
        print(f"{'TOTAL':<20} {total_avm:>12} {total_baseline:>12} {total_reduction:>+12}")
        print(f"\n💰 Token savings: {avg_token_savings:.0%} = lower LLM costs")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AVM vs Baseline Comparison")
    parser.add_argument("-r", "--rounds", type=int, default=5,
                        help="Number of rounds per benchmark (default: 5)")
    parser.add_argument("-s", "--scale", type=int, default=100,
                        help="Data scale factor (default: 100)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    
    bench = ComparisonBenchmark(rounds=args.rounds, scale=args.scale, verbose=args.verbose)
    bench.run_all()


if __name__ == "__main__":
    main()
