#!/usr/bin/env python3
"""
AVM Benchmark Suite

Scenarios:
1. Write throughput - bulk memory creation
2. Read throughput - sequential/random reads
3. Recall performance - semantic search at scale
4. Knowledge graph - link creation and traversal
5. Multi-agent - concurrent access patterns
"""

import os
import sys
import time
import random
import string
import tempfile
import argparse
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure AVM is importable
sys.path.insert(0, os.path.expanduser("~/.openclaw/workspace/vfs"))

os.environ["XDG_DATA_HOME"] = tempfile.mkdtemp()

from avm import AVM
from avm.graph import EdgeType


@dataclass
class BenchResult:
    name: str
    ops: int
    elapsed_sec: float
    ops_per_sec: float
    avg_latency_ms: float
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


def generate_content(size: int = 500) -> str:
    """Generate random text content."""
    words = ["memory", "knowledge", "graph", "agent", "recall", "token",
             "semantic", "vector", "embedding", "context", "important",
             "trading", "signal", "market", "analysis", "pattern"]
    return " ".join(random.choices(words, k=size // 5))


def generate_tags() -> List[str]:
    """Generate random tags."""
    all_tags = ["trading", "analysis", "market", "tech", "notes", 
                "important", "review", "draft", "archived", "signal"]
    return random.sample(all_tags, k=random.randint(1, 3))


class Benchmark:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[BenchResult] = []
    
    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")
    
    def run_all(self, scale: int = 100):
        """Run all benchmarks."""
        print(f"\n{'='*60}")
        print(f"AVM Benchmark Suite (scale={scale})")
        print(f"{'='*60}\n")
        
        self.bench_write_throughput(scale)
        self.bench_read_throughput(scale)
        self.bench_recall_performance(scale)
        self.bench_knowledge_graph(scale)
        self.bench_multi_agent(scale)
        
        self.print_report()
    
    def bench_write_throughput(self, n: int):
        """Benchmark: Write throughput."""
        print(f"[1/5] Write Throughput ({n} memories)...")
        
        avm = AVM()
        mem = avm.agent_memory("bench_writer")
        latencies = []
        
        start = time.perf_counter()
        for i in range(n):
            content = generate_content(300)
            tags = generate_tags()
            importance = random.uniform(0.3, 1.0)
            
            t0 = time.perf_counter()
            mem.remember(
                content,
                title=f"memory_{i:05d}",
                tags=tags,
                importance=importance
            )
            latencies.append((time.perf_counter() - t0) * 1000)
        
        elapsed = time.perf_counter() - start
        
        self.results.append(BenchResult(
            name="Write Throughput",
            ops=n,
            elapsed_sec=elapsed,
            ops_per_sec=n / elapsed,
            avg_latency_ms=statistics.mean(latencies),
            p50_ms=statistics.median(latencies),
            p95_ms=sorted(latencies)[int(n * 0.95)] if n > 20 else 0,
            p99_ms=sorted(latencies)[int(n * 0.99)] if n > 100 else 0,
        ))
        print(f"      Done: {n/elapsed:.1f} ops/sec\n")
    
    def bench_read_throughput(self, n: int):
        """Benchmark: Read throughput."""
        print(f"[2/5] Read Throughput ({n} reads)...")
        
        avm = AVM()
        mem = avm.agent_memory("bench_reader")
        
        # Create some memories first
        paths = []
        for i in range(min(n, 100)):
            node = mem.remember(generate_content(200), title=f"read_test_{i}")
            paths.append(node.path)
        
        latencies = []
        start = time.perf_counter()
        
        for _ in range(n):
            path = random.choice(paths)
            t0 = time.perf_counter()
            avm.read(path)
            latencies.append((time.perf_counter() - t0) * 1000)
        
        elapsed = time.perf_counter() - start
        
        self.results.append(BenchResult(
            name="Read Throughput",
            ops=n,
            elapsed_sec=elapsed,
            ops_per_sec=n / elapsed,
            avg_latency_ms=statistics.mean(latencies),
            p50_ms=statistics.median(latencies),
            p95_ms=sorted(latencies)[int(n * 0.95)] if n > 20 else 0,
            p99_ms=sorted(latencies)[int(n * 0.99)] if n > 100 else 0,
        ))
        print(f"      Done: {n/elapsed:.1f} ops/sec\n")
    
    def bench_recall_performance(self, n: int):
        """Benchmark: Recall (semantic search) performance."""
        print(f"[3/5] Recall Performance ({n} queries)...")
        
        avm = AVM()
        mem = avm.agent_memory("bench_recall")
        
        # Create diverse memories
        topics = ["trading signals", "market analysis", "risk management",
                  "technical indicators", "portfolio strategy", "macro trends"]
        
        for i in range(min(n, 200)):
            topic = random.choice(topics)
            content = f"{topic}: {generate_content(200)}"
            mem.remember(content, title=f"topic_{i}", importance=random.uniform(0.5, 1.0))
        
        queries = ["trading signals recent", "risk analysis", "market trends",
                   "portfolio allocation", "technical setup", "macro outlook"]
        
        latencies = []
        tokens_returned = []
        
        start = time.perf_counter()
        for _ in range(n):
            query = random.choice(queries)
            t0 = time.perf_counter()
            result = mem.recall(query, max_tokens=2000)
            latencies.append((time.perf_counter() - t0) * 1000)
            tokens_returned.append(len(result) // 4)  # Rough estimate
        
        elapsed = time.perf_counter() - start
        
        self.results.append(BenchResult(
            name="Recall Performance",
            ops=n,
            elapsed_sec=elapsed,
            ops_per_sec=n / elapsed,
            avg_latency_ms=statistics.mean(latencies),
            p50_ms=statistics.median(latencies),
            p95_ms=sorted(latencies)[int(n * 0.95)] if n > 20 else 0,
            p99_ms=sorted(latencies)[int(n * 0.99)] if n > 100 else 0,
            extra={"avg_tokens": statistics.mean(tokens_returned)}
        ))
        print(f"      Done: {n/elapsed:.1f} ops/sec\n")
    
    def bench_knowledge_graph(self, n: int):
        """Benchmark: Knowledge graph operations."""
        print(f"[4/5] Knowledge Graph ({n} operations)...")
        
        avm = AVM()
        mem = avm.agent_memory("bench_graph")
        
        # Create nodes
        nodes = []
        for i in range(min(n, 100)):
            node = mem.remember(generate_content(150), title=f"node_{i}")
            nodes.append(node.path)
        
        # Create links (use avm.link not mem.link)
        link_latencies = []
        start = time.perf_counter()
        
        edge_types = [EdgeType.RELATED, EdgeType.DERIVED, EdgeType.PEER, EdgeType.CITATION]
        
        for _ in range(n):
            if len(nodes) >= 2:
                src, dst = random.sample(nodes, 2)
                edge_type = random.choice(edge_types)
                
                t0 = time.perf_counter()
                avm.link(src, dst, edge_type)
                link_latencies.append((time.perf_counter() - t0) * 1000)
        
        elapsed = time.perf_counter() - start
        
        # Test explore (via avm.links)
        explore_latencies = []
        for _ in range(min(n, 50)):
            path = random.choice(nodes)
            t0 = time.perf_counter()
            avm.links(path)
            explore_latencies.append((time.perf_counter() - t0) * 1000)
        
        self.results.append(BenchResult(
            name="Knowledge Graph",
            ops=n,
            elapsed_sec=elapsed,
            ops_per_sec=n / elapsed,
            avg_latency_ms=statistics.mean(link_latencies) if link_latencies else 0,
            p50_ms=statistics.median(link_latencies) if link_latencies else 0,
            p95_ms=sorted(link_latencies)[int(len(link_latencies) * 0.95)] if len(link_latencies) > 20 else 0,
            extra={"explore_avg_ms": statistics.mean(explore_latencies) if explore_latencies else 0}
        ))
        print(f"      Done: {n/elapsed:.1f} ops/sec\n")
    
    def bench_multi_agent(self, n: int):
        """Benchmark: Multi-agent concurrent access."""
        print(f"[5/5] Multi-Agent Concurrency ({n} ops, 4 agents)...")
        
        avm = AVM()
        agents = [avm.agent_memory(f"agent_{i}") for i in range(4)]
        
        def agent_work(agent_id: int, ops: int):
            agent = agents[agent_id]
            latencies = []
            
            for i in range(ops):
                t0 = time.perf_counter()
                if i % 3 == 0:
                    agent.remember(generate_content(100), title=f"a{agent_id}_m{i}")
                elif i % 3 == 1:
                    agent.recall("trading signals", max_tokens=1000)
                else:
                    agent.topics()
                latencies.append((time.perf_counter() - t0) * 1000)
            
            return latencies
        
        ops_per_agent = n // 4
        all_latencies = []
        
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(agent_work, i, ops_per_agent) for i in range(4)]
            for f in as_completed(futures):
                all_latencies.extend(f.result())
        
        elapsed = time.perf_counter() - start
        total_ops = len(all_latencies)
        
        self.results.append(BenchResult(
            name="Multi-Agent",
            ops=total_ops,
            elapsed_sec=elapsed,
            ops_per_sec=total_ops / elapsed,
            avg_latency_ms=statistics.mean(all_latencies),
            p50_ms=statistics.median(all_latencies),
            p95_ms=sorted(all_latencies)[int(total_ops * 0.95)] if total_ops > 20 else 0,
            p99_ms=sorted(all_latencies)[int(total_ops * 0.99)] if total_ops > 100 else 0,
            extra={"agents": 4, "ops_per_agent": ops_per_agent}
        ))
        print(f"      Done: {total_ops/elapsed:.1f} ops/sec\n")
    
    def print_report(self):
        """Print benchmark report."""
        print(f"\n{'='*60}")
        print("BENCHMARK REPORT")
        print(f"{'='*60}\n")
        
        # Table header
        print(f"{'Benchmark':<20} {'Ops':>8} {'ops/s':>10} {'Avg':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
        print(f"{'-'*20} {'-'*8} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        
        for r in self.results:
            print(f"{r.name:<20} {r.ops:>8} {r.ops_per_sec:>10.1f} "
                  f"{r.avg_latency_ms:>7.2f}ms {r.p50_ms:>7.2f}ms "
                  f"{r.p95_ms:>7.2f}ms {r.p99_ms:>7.2f}ms")
            if r.extra:
                for k, v in r.extra.items():
                    print(f"  └─ {k}: {v:.2f}" if isinstance(v, float) else f"  └─ {k}: {v}")
        
        print(f"\n{'='*60}")
        print("Summary:")
        total_ops = sum(r.ops for r in self.results)
        total_time = sum(r.elapsed_sec for r in self.results)
        print(f"  Total operations: {total_ops}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Overall throughput: {total_ops/total_time:.1f} ops/sec")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="AVM Benchmark Suite")
    parser.add_argument("-s", "--scale", type=int, default=100,
                        help="Scale factor (default: 100)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()
    
    bench = Benchmark(verbose=args.verbose)
    bench.run_all(scale=args.scale)


if __name__ == "__main__":
    main()
