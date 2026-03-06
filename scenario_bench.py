#!/usr/bin/env python3
"""
AVM Scenario Benchmarks

Real-world scenarios:
1. Discovery - Find target through graph navigation, not keyword match
2. Collaboration - Multi-agent shared memory efficiency
3. Cold Start - New agent learning existing knowledge base
"""

import os
import sys
import tempfile
import random
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple

sys.path.insert(0, os.path.expanduser("~/.openclaw/workspace/vfs"))
os.environ["XDG_DATA_HOME"] = tempfile.mkdtemp()

from avm import AVM
from avm.graph import EdgeType


def generate_entity_id() -> str:
    """Generate opaque ID that can't be keyword searched."""
    return hashlib.md5(str(random.random()).encode()).hexdigest()[:8]


@dataclass 
class ScenarioResult:
    name: str
    success: bool
    metric: float
    details: Dict


class ScenarioBenchmark:
    def __init__(self):
        self.results: List[ScenarioResult] = []
    
    def run_all(self):
        print("\n" + "="*60)
        print("AVM Scenario Benchmarks")
        print("="*60 + "\n")
        
        self.bench_graph_discovery()
        self.bench_multi_agent_collab()
        self.bench_cold_start()
        self.bench_chain_reasoning()
        
        self.print_report()
    
    def bench_graph_discovery(self):
        """
        Scenario: Find a target through graph links, not keywords.
        
        Setup:
        - Create entities with opaque IDs (no keyword match possible)
        - Link them: A -> B -> C -> TARGET
        - Agent must navigate graph to find TARGET
        
        Metric: Number of hops to find target
        """
        print("[1/4] Graph Discovery (no keyword cheating)...")
        
        avm = AVM()
        mem = avm.agent_memory("explorer")
        
        # Create a knowledge graph with opaque entities
        # Structure: entry_point -> sector -> company -> target_insight
        
        entry_id = generate_entity_id()
        sector_ids = [generate_entity_id() for _ in range(3)]
        company_ids = [generate_entity_id() for _ in range(9)]  # 3 per sector
        target_id = generate_entity_id()
        
        # Entry point - this is the only searchable content
        entry = mem.remember(
            f"Market overview for Q4. Sectors: {', '.join(sector_ids[:3])}",
            title="market_entry",
            tags=["entry"]
        )
        
        # Sectors (linked from entry)
        sector_paths = []
        for i, sid in enumerate(sector_ids):
            node = mem.remember(
                f"Sector {sid}: companies {', '.join(company_ids[i*3:(i+1)*3])}",
                title=f"sector_{sid}"
            )
            sector_paths.append(node.path)
            avm.link(entry.path, node.path, EdgeType.PARENT)
        
        # Companies (linked from sectors)
        company_paths = []
        for i, cid in enumerate(company_ids):
            sector_idx = i // 3
            node = mem.remember(
                f"Company {cid}: financial data encrypted",
                title=f"company_{cid}"
            )
            company_paths.append(node.path)
            avm.link(sector_paths[sector_idx], node.path, EdgeType.PARENT)
        
        # Target insight (linked from one company)
        target_company = random.choice(company_paths)
        target = mem.remember(
            f"TARGET_INSIGHT_{target_id}: Critical alpha signal discovered",
            title=f"insight_{target_id}"
        )
        avm.link(target_company, target.path, EdgeType.DERIVED)
        
        # Now try to find target through navigation
        hops = 0
        found = False
        visited: Set[str] = set()
        current_paths = [entry.path]
        
        while hops < 10 and not found:
            hops += 1
            next_paths = []
            
            for path in current_paths:
                if path in visited:
                    continue
                visited.add(path)
                
                # Check if this is target
                node = avm.read(path)
                if node and "TARGET_INSIGHT" in node.content:
                    found = True
                    break
                
                # Get linked nodes
                links = avm.links(path, direction="out")
                for link in links:
                    if link.target not in visited:
                        next_paths.append(link.target)
            
            if not found:
                current_paths = next_paths
        
        self.results.append(ScenarioResult(
            name="Graph Discovery",
            success=found,
            metric=hops if found else -1,
            details={
                "hops_to_target": hops,
                "nodes_visited": len(visited),
                "found": found,
            }
        ))
        print(f"      Found in {hops} hops, visited {len(visited)} nodes\n")
    
    def bench_multi_agent_collab(self):
        """
        Scenario: Multiple agents with isolated memories.
        
        Setup:
        - Agent A and B each store private memories
        - Check isolation: A cannot see B's memories
        - Check: each can see their own
        
        Metric: Isolation correctness
        """
        print("[2/4] Multi-Agent Isolation...")
        
        avm = AVM()
        agent_a = avm.agent_memory("agent_a")
        agent_b = avm.agent_memory("agent_b")
        
        # Each agent stores private data
        secret_a = generate_entity_id()
        secret_b = generate_entity_id()
        
        agent_a.remember(f"SECRET_A_{secret_a}: confidential data", title="secret_a")
        agent_b.remember(f"SECRET_B_{secret_b}: confidential data", title="secret_b")
        
        # A recalls - should see own, not B's
        a_recall = agent_a.recall("SECRET confidential", max_tokens=2000)
        a_sees_own = f"SECRET_A_{secret_a}" in a_recall
        a_sees_other = f"SECRET_B_{secret_b}" in a_recall
        
        # B recalls - should see own, not A's
        b_recall = agent_b.recall("SECRET confidential", max_tokens=2000)
        b_sees_own = f"SECRET_B_{secret_b}" in b_recall
        b_sees_other = f"SECRET_A_{secret_a}" in b_recall
        
        isolation_correct = (a_sees_own and not a_sees_other and 
                            b_sees_own and not b_sees_other)
        
        self.results.append(ScenarioResult(
            name="Multi-Agent Isolation",
            success=isolation_correct,
            metric=1.0 if isolation_correct else 0.0,
            details={
                "a_sees_own": a_sees_own,
                "a_leak": a_sees_other,
                "b_sees_own": b_sees_own,
                "b_leak": b_sees_other,
                "isolation_correct": isolation_correct,
            }
        ))
        print(f"      Isolation: {'✓' if isolation_correct else '✗'}\n")
    
    def bench_cold_start(self):
        """
        Scenario: Agent discovering its own knowledge base.
        
        Setup:
        - Create 100 memories across topics
        - Use topics() to discover structure
        - Measure: how well topics() maps the space
        
        Metric: Topic discovery coverage
        """
        print("[3/4] Cold Start Discovery...")
        
        avm = AVM()
        mem = avm.agent_memory("cold_start")
        
        # Create knowledge base with tags
        topics = ["trading", "risk", "macro", "technical", "fundamentals"]
        topic_counts = {t: 0 for t in topics}
        
        for i in range(100):
            topic = random.choice(topics)
            mem.remember(
                f"{topic.upper()}: data point {i} with metrics",
                title=f"{topic}_{i}",
                tags=[topic]
            )
            topic_counts[topic] += 1
        
        # Discovery process
        queries = 0
        discovered_topics: Set[str] = set()
        
        # Strategy: use topics() to discover
        topics_str = mem.topics()
        queries += 1
        
        # Parse topics string for our known tags
        for topic in topics:
            if topic in topics_str.lower():
                discovered_topics.add(topic)
        
        # Follow up queries to explore each topic
        for topic in list(discovered_topics)[:5]:
            mem.recall(f"{topic} data metrics", max_tokens=500)
            queries += 1
        
        coverage = len(discovered_topics) / len(topics)
        
        self.results.append(ScenarioResult(
            name="Cold Start",
            success=coverage > 0.8,
            metric=queries,
            details={
                "queries_used": queries,
                "topics_discovered": len(discovered_topics),
                "total_topics": len(topics),
                "coverage": coverage,
            }
        ))
        print(f"      {queries} queries for {coverage:.0%} coverage\n")
    
    def bench_chain_reasoning(self):
        """
        Scenario: Following a chain of derived insights.
        
        Setup:
        - Create: observation -> analysis -> insight -> action
        - Each step derives from previous
        - Agent must reconstruct reasoning chain
        
        Metric: Chain completeness
        """
        print("[4/4] Chain Reasoning...")
        
        avm = AVM()
        mem = avm.agent_memory("reasoner")
        
        # Build reasoning chains
        chains_complete = 0
        total_chains = 5
        
        for chain_id in range(total_chains):
            # Create chain: obs -> analysis -> insight -> action
            cid = generate_entity_id()
            
            obs = mem.remember(
                f"OBSERVATION_{cid}: Price moved 5%",
                title=f"obs_{cid}"
            )
            
            analysis = mem.remember(
                f"ANALYSIS_{cid}: Volume confirms move",
                title=f"analysis_{cid}"
            )
            avm.link(obs.path, analysis.path, EdgeType.DERIVED)
            
            insight = mem.remember(
                f"INSIGHT_{cid}: Breakout pattern forming",
                title=f"insight_{cid}"
            )
            avm.link(analysis.path, insight.path, EdgeType.DERIVED)
            
            action = mem.remember(
                f"ACTION_{cid}: Enter position at support",
                title=f"action_{cid}"
            )
            avm.link(insight.path, action.path, EdgeType.DERIVED)
            
            # Now try to reconstruct chain from action
            current = action.path
            chain = [current]
            
            for _ in range(5):  # Max 5 hops back
                links = avm.links(current, direction="in")
                derived_links = [l for l in links if l.edge_type.value == "derived"]
                if not derived_links:
                    break
                current = derived_links[0].source
                chain.append(current)
            
            # Check if full chain reconstructed (4 nodes)
            if len(chain) >= 4:
                chains_complete += 1
        
        completeness = chains_complete / total_chains
        
        self.results.append(ScenarioResult(
            name="Chain Reasoning",
            success=completeness > 0.8,
            metric=completeness,
            details={
                "chains_complete": chains_complete,
                "total_chains": total_chains,
                "completeness": completeness,
            }
        ))
        print(f"      {chains_complete}/{total_chains} chains reconstructed\n")
    
    def print_report(self):
        print("="*60)
        print("SCENARIO REPORT")
        print("="*60 + "\n")
        
        print(f"{'Scenario':<25} {'Success':>8} {'Metric':>10}")
        print("-"*60)
        
        for r in self.results:
            status = "✓" if r.success else "✗"
            metric_str = f"{r.metric:.2f}" if isinstance(r.metric, float) else str(r.metric)
            print(f"{r.name:<25} {status:>8} {metric_str:>10}")
            for k, v in r.details.items():
                if isinstance(v, float):
                    print(f"  └─ {k}: {v:.2f}")
                else:
                    print(f"  └─ {k}: {v}")
        
        success_rate = sum(1 for r in self.results if r.success) / len(self.results)
        print("\n" + "="*60)
        print(f"Scenario Success Rate: {success_rate:.0%}")
        print("="*60 + "\n")


if __name__ == "__main__":
    bench = ScenarioBenchmark()
    bench.run_all()
