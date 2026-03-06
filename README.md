# AVM Benchmark Suite

Performance benchmarks for [AVM](https://github.com/aivmem/avm) (AI Virtual Memory).

## Key Result: 89% Token Savings

AVM trades minimal compute for massive token reduction:

```
======================================================================
Token Usage Comparison:
----------------------------------------------------------------------
Benchmark              AVM Tokens     Baseline    Reduction
----------------------------------------------------------------------
Recall                       1050         3600        +2550
Context Building               16         1953        +1937
Knowledge Lookup               69         2458        +2389
----------------------------------------------------------------------
TOTAL                        1135         8011        +6876

💰 Token savings: 89% = lower LLM costs
======================================================================
```

## Benchmarks

| Suite | Description |
|-------|-------------|
| `bench.py` | Performance throughput |
| `quality_bench.py` | Recall quality metrics |
| `scenario_bench.py` | Real-world scenarios |
| `compare_bench.py` | AVM vs baseline comparison |

## Usage

```bash
# Performance benchmarks
python bench.py -s 100

# Quality benchmarks  
python quality_bench.py -s 50

# Scenario benchmarks
python scenario_bench.py

# Comparison (key metric: token savings)
python compare_bench.py
```

## Performance Results (scale=50)

```
============================================================
BENCHMARK REPORT
============================================================

Benchmark                 Ops      ops/s      Avg      P50      P95      P99
-------------------- -------- ---------- -------- -------- -------- --------
Write Throughput           50     1198.9    0.83ms    0.81ms    1.01ms    0.00ms
Read Throughput            50    11697.3    0.08ms    0.08ms    0.09ms    0.00ms
Recall Performance         50     2914.0    0.34ms    0.13ms    0.94ms    0.00ms
Knowledge Graph            50     2736.7    0.36ms    0.35ms    0.45ms    0.00ms
Multi-Agent                48      622.4    5.39ms    0.94ms   29.38ms    0.00ms

============================================================
Summary:
  Total operations: 248
  Overall throughput: 1564.4 ops/sec
============================================================
```

## Scenario Results

```
======================================================================
SCENARIO REPORT
======================================================================

Scenario                   Success     Metric         Time
----------------------------------------------------------------------
Graph Discovery                  ✓          4       36.3ms
  └─ hops_to_target: 4
  └─ ms_per_hop: 9.08
Multi-Agent Isolation            ✓       1.00        2.7ms
  └─ isolation_correct: True
Cold Start                       ✓          6       98.8ms
  └─ coverage: 100%
Chain Reasoning                  ✓       1.00       22.9ms
  └─ completeness: 100%

======================================================================
Scenario Success Rate: 100%
======================================================================
```

## Metrics

### Performance (`bench.py`)

| Metric | Description |
|--------|-------------|
| Write Throughput | Bulk memory creation |
| Read Throughput | Sequential/random reads |
| Recall Performance | Semantic search |
| Knowledge Graph | Link creation/traversal |
| Multi-Agent | Concurrent access |

### Quality (`quality_bench.py`)

| Metric | Description |
|--------|-------------|
| Recall Precision | F1 score |
| Token Efficiency | Compression ratio |
| Relevance Ranking | NDCG |
| Importance Filtering | High priority ratio |

### Scenarios (`scenario_bench.py`)

| Scenario | Description |
|----------|-------------|
| Graph Discovery | Navigate links without keyword match |
| Multi-Agent Isolation | Private memories stay private |
| Cold Start | Discover KB structure |
| Chain Reasoning | Reconstruct derivation chains |

### Comparison (`compare_bench.py`)

| Metric | Description |
|--------|-------------|
| Token Savings | % reduction vs dumping all content |
| Time | Compute overhead |

## Requirements

- Python 3.10+
- AVM installed (`pip install -e path/to/avm`)

## License

MIT
