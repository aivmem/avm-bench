# AVM Benchmark Suite

Performance benchmarks for [AVM](https://github.com/aivmem/avm) (AI Virtual Memory).

## Scenarios

| Benchmark | Description |
|-----------|-------------|
| Write Throughput | Bulk memory creation with tags and importance |
| Read Throughput | Sequential and random path reads |
| Recall Performance | Semantic search with token budgets |
| Knowledge Graph | Link creation and graph traversal |
| Multi-Agent | Concurrent access from 4 agents |

## Usage

```bash
# Default (scale=100)
python bench.py

# Larger scale
python bench.py -s 1000

# Verbose
python bench.py -s 500 -v
```

## Sample Output

```
============================================================
AVM Benchmark Suite (scale=100)
============================================================

[1/5] Write Throughput (100 memories)...
      Done: 523.4 ops/sec

[2/5] Read Throughput (100 reads)...
      Done: 12453.2 ops/sec

[3/5] Recall Performance (100 queries)...
      Done: 89.3 ops/sec

[4/5] Knowledge Graph (100 operations)...
      Done: 1234.5 ops/sec

[5/5] Multi-Agent Concurrency (100 ops, 4 agents)...
      Done: 456.7 ops/sec

============================================================
BENCHMARK REPORT
============================================================

Benchmark            Ops      ops/s      Avg      P50      P95      P99
-------------------- -------- ---------- -------- -------- -------- --------
Write Throughput          100      523.4    1.91ms   1.85ms   2.34ms   2.89ms
Read Throughput           100    12453.2    0.08ms   0.07ms   0.12ms   0.15ms
Recall Performance        100       89.3   11.20ms  10.85ms  15.23ms  18.45ms
Knowledge Graph           100     1234.5    0.81ms   0.78ms   1.12ms   1.35ms
Multi-Agent               100      456.7    2.19ms   2.05ms   3.45ms   4.12ms

============================================================
Summary:
  Total operations: 500
  Total time: 1.23s
  Overall throughput: 406.5 ops/sec
============================================================
```

## Requirements

- Python 3.10+
- AVM installed (`pip install -e path/to/avm`)

## License

MIT
