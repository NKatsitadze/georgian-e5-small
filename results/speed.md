CPU only, 6 threads, texts capped at 512 tokens. 512 GeorgianFAQ documents (median 82 tokens), batch 32.

| model | params | dims | docs/s | × slower than fastest | query p50 ms | query p95 ms | peak RAM MB |
|---|---|---|---|---|---|---|---|
| intfloat/multilingual-e5-small | 118M | 384 | 101.8 | 1.0× | 5.9 | 6.3 | 1860 |
| intfloat/multilingual-e5-base | 278M | 768 | 36.2 | 2.8× | 17.8 | 18.2 | 2760 |
| intfloat/multilingual-e5-large | 560M | 1024 | 11.0 | 9.3× | 57.4 | 58.3 | 3961 |
| BAAI/bge-m3 | 568M | 1024 | 11.0 | 9.3× | 53.1 | 54.1 | 3959 |
| Snowflake/snowflake-arctic-embed-l-v2.0 | 568M | 1024 | 11.0 | 9.3× | 56.1 | 60.9 | 3909 |
