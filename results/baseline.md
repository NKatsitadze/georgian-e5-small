nDCG@10 (0-100, higher is better) on the Georgian subsets of MTEB retrieval tasks.

| model | avg | Belebele eng_Latn-kat_Geor | Belebele kat_Geor-eng_Latn | Belebele kat_Geor-kat_Geor | GeorgianFAQ default | WebFAQ kat |
|---|---|---|---|---|---|---|
| Snowflake/snowflake-arctic-embed-l-v2.0 | 85.5 | 85.9 | 89.2 | 91.0 | 69.5 | 91.8 |
| BAAI/bge-m3 | 85.1 | 87.6 | 89.0 | 91.1 | 67.2 | 90.7 |
| ceiling/bge-m3-pca384 | 84.8 | 87.2 | 88.6 | 90.9 | 66.7 | 90.4 |
| models/soup-v3v6 | 78.0 | 78.7 | 82.5 | 87.8 | 56.8 | 84.0 |
| release/full | 77.9 | 78.7 | 82.5 | 87.9 | 56.7 | 84.0 |
| release/lite | 77.9 | 78.7 | 82.5 | 87.9 | 56.6 | 84.0 |
| rank-v6/student-v6-epoch1 | 77.7 | 77.8 | 82.9 | 87.7 | 56.2 | 83.8 |
| rank-v3/student-v3-epoch1 | 77.4 | 77.9 | 82.0 | 87.0 | 56.5 | 83.6 |
| rank-v3/student-v3-epoch1-len512 | 77.4 | 77.9 | 82.0 | 87.1 | 56.4 | 83.6 |
| rank-v3/student-v3-avg12 | 77.4 | 78.0 | 81.8 | 87.0 | 56.6 | 83.4 |
| rank-v3/student-v3-epoch2 | 77.2 | 77.8 | 81.5 | 86.7 | 56.6 | 83.4 |
| rank-v5/student-v5-epoch1 | 77.2 | 77.7 | 81.5 | 87.0 | 56.5 | 83.2 |
| rank-v4/student-v4-epoch1 | 77.1 | 77.7 | 81.4 | 86.9 | 56.6 | 82.8 |
| l4sha/georgian-text-embeddings | 74.4 | 72.0 | 79.4 | 82.6 | 56.6 | 81.5 |
| l4sha/georgian-text-embeddings (+e5 prefixes) | 74.0 | 71.7 | 78.9 | 82.4 | 56.5 | 80.6 |
| intfloat/multilingual-e5-large | 73.4 | 77.5 | 87.5 | 81.8 | 45.1 | 74.9 |
| distill-v2/student-epoch6 | 72.6 | 73.1 | 78.0 | 81.9 | 52.0 | 78.2 |
| distill-v2/student-epoch5 | 72.5 | 72.6 | 78.0 | 81.6 | 52.0 | 78.3 |
| distill-v2/student-epoch4 | 71.5 | 72.1 | 76.7 | 80.8 | 51.0 | 76.9 |
| altaidevorg/bge-m3-distill-8l | 70.3 | 64.9 | 76.1 | 76.6 | 53.6 | 80.3 |
| Vibels/georgian-e5-finetuned | 69.8 | 67.3 | 78.6 | 79.1 | 47.5 | 76.2 |
| distill-v2/student-epoch3 | 69.2 | 68.8 | 74.4 | 78.3 | 49.1 | 75.5 |
| microsoft/harrier-oss-v1-270m | 68.9 | 64.5 | 81.2 | 73.5 | 49.1 | 76.4 |
| Vibels/georgian-e5-finetuned (+e5 prefixes) | 68.8 | 69.4 | 77.7 | 78.1 | 44.7 | 73.9 |
| Qwen/Qwen3-Embedding-0.6B | 66.5 | 56.1 | 76.1 | 73.1 | 49.2 | 77.9 |
| distill-v2/student-epoch2 | 65.5 | 64.7 | 71.2 | 74.7 | 45.2 | 71.7 |
| intfloat/multilingual-e5-base | 64.1 | 61.3 | 78.9 | 75.6 | 38.7 | 66.2 |
| alphaedge-ai/multilingual-e5-small-kat-32768 | 61.5 | 57.5 | 72.8 | 66.9 | 41.4 | 69.1 |
| intfloat/multilingual-e5-small (no prefixes) | 61.5 | 57.2 | 72.8 | 67.0 | 41.4 | 69.1 |
| distill-v1/student-epoch2 | 60.6 | 59.4 | 66.9 | 70.8 | 40.5 | 65.6 |
| alphaedge-ai/multilingual-e5-small-kat-32768 (+e5 prefixes) | 57.5 | 54.2 | 71.4 | 68.7 | 29.0 | 64.2 |
| intfloat/multilingual-e5-small | 57.3 | 53.7 | 71.6 | 68.5 | 29.0 | 63.9 |
| distill-v2/student-epoch1 | 56.2 | 54.9 | 62.2 | 66.5 | 36.3 | 61.0 |
| distill-v1/student-epoch1 | 56.0 | 55.0 | 61.9 | 66.0 | 36.4 | 60.5 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 53.2 | 54.1 | 68.9 | 48.6 | 33.6 | 61.1 |
| codefuse-ai/F2LLM-v2-80M | 50.7 | 40.6 | 50.1 | 52.1 | 38.5 | 72.2 |
| ibm-granite/granite-embedding-97m-multilingual-r2 | 39.7 | 10.0 | 22.8 | 50.4 | 40.8 | 74.3 |
