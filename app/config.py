import os

DOCS_DIR = os.getenv("DOCS_DIR", "/mnt/external/rag-project/docs")

POSTGRES_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "user": "postgres",
    "password": "mysecretpassword",
    "database": "rag_db",
    "table_name": "granite_bgem3_cli",
    "embed_dim": 1024,
}

LLM_CONFIG = {
    "model": "ibm-granite/granite-4.1-8b",
    "api_base": "http://localhost:8000/v1",
    "api_key": "EMPTY",
}

JUDGE_CONFIG = {
    "model": "deepseek-v4-flash",
    "api_base": "https://api.deepseek.com",
    "api_key": os.getenv("API_KEY", ""),
    "temperature": 0.0,
    "thinking": "disabled",
}

RETRIEVAL_CONFIG = {
    "dense_top_k": 10,
    "sparse_top_k": 10,
    "fusion_top_k": 10,
    "num_queries": 4,
    "fusion_mode": "reciprocal_rerank",
    "rerank_on": True,
    "rerank_top_n": 5,
}