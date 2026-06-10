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