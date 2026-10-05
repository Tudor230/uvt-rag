import phoenix as px
from phoenix.otel import register
import nest_asyncio

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    set_global_handler,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank

from app.config import (
    POSTGRES_CONFIG,
    LLM_CONFIG,
    RETRIEVAL_CONFIG,
)
from app.prompts import query_gen_prompt
import logging
import sys



def setup_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        force=True
    )


def setup_phoenix():
    register(endpoint="http://localhost:6006/v1/traces")
    set_global_handler("arize_phoenix")


def configure_models():
    """Set global embed/LLM settings. Idempotent."""
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device="cuda",
        max_length=8192,
    )
    Settings.llm = OpenAILike(
        model=LLM_CONFIG["model"],
        api_base=LLM_CONFIG["api_base"],
        api_key=LLM_CONFIG["api_key"],
        is_chat_model=True,
    )


def build_vector_store():
    """Build the PGVectorStore client (hybrid search)."""
    return PGVectorStore.from_params(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"],
        database=POSTGRES_CONFIG["database"],
        table_name=POSTGRES_CONFIG["table_name"],
        embed_dim=POSTGRES_CONFIG["embed_dim"],
        hybrid_search=True,
        text_search_config="simple",
    )


def _postgres_conn():
    import psycopg2

    return psycopg2.connect(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"],
        database=POSTGRES_CONFIG["database"],
    )


def vector_store_row_count() -> int:
    """Number of rows currently in the vector store table."""
    conn = _postgres_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM data_{POSTGRES_CONFIG['table_name']}")
            return int(cur.fetchone()[0])
    finally:
        conn.close()


def truncate_vector_store():
    conn = _postgres_conn()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE data_{POSTGRES_CONFIG['table_name']}")
    finally:
        conn.close()
    logging.getLogger(__name__).info("Purged vector store table.")


def build_index():
    """Wrap the existing vector store in a queryable index.

    The store must be populated first (see ``python -m app.ingest``); this
    function never purges or ingests. Raises when the store is empty so
    workers fail fast instead of re-ingesting.
    """
    # -----------------------
    # Embeddings
    # -----------------------
    configure_models()

    # -----------------------
    # Vector DB
    # -----------------------
    vector_store = build_vector_store()
    count = vector_store_row_count()
    if count == 0:
        raise RuntimeError(
            f"Vector store table data_{POSTGRES_CONFIG['table_name']} is empty. "
            "Run `python -m app.ingest` to populate it first."
        )

    logging.getLogger(__name__).info("Using existing vector store (%d rows).", count)
    return VectorStoreIndex.from_vector_store(vector_store)


def _build_dense(index, cfg: dict):
    """Dense (vector) retriever: top_k override or dense_top_k."""
    return index.as_retriever(similarity_top_k=cfg.get("top_k", cfg["dense_top_k"]))


def _build_sparse(index, cfg: dict):
    """Sparse (full-text) retriever: top_k override or sparse_top_k."""
    return index.as_retriever(
        vector_store_query_mode="sparse",
        similarity_top_k=cfg.get("top_k", cfg["sparse_top_k"]),
    )


def _build_fusion(index, cfg: dict, retriever_class=None):
    """Query fusion over dense + sparse retrievers."""
    dense = _build_dense(index, cfg)
    sparse = _build_sparse(index, cfg)
    retriever_class = retriever_class or QueryFusionRetriever
    return retriever_class(
        [dense, sparse],
        similarity_top_k=cfg["fusion_top_k"],
        num_queries=cfg["num_queries"],
        mode=cfg["fusion_mode"],
        use_async=True,
        query_gen_prompt=query_gen_prompt,
    )


RETRIEVER_BUILDERS = {
    "dense": _build_dense,
    "sparse": _build_sparse,
    "fusion": _build_fusion,
}


def build_retriever(index, cfg: dict | None = None, retriever_class=None):
    """Build the retriever for the given index from a config dict.

    cfg keys (see RETRIEVAL_CONFIG defaults):
        retriever_type: "fusion" (default) or "dense" or "sparse"
        top_k: optional override for the non-fusion retrievers' similarity_top_k
        dense_top_k, sparse_top_k: per-source top-k
        fusion_top_k: final fused top-k
        num_queries: sub-queries generated by the fusion retriever
        fusion_mode: e.g. "reciprocal_rerank"

    retriever_class: optional QueryFusionRetriever subclass (e.g. one that
        records generated sub-queries). Defaults to QueryFusionRetriever.

    To add another retriever type, add a builder function and register it in
    RETRIEVER_BUILDERS under its retriever_type name.
    """
    cfg = {**RETRIEVAL_CONFIG, **(cfg or {})}

    retriever_type = cfg.get("retriever_type", "fusion")
    builder = RETRIEVER_BUILDERS.get(retriever_type)
    if builder is None:
        raise ValueError(
            f"Unknown retriever_type: {retriever_type}. "
            f"Known types: {sorted(RETRIEVER_BUILDERS)}"
        )
    if retriever_type == "fusion":
        return _build_fusion(index, cfg, retriever_class)
    return builder(index, cfg)


def build_reranker(cfg: dict | None = None):
    """Build the node postprocessor (LLM rerank) from a config dict."""
    cfg = {**RETRIEVAL_CONFIG, **(cfg or {})}
    if not cfg["rerank_on"]:
        return None
    return LLMRerank(
        choice_batch_size=cfg.get("rerank_batch_size", 10),
        top_n=cfg["rerank_top_n"],
    )


def initialize_rag():
    setup_logging()
    nest_asyncio.apply()
    setup_phoenix()

    index = build_index()

    retriever = build_retriever(index)
    reranker = build_reranker()

    # -----------------------
    # Query engine
    # -----------------------
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_mode="tree_summarize", ## TODO: Need to split the pipeline to allow different response modes
        node_postprocessors=[reranker] if reranker else None,
        streaming=True,
    )

    return query_engine

def cli_loop():
    engine = initialize_rag()
    
    print("Type your questions below. Enter 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("\n[ASK RAG] > ").strip()
            
            if not user_input:
                continue
            if user_input.lower() in ['exit', 'quit']:
                print("Shutting down engine session. Goodbye!")
                px.close_app()
                break
                
            
            response = engine.query(user_input)
            
            print("="*78)
            print(f"\n[FINAL RESPONSE]:\n{response}\n")
            
        except KeyboardInterrupt:
            print("\nSession aborted cleanly.")
            px.close_app()
            break
        except Exception as e:
            print(f"\n An unexpected runtime error occurred: {e}")
            px.close_app()

if __name__ == "__main__":
    cli_loop()