import phoenix as px
from phoenix.otel import register
import nest_asyncio

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    set_global_handler,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank

from app.config import DOCS_DIR, POSTGRES_CONFIG, LLM_CONFIG
from app.prompts import query_gen_prompt
import logging
import sys



def setup_logging():
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        force=True
    )


def initialize_rag():
    setup_logging()
    nest_asyncio.apply()

    register(endpoint="http://localhost:6006/v1/traces")
    set_global_handler("arize_phoenix")

    # -----------------------
    # Embeddings
    # -----------------------
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        device="cuda",
        max_length=8192,
    )

    # -----------------------
    # LLM
    # -----------------------
    Settings.llm = OpenAILike(
        model=LLM_CONFIG["model"],
        api_base=LLM_CONFIG["api_base"],
        api_key=LLM_CONFIG["api_key"],
        is_chat_model=True,
    )

    # -----------------------
    # Vector DB
    # -----------------------
    vector_store = PGVectorStore.from_params(
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

    # -----------------------
    # Load docs + index
    # -----------------------
    import psycopg2

    conn = psycopg2.connect(
        host=POSTGRES_CONFIG["host"],
        port=POSTGRES_CONFIG["port"],
        user=POSTGRES_CONFIG["user"],
        password=POSTGRES_CONFIG["password"],
        database=POSTGRES_CONFIG["database"],
    )
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"TRUNCATE data_{POSTGRES_CONFIG['table_name']}")
        logging.getLogger(__name__).info("Purged vector store table.")
    conn.close()

    from app.utils import DoclingIngester

    nodes = DoclingIngester(docs_dir=DOCS_DIR).ingest()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        show_progress=True,
    )

    # -----------------------
    # Retrievers
    # -----------------------
    dense = index.as_retriever(similarity_top_k=10)
    sparse = index.as_retriever(
        vector_store_query_mode="sparse",
        similarity_top_k=10,
    )

    retriever = QueryFusionRetriever(
        [dense, sparse],
        similarity_top_k=10,
        num_queries=4,
        mode="reciprocal_rerank",
        use_async=True,
        query_gen_prompt=query_gen_prompt,
    )

    reranker = LLMRerank(choice_batch_size=10, top_n=5)

    # -----------------------
    # Query engine
    # -----------------------
    query_engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        response_mode="tree_summarize", ## TODO: Need to split the pipeline to allow different response modes
        node_postprocessors=[reranker],
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