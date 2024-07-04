import asyncio

from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from qdrant_client import models
from tqdm.asyncio import tqdm
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from sentence_transformers import SentenceTransformer

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval


async def main():
    config = dotenv_values(".env")

    # 初始化 LLM 嵌入模型 和 Reranker
    llm = Ollama(
        model="qwen", base_url=config["OLLAMA_URL"], temperature=0, request_timeout=120
    )
    # embeding = HuggingFaceEmbedding(
    #     model_name="BAAI/bge-small-zh-v1.5",
    #     cache_folder="./",
    #     embed_batch_size=128,
    # )
    # Settings.embed_model = embeding

    # init embedding model and reranker model
    # embed_args = {'model_name': 'aliQwen/gte_Qwen2-7B-instruct', 'cache_folder': './', 'embed_batch_size': 128,}
    # embed_model = HuggingFaceEmbedding(**embed_args)

    # embed_model = HuggingFaceEmbedding(
    #     model_name="BAAI/bge-m3", embed_batch_size=12, max_length=8192
    # )
    embed_model = SentenceTransformer('lier007/xiaobu-embedding-v2')
    Settings.embed_model = embed_model
    print("embed model loaded!!!!")
    reranker_model = FlagEmbeddingReranker(top_n=3, model='BAAI/bge-reranker-v2-m3', use_fp16=True)
    print("reranker model loaded!!!!")
    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(config, reindex=False)

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )

    if collection_info.points_count == 0:
        data = read_data("data")
        pipeline = build_pipeline(llm, embed_model, vector_store=vector_store)
        # 暂时停止实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # 恢复实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(len(data))

    retriever = QdrantRetriever(vector_store, embed_model, similarity_top_k=3)

    queries = read_jsonl("question.jsonl")

    # 生成答案
    print("Start generating answers...")

    results = []
    for query in tqdm(queries, total=len(queries)):
        result = await generation_with_knowledge_retrieval(
            query["query"], retriever, llm, reranker=reranker_model
        )
        results.append(result)

    # 处理结果
    save_answers(queries, results, "submit_result.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
