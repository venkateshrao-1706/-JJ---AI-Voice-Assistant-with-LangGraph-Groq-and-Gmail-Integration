import os
import json
import hashlib
import redis
import time
import functools
import shutil
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from llama_cloud_services import LlamaParse
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from langchain_classic.storage import LocalFileStore, create_kv_docstore

load_dotenv()

# make sure these keys exist before doing any heavy work
for key_name in ["GROQ_API_KEY", "LLAMA_CLOUD_API_KEY"]:
    if not os.getenv(key_name):
        raise ValueError(f"{key_name} is missing. Add it to your .env file")

# path is read from .env so it works on any machine
pdf_path = os.getenv("PDF_PATH", "data/Employee-Handbook.pdf")
parsed_cache = "parsed_handbook.json"

# only wipe the stores when you really want a fresh start (set FORCE_RESET=true in .env)
FORCE_RESET = os.getenv("FORCE_RESET", "false").lower() == "true"

llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0)

hf_cross_encoder = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-large"
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)


def simple_rate_limiter(max_calls: int, period_counts: int):
    call_history = []

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            call_history[:] = [t for t in call_history if t > now - period_counts]
            if len(call_history) >= max_calls:
                raise Exception("Rate limit is exceeded, please wait a bit and try again")
            call_history.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator


if FORCE_RESET:
    print("Cleaning old stores and cache for a fresh sync...")
    for path in ["./qdrant_data", "./parent_store", "parsed_handbook.json"]:
        if os.path.exists(path):
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    print("Cleanup complete!")


def load_handbook_docs(pdf_path: str):
    if os.path.exists(parsed_cache):
        try:
            with open(parsed_cache, 'r', encoding="utf-8") as f:
                data = json.load(f)
                if data:  # make sure the cache isn't empty
                    print("Loaded parsed documents from cache")
                    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
        except json.JSONDecodeError:
            print("Cache file corrupted. Re-parsing...")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    parser = LlamaParse(
        result_type="markdown",
        verbose=True,
        system_prompt=("This is an enterprise employee handbook. "
            "Accurately extract section numbers, headers, bullet points, and tables.")
    )
    raw = parser.load_data(pdf_path)

    docs = [Document(page_content=d.text, metadata=d.metadata) for d in raw]

    if not docs:
        raise ValueError("LlamaParse returned 0 documents. Please check your PDF file and API key.")

    with open(parsed_cache, "w", encoding="utf-8") as f:
        json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in docs], f, ensure_ascii=False, default=str)

    print("Parsed with LlamaParse and saved to cache")
    return docs


docs = load_handbook_docs(pdf_path)

client = QdrantClient(path="./qdrant_data")
fs = LocalFileStore("./parent_store")
doc_store = create_kv_docstore(fs)
collection_name = 'enterprise_qdrant'

if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=1024,
            distance=models.Distance.COSINE
        )
    )

vector_store = QdrantVectorStore(
    client=client,
    embedding=embeddings,
    collection_name=collection_name
)

Parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

retriever = ParentDocumentRetriever(
    vectorstore=vector_store,
    parent_splitter=Parent_splitter,
    child_splitter=child_splitter,
    docstore=doc_store
)

if client.count(collection_name=collection_name).count == 0:
    retriever.add_documents(docs)
    print("All pages ingested successfully!")
else:
    print("Using existing vector store, skipping ingestion")

parent_chunks = Parent_splitter.split_documents(docs)

if not parent_chunks:
    raise ValueError("Parent splitter resulted in 0 chunks. Check your chunk size settings.")

BM25 = BM25Retriever.from_documents(parent_chunks)
BM25.k = 5

Hybrid = EnsembleRetriever(retrievers=[BM25, retriever], weights=[0.3, 0.7])
compressor = CrossEncoderReranker(model=hf_cross_encoder, top_n=5)

compressor_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=Hybrid
)

redis_client = redis.Redis(
    host="127.0.0.1",
    port=6379,
    db=0,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=2
)


def execute_rag_pipeline(query: str):
    retrieved_parent = None

    # clean the query first so "Leave policy?" and "leave  policy?" share one cache entry
    clean_query = " ".join(query.lower().split())
    query_hash = hashlib.md5(clean_query.encode("utf-8")).hexdigest()
    cache_key = f"rag_top3_{query_hash}"
    cached_result = None

    try:
        redis_client.ping()
        REDIS_AVAILABLE = True
        cached_result = redis_client.get(cache_key)
        print("Redis is working fine and connected successfully")
    except redis.exceptions.RedisError as e:
        REDIS_AVAILABLE = False
        print(f"Running without cache (without redis): {e}")

    if cached_result is not None:
        retrieved_parent = [
            Document(page_content=d["page_content"], metadata=d["metadata"])
            for d in json.loads(cached_result)
        ]
        print("Loaded from Redis cache")
        return retrieved_parent

    # cache miss (or Redis is down), so run the full retrieval
    retrieved_parent = compressor_retriever.invoke(query)
    print("Retrieved fresh results (not from cache)")

    # save to Redis only when we actually did a fresh retrieval
    if REDIS_AVAILABLE:
        data = [
            {"page_content": d.page_content, "metadata": d.metadata}
            for d in retrieved_parent
        ]
        try:
            redis_client.set(cache_key, json.dumps(data), ex=3600)
        except redis.exceptions.RedisError:
            print("Unable to save cache via Redis")

    return retrieved_parent


# the rate limit belongs here, because the LLM call is the one that costs money / hits Groq limits
@simple_rate_limiter(max_calls=int(os.getenv("LLM_CALLS_PER_MINUTE", "20")), period_counts=60)
def call_llm(prompt: str):
    return llm.invoke(prompt).content


def answer_question(query: str):
    docs = execute_rag_pipeline(query)
    if not docs:
        return "I couldn't find this in the handbook."
    context = "\n\n".join(d.page_content for d in docs)
    prompt = f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I couldn't find this in the handbook."
Context:
{context}
Question: {query}
Answer:"""
    return call_llm(prompt)


if __name__ == "__main__":
    query = "If you suspect that an offender doesn’t realize they are guilty of harassment, you could talk to them directly and what should come next??"
    print(answer_question(query))




# from langchain_groq import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_qdrant import QdrantVectorStore
# # from langchain_core.output_parsers import StrOutputParser
# # from langchain_core.prompts import PromptTemplate
# # from langchain_community.embeddings import SentenceTransformerEmbeddings
# # from langchain_core.runnables import (RunnableSequence,RunnableParallel,RunnableBranch,RunnablePassthrough,RunnableLambda)
# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from llama_cloud_services import LlamaParse
# from langchain_classic.retrievers import ParentDocumentRetriever
# from langchain_core.stores import InMemoryStore
# from langchain_classic.retrievers import EnsembleRetriever
# from langchain_community.retrievers import BM25Retriever
# from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
# from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
# from langchain_community.cross_encoders import HuggingFaceCrossEncoder
# from langchain_core.documents import Document
# from langchain_classic.storage import LocalFileStore, create_kv_docstore
# import json
# import hashlib
# import redis
# import time
# import functools
# from langchain_classic.storage import LocalFileStore, create_kv_docstore
# from dotenv import load_dotenv
# load_dotenv()
# import os

# pdf_path = "C:/Users/venka/OneDrive/Documents/RERSUME FOR VENKATESH/Employee-Handbook (6).pdf"

# os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv["LLAMA_CLOUD_API_KEY"]

# llm = ChatGroq(model="openai/gpt-oss-20b",temperature=0.25)

# hf_cross_encoder = HuggingFaceCrossEncoder(
#     model_name="BAAI/bge-reranker-large"
# )

# def simple_rate_limiter(max_calls:int,period_counts:int):
#     call_history = []
#     def decorator(func):
#        @functools.wraps(func)
#        def wrapper(*args , **kwargs):
#         now = time.time()
#         call_history[:]=[t for t in call_history if t> now - period_counts]
         
#         if len(call_history)>=max_calls:
#             raise Exception(f"Rate limit is exceeded")
        
#         call_history.append(now)
#         return func(*args,**kwargs)
#        return wrapper
#     return decorator

# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-large-en-v1.5"
# )

# parsed_cache = "parsed_handbook.json"
# pdf_path = "C:/Users/venka/OneDrive/Documents/RERSUME FOR VENKATESH/Employee-Handbook (6).pdf"


# def load_handbook_docs(pdf_path:str):
#     if os.path.exists(parsed_cache):
#         with open(parsed_cache,'r',encoding="utf-8") as f:
#             data = json.load(f)
#             print("loaded Parsed documents availables")
#             return [Document(page_content = d["page_content"],metadata = d["metadata"])
#             for d in data]
    
#     if not  os.path.exists(pdf_path):
#         raise FileNotFoundError(f"File not found:{pdf_path}")

#     parser = LlamaParse(
#         result_type= "markdown",
#         verbose=True,
#         parsing_instruction=("This is an enterprise employee handbook. "
#             "Accurately extract section numbers, headers, bullet points, and tables."))

#     raw = parser.load_data(pdf_path)
#     docs = [Document(page_content=d.text,metadata = d.metadata)
#         for d in raw
#         ]
#         # Saving data for next time 
#     with open(parsed_cache,"w",encoding="utf-8") as f :
#             json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in docs],f, ensure_ascii=False, default=str
#             )
            
#     print(f"parsed with Llmaparse and saved to cache")
#     return docs


# docs = load_handbook_docs(pdf_path)


# client = QdrantClient(path="./qdrant_data")
# fs = LocalFileStore("./parent_store")
# doc_store = create_kv_docstore(fs)

# collection_name = 'enterprise_qdrant'


# from qdrant_client.http import models

# # Creating collection first (dimension 1024 is for BAAI/bge-large-en-v1.5)

# if not client.collection_exists(collection_name):
#     client.create_collection(
#         collection_name=collection_name,
#         vectors_config=models.VectorParams(
#             size=1024,
#             distance=models.Distance.COSINE
#         )
#     )

# vector_store = QdrantVectorStore(
#     client = client,
#     embedding=embeddings,
#     collection_name= collection_name
#      )

# Parent_splitter= RecursiveCharacterTextSplitter(
#     chunk_size = 2000,
#     chunk_overlap = 200)

# child_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 400,
#     chunk_overlap = 50
# )

# retriever = ParentDocumentRetriever(
#     vectorstore=vector_store,
#     parent_splitter=Parent_splitter,
#     child_splitter=child_splitter,
#     docstore=doc_store
# )
# query = " If you suspect that an offender doesn’t realize they are guilty of harassment, you could talk to them directly and what should come next??"

# if client.count(collection_name).count == 0:
#     retriever.add_documents(docs)
#     print("All pages ingested successfully!")
# else:
#     print("Using existing vector store, skipping ingestion")

# BM25 = BM25Retriever.from_documents(Parent_splitter.split_documents(docs))
# BM25.k = 2

# Hybrid = EnsembleRetriever(retrievers=[BM25,retriever],weights=[0.6,0.4])


# redis_client = redis.Redis(
#     host="127.0.0.1",
#     port=6379,
#     db=0,
#     decode_responses=True,
#     socket_connect_timeout=2,
#     socket_timeout=2
# )

# @simple_rate_limiter(max_calls=3, period_counts=60)
# def execute_rag_pipeline(query: str):
#      retrieved_parent = None

#      query_hash = hashlib.md5(query.encode("utf-8")).hexdigest()

#      cache_key = f"rag_retrived_docs{query_hash}"

#      cached_result = None

#      try:
#          redis_client.ping()
#          REDIS_AVAILABLE =True
#          cached_result = redis_client.get(cache_key)
#          print("redis is working fine and  connected sucessfully")
#      except redis.exceptions.RedisError as e:
#         REDIS_AVAILABLE = False
#         print(f"Running Without cache(without redis){e}")

#      if cached_result is not None:
#        retrieved_parent = [
#         Document(page_content=d["page_content"],metadata = d["metadata"])
#         for d in json.loads(cached_result)]
#        print("loaded from redis cache")
#      else:
#       retrieved_parent = compressor_retriever.invoke(query)
#       print("Loaded without redis")

#        # Saving the documents
#      if REDIS_AVAILABLE == True:
#        data = [
#         {"page_content":d.page_content,"metadata":d.metadata}
#        for d in retrieved_parent]

#        try:
#         redis_client.ping()
#         REDIS_AVAILABLE = True
#         cached_result = redis_client.get(cache_key)
#         print("redis is working fine and connected successfully")
#        except redis.exceptions.RedisError :
#         REDIS_AVAILABLE = False
#         print(f"Running without cache (without redis) ")

#        try:
#           redis_client.set(cache_key,json.dumps(data),ex=3600)
#        except redis.exceptions.RedisError:
#             print("Unable to save cache via redis")

#      return retrieved_parent



# def answer_question(query: str):
#     docs = execute_rag_pipeline(query)
#     if not docs:
#         return "I couldn't find this in the handbook."

#     context = "\n\n".join(d.page_content for d in docs)
#     prompt = f"""Answer the question using ONLY the context below.
# If the answer isn't in the context, say "I couldn't find this in the handbook."

# Context:
# {context}

# Question: {query}
# Answer:"""

#     return llm.invoke(prompt).content

# print(answer_question(query))




# from slowapi import Limiter, _rate_limit_exceeded_handler
# from slowapi.util import get_remote_address
# from slowapi.errors import RateLimitExceeded
# from fastapi import FastAPI, Request
# from pydantic import BaseModel

# # 1. Initialize Limiter (tracks requests by client IP address)
# limiter = Limiter(key_func=get_remote_address)
# app = FastAPI()

# # Register the error handler for when limits are exceeded
# app.state.limiter = limiter
# app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# class QueryRequest(BaseModel):
#     question: str

# # 2. Apply the rate limit decorator to your RAG route
# @app.post("/query-rag")
# @limiter.limit("5/minute")  # Limit to 5 requests per minute per IP
# def query_rag_endpoint(request: Request, body: QueryRequest):
#     # Your existing retriever & LLM chain logic here
#     # query = body.question
#     # retriever_parent = retriever.invoke(query)
    
#     return {
#         "status": "success",
#         "query": body.question,
#         "answer": "Generated RAG response..."
#     }

# chunking = text_splitter.split_documents(docs)


# prompt = PromptTemplate.from_template(
#     """You are helpful ai assistant ,
#     "Answer only based upon the infomation provided to you"
#     "if you don't know the answer just say i don't know don't hallucinate"
#     context:{context}
#     What is {dash}?"""
# )

# # for document in loader.lazy_load():
# #     print(document)

# embeddings = HuggingFaceEmbeddings(
#     model_name="BAAI/bge-large-en-v1.5"
# )

# vector_store = QdrantVectorStore.from_documents(chunking,embedding = embeddings,path = "./qdrant3.db",collection_name = "90")

# retriever = vector_store.as_retriever(search_kwargs = {"k":2})


# def formated_docs (docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     RunnableParallel({
#         # 1. Fetch and format the context using the 'dash' query
#         "context": lambda x: formated_docs(retriever.invoke(x["dash"])),
#         # 2. Pass through the 'dash' variable so the prompt can use it
#         "dash": lambda x: x["dash"]
#     })
#     | prompt
#     | llm
#     | parser
# )
# result = rag_chain.invoke({"dash":"key-value"})

# print(result)

# retriever = vector_store.asimilarity_search

