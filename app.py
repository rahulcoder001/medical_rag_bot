import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from langchain_pinecone import PineconeVectorStore
from huggingface_hub import login
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from transformers import AutoTokenizer, pipeline

from dotenv import load_dotenv
load_dotenv()
# Environment setup

pinecone_key= os.environ["PINECONE_API_KEY"]
token = os.environ["HUGGINGFACEHUB_API_TOKEN"]

os.environ["PINECONE_API_KEY"] = pinecone_key
login(token= token)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = token

# Initialize FastAPI app
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_models()
    yield
    # Shutdown (cleanup if needed)
    pass

app = FastAPI(
    title="Medical RAG Bot API",
    description="A medical question-answering API using RAG (Retrieval Augmented Generation)",
    version="1.0.0",
    lifespan=lifespan
)

# Request/Response models
class QueryRequest(BaseModel):
    query: str
    top_k_candidates: Optional[int] = 12
    include_sources: Optional[bool] = False

class SourceDocument(BaseModel):
    content: str
    metadata: Optional[dict] = {}

class QueryResponse(BaseModel):
    answer: str
    used_docs: Optional[List[SourceDocument]] = []
    prompt_length_tokens: Optional[int] = None
    status: str = "success"

# Global variables for model components
embeddings = None
docsearch = None
retriever = None
tokenizer = None
pipe = None

# Model configuration
MODEL_NAME = "google/flan-t5-large"
MAX_MODEL_INPUT_TOKENS = 512
RESERVED_FOR_Q_AND_PROMPT = 120
MAX_CONTEXT_TOKENS = MAX_MODEL_INPUT_TOKENS - RESERVED_FOR_Q_AND_PROMPT
GEN_MAX_NEW_TOKENS = 300
TEMPERATURE = 0.3

SYSTEM_PROMPT = (
    "You are a medical assistant for question-answering. "
    "Use ONLY the following retrieved context to answer the question. "
    "If the answer is not contained in the context, say 'I don't know'. "
    "Answer concisely (max 3 lines).\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}\n\nMedical Answer:"
)

def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

def initialize_models():
    """Initialize all models and components on startup"""
    global embeddings, docsearch, retriever, tokenizer, pipe
    
    print("Initializing embeddings...")
    embeddings = download_hugging_face_embeddings()
    
    print("Connecting to Pinecone...")
    index_name = "medicalbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    print("Loading language model...")
    pipe = pipeline(
        "text2text-generation",
        model=MODEL_NAME,
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        model_kwargs={"temperature": TEMPERATURE},
    )
    
    print("Models initialized successfully!")

def n_tokens(text: str) -> int:
    """Return token count using the model tokenizer (approx)."""
    return len(tokenizer.encode(text, truncation=False))

def build_context_from_docs(
    docs: List[Document],
    question: str,
    max_context_tokens: int = MAX_CONTEXT_TOKENS
) -> tuple:
    """
    Build concatenated context from retrieved docs ensuring token budget not exceeded.
    Returns (context_text, used_documents).
    Strategy: iterate docs in order and add until token budget full.
    """
    used_docs = []
    context_parts = []
    current_tokens = 0

    for doc in docs:
        part = doc.page_content.strip()
        part_tokens = n_tokens(part)
        if part_tokens > max_context_tokens:
            token_ids = tokenizer.encode(part, truncation=True, max_length=max_context_tokens)
            part = tokenizer.decode(token_ids, skip_special_tokens=True)
            part_tokens = n_tokens(part)

        if current_tokens + part_tokens > max_context_tokens:
            if not context_parts:
                context_parts.append(part)
                used_docs.append(doc)
            break

        context_parts.append(part)
        used_docs.append(doc)
        current_tokens += part_tokens

    context_text = "\n\n---\n\n".join(context_parts)
    return context_text, used_docs

def make_prompt(context: str, question: str) -> str:
    return SYSTEM_PROMPT.format(context=context, question=question)

def safe_rag_answer(query: str, top_k_candidates: int = 10):
    """
    1. Retrieve up to top_k_candidates docs
    2. Build a context that fits token budget
    3. Call local HF pipeline with that prompt
    4. Return generated answer and used docs
    """
    try:
        candidates = retriever.get_relevant_documents(query) 
    except TypeError:
        candidates = retriever.get_relevant_documents(query) 
    candidates = candidates[:top_k_candidates]

    # Build context within token limit
    context_text, used_docs = build_context_from_docs(candidates, query, max_context_tokens=MAX_CONTEXT_TOKENS)
    if not context_text.strip():
        return {"answer": "I don't know (no relevant context found).", "used_docs": []}

    prompt_text = make_prompt(context_text, query)

    prompt_tokens = n_tokens(prompt_text)
    if prompt_tokens > MAX_MODEL_INPUT_TOKENS - 10:
        header = "You are a medical assistant for question-answering. Use ONLY the following retrieved context to answer the question. If the answer is not contained in the context, say 'I don't know'. Answer concisely (max 3 lines).\n\nContext:\n"
        header_ids = tokenizer.encode(header, add_special_tokens=False)
        question_part = f"\n\nQuestion:\n{query}\n\nMedical Answer:"
        question_ids = tokenizer.encode(question_part, add_special_tokens=False)
        budget_for_context = MAX_MODEL_INPUT_TOKENS - (len(header_ids) + len(question_ids) + 20)
        context_ids = tokenizer.encode(context_text, add_special_tokens=False)
        if budget_for_context <= 0:
            truncated_context = ""
        else:
            truncated_context = tokenizer.decode(context_ids[-budget_for_context:], skip_special_tokens=True)
        prompt_text = header + truncated_context + question_part

    # Generate response
    gen = pipe(prompt_text, max_new_tokens=GEN_MAX_NEW_TOKENS, do_sample=False)
    generated_text = gen[0]["generated_text"].strip()

    if generated_text.startswith(prompt_text):
        answer = generated_text[len(prompt_text):].strip()
    else:
        answer = generated_text

    return {"answer": answer, "used_docs": used_docs, "prompt_length_tokens": n_tokens(prompt_text)}

# API Endpoints
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    initialize_models()
    yield
    # Shutdown (cleanup if needed)
    pass

# Update app initialization to use lifespan
app = FastAPI(
    title="Medical RAG Bot API",
    description="A medical question-answering API using RAG (Retrieval Augmented Generation)",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    return {
        "message": "Medical RAG Bot API is running!",
        "endpoints": {
            "POST /query": "Submit a medical query",
            "GET /health": "Check API health"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "retriever_loaded": retriever is not None
    }

@app.post("/query", response_model=QueryResponse)
async def query_medical_bot(request: QueryRequest):
    """
    Process a medical query using RAG
    
    - **query**: The medical question to ask
    - **top_k_candidates**: Number of documents to consider (default: 12)
    - **include_sources**: Whether to include source documents in response (default: False)
    """
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Get answer from RAG system
        result = safe_rag_answer(request.query, top_k_candidates=request.top_k_candidates)
        
        # Prepare response
        response_data = {
            "answer": result["answer"],
            "prompt_length_tokens": result.get("prompt_length_tokens"),
            "status": "success"
        }
        
        # Include sources if requested
        if request.include_sources and "used_docs" in result:
            response_data["used_docs"] = [
                SourceDocument(
                    content=doc.page_content,
                    metadata=getattr(doc, 'metadata', {})
                )
                for doc in result["used_docs"]
            ]
        
        return QueryResponse(**response_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check for individual components
@app.get("/status/detailed")
async def detailed_status():
    return {
        "embeddings_loaded": embeddings is not None,
        "docsearch_connected": docsearch is not None,
        "retriever_ready": retriever is not None,
        "tokenizer_loaded": tokenizer is not None,
        "pipeline_loaded": pipe is not None,
        "cuda_available": torch.cuda.is_available(),
        "model_name": MODEL_NAME
    }

if __name__ == "__main__":
    # Run the API
    uvicorn.run(
        "app:app",  # Updated to match your filename (app.py)
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to False in production
        workers=1  # Single worker to avoid model loading issues
    )
