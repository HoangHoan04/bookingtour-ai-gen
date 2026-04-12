import os
from fastapi import FastAPI, Depends

# Load environment variables (only in non-production)
if os.getenv("NODE_ENV") != "production":
    try:
        from dotenv import load_dotenv
        import pathlib
        
        # Load appropriate .env file based on environment
        env = os.getenv("NODE_ENV", "development")
        env_file = ".env.dev" if env == "development" else ".env"
        env_path = pathlib.Path(__file__).parent / env_file
        
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            print(f"✅ Loaded environment from: {env_file}")
        else:
            print(f"⚠️  {env_file} not found, using system environment variables")
    except ImportError:
        print("⚠️  dotenv not available (production environment)")

from service.knowledge_base import init_knowledge
from contextlib import asynccontextmanager

from service.sql_agent import ask_database_agent
from schemas import ChatRequest, ChatResponse
from fastapi.middleware.cors import CORSMiddleware
from service.chatbot import chat_with_memory, verify_key

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_knowledge()
    yield


app = FastAPI(title="Travel AI Microservice with Gemini", lifespan=lifespan)

# Đọc CORS origins từ environment variable
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3350").split(",")
origins = [origin.strip() for origin in ALLOWED_ORIGINS]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)
@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    authorized: bool = Depends(verify_key)
):
    answer = await chat_with_memory(payload.question)
    return ChatResponse(answer=answer)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "bookingtour-ai-gen"}

# Chạy server: uvicorn main:app --reload --port 8000