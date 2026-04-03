import os
from fastapi import FastAPI, HTTPException, Header, Depends

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

# 1. Import thư viện Gemini & LangChain
from langchain_google_genai import ChatGoogleGenerativeAI

from service.knowledge_base import init_knowledge, find_relevant_tours
from contextlib import asynccontextmanager

from service.sql_agent import ask_database_agent
from schemas import ChatRequest, ChatResponse
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from fastapi.middleware.cors import CORSMiddleware

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

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_retries=2,
)

SYSTEM_PROMPT = """
Bạn là một chuyên gia tư vấn du lịch (Travel Specialist) có tâm và giàu vốn sống. 
Nhiệm vụ của bạn là biến những thông tin khô khan từ database thành một hành trình đầy cảm hứng.

KHI NHẬN ĐƯỢC DỮ LIỆU TOUR:
1. Xác định các địa danh trong phần 'Điểm nổi bật' (Highlights).
2. Dùng kiến thức sâu rộng của bạn để mô tả SỐNG ĐỘNG về các địa danh đó (vẻ đẹp, không khí, cảm xúc khi ở đó). Với độ dài vừa phải để khách hàng có thể hình dung ra nhưng không quá dài dòng.
   - Ví dụ: Thấy 'Cầu Vàng' -> tả về đôi bàn tay khổng lồ giữa mây trời. 
   - Thấy 'Hội An' -> tả về đèn lồng và nét rêu phong.
3. Tuyệt đối KHÔNG bịa đặt về: Giá tour, Ngày khởi hành, hoặc các dịch vụ bao gồm nếu không có trong dữ liệu. Chỉ được "múa bút" ở phần mô tả cảnh sắc và trải nghiệm.

PHONG CÁCH NGÔN NGỮ:
- Truyền cảm hứng, chuyên nghiệp, sử dụng các từ ngữ gợi hình gợi cảm.
- Trình bày bằng Markdown: dùng Bold cho địa danh, Bullet points cho các trải nghiệm.
- Kết thúc bằng một câu kêu gọi (Call-to-action) thân thiện.

NGỮ CẢNH HỆ THỐNG:
{context}

CÂU HỎI CỦA KHÁCH:
{question}
"""

async def verify_key(x_api_key: str = Header(...)):
    EXPECTED_API_KEY = os.getenv("API_SECRET_KEY", "SECRET_KEY_BETWEEN_NODE_AND_PYTHON")
    if x_api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Unauthorized Access")

rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Dựa trên lịch sử trò chuyện, hãy chuyển câu hỏi mới của người dùng thành một câu truy vấn độc lập, đầy đủ ý nghĩa để tìm kiếm trong database du lịch."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chat_history = []

def chat_with_memory(user_question: str):
    global chat_history
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        max_retries=2,
    )

    if len(chat_history) > 0:
        rephrase_chain = rephrase_prompt | llm
        standalone_query = rephrase_chain.invoke({
            "chat_history": chat_history,
            "question": user_question
        }).content
    else:
        standalone_query = user_question

    docs = find_relevant_tours(standalone_query, k=3)
    context = "\n---\n".join(docs)

    qa_chain = qa_prompt | llm
    response = qa_chain.invoke({
        "context": context,
        "chat_history": chat_history,
        "question": user_question
    })

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=response.content))
    if len(chat_history) > 10: # Giữ 5 cặp hội thoại
        chat_history = chat_history[-10:]

    return response.content


@app.post("/api/v1/chat-with-data", response_model=ChatResponse)
async def chat_with_data(
    payload: ChatRequest,
    authorized: bool = Depends(verify_key)
):

    answer = await ask_database_agent(payload.question)
    return ChatResponse(answer=answer)

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    authorized: bool = Depends(verify_key)
):
    answer = chat_with_memory(payload.question)
    return ChatResponse(answer=answer)

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "bookingtour-ai-gen"}

# Chạy server: uvicorn main:app --reload --port 8000