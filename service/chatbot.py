
import os
from fastapi import Header, HTTPException
from langchain_google_genai import ChatGoogleGenerativeAI
from service.knowledge_base import find_relevant_tours
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from service.sql_agent import ask_database_agent


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_retries=2,
)

ROUTER_PROMPT = """Bạn là một bộ điều phối câu hỏi thông minh cho hệ thống du lịch. 
Dựa trên câu hỏi của người dùng, hãy trả về CHÍNH XÁC một trong hai từ sau:

- 'SQL': Nếu câu hỏi liên quan đến số lượng, thống kê, đếm (bao nhiêu tour, tổng số...), so sánh giá (rẻ nhất, đắt nhất), hoặc liệt kê danh sách thô.
- 'RAG': Nếu câu hỏi liên quan đến tư vấn sở thích, mô tả cảnh đẹp, hỏi chi tiết về trải nghiệm, không gian hoặc cần lời khuyên du lịch.

Chỉ trả về từ 'SQL' hoặc 'RAG', không giải thích thêm."""

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


llm_router = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_creative = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

router_prompt = ChatPromptTemplate.from_messages([
    ("system", ROUTER_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

rephrase_prompt = ChatPromptTemplate.from_messages([
    ("system", "Chuyển câu hỏi mới thành câu truy vấn độc lập để tìm kiếm."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

chat_history = []

async def chat_with_memory(user_question: str):
    global chat_history

    router_chain = router_prompt | llm_router
    intent_response = router_chain.invoke({
        "chat_history": chat_history,
        "question": user_question
    })
    intent = intent_response.content.strip().upper()
    print(f"DEBUG: Intent detected -> {intent}")
    if "SQL" in intent:
        # Gọi SQL Agent để đếm/thống kê chính xác trong Database
        response_content = await ask_database_agent(user_question)
    else:
        # Chạy luồng RAG (Vector Search) để tư vấn cảm hứng
        if len(chat_history) > 0:
            rephrase_chain = rephrase_prompt | llm_router
            standalone_query = rephrase_chain.invoke({
                "chat_history": chat_history,
                "question": user_question
            }).content
        else:
            standalone_query = user_question

        # Lấy Top-K ngữ cảnh (chỉ dùng cho tư vấn)
        docs = find_relevant_tours(standalone_query, k=3)
        context = "\n---\n".join(docs)

        qa_chain = qa_prompt | llm_creative
        response = qa_chain.invoke({
            "context": context,
            "chat_history": chat_history,
            "question": user_question
        })
        response_content = response.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=response_content))
    if len(chat_history) > 10: # Giữ 5 cặp hội thoại
        chat_history = chat_history[-10:]

    return response_content
