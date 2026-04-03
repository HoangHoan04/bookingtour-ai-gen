import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:root@localhost:5432/booking_tour_dev")

def get_sql_agent():
    db = SQLDatabase.from_uri(DB_URL, include_tables=['tours'])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        max_retries=2,
        temperature=0
    )

    prefix_prompt = """
        Bạn là một chuyên gia phân tích dữ liệu SQL.
        Nhiệm vụ: Trả lời câu hỏi người dùng bằng cách truy vấn Database PostgreSQL.

        QUY TẮC QUAN TRỌNG VỀ TÊN CỘT (CỰC KỲ QUAN TRỌNG):
        1. Database này sử dụng PostgreSQL.
        2. Các tên cột được viết kiểu camelCase (ví dụ: shortDescription, viewCount).
        3. Khi viết SQL, BẮT BUỘC phải để tên cột trong dấu ngoặc kép đôi ("").
           - SAI: SELECT shortDescription FROM tours
           - ĐÚNG: SELECT "shortDescription" FROM tours

        4. Chỉ trả về thông tin người dùng hỏi, không trả lời thừa.
        5. Nếu không tìm thấy thông tin, hãy nói "Tôi không tìm thấy dữ liệu phù hợp".
        """

    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description",
        verbose=True,
        prefix=prefix_prompt
    )

    return agent_executor

async def ask_database_agent(question: str):
    agent = get_sql_agent()
    try:
        inputs = {"input": question}
        response = await agent.ainvoke(inputs)
        return response["output"]
    except Exception as e:
        print(f"SQL Agent Error: {e}")
        return "Xin lỗi, tôi gặp khó khăn khi truy vấn dữ liệu này."