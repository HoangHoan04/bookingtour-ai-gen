import os
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:root@localhost:5432/booking_tour_dev")

def get_sql_agent():
    db = SQLDatabase.from_uri(DB_URL, include_tables=['tours', 'tour_details', 'tour_prices'])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        max_retries=2,
        temperature=0
    )

    prefix_prompt = """
            Bạn là một chuyên gia PostgreSQL. 
            Nhiệm vụ: Trình bày dữ liệu tour du lịch từ Database.

            QUY TẮC ĐỊNH DẠNG (BẮT BUỘC):
            1. Khi bạn đã tìm thấy dữ liệu và muốn trả lời người dùng, bạn PHẢI bắt đầu bằng cụm từ: "Final Answer:".
            2. Nếu không có từ khóa "Final Answer:", hệ thống sẽ báo lỗi.
            3. Tuyệt đối không giải thích thêm sau khi đã có Final Answer.

            QUY TẮC SQL (CỰC KỲ QUAN TRỌNG):
            - Luôn dùng dấu ngoặc kép cho tên cột camelCase: "shortDescription", "startLocation", "priceType".
            - Để lấy thông tin giá và ngày, bạn cần JOIN: tours -> tour_details -> tour_prices.
            - Nếu hỏi số lượng, hãy dùng COUNT(*).
            """

    agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description",
        verbose=True,
        prefix=prefix_prompt,
        handle_parsing_errors="Lỗi định dạng! Hãy nhớ câu trả lời cuối cùng phải bắt đầu bằng 'Final Answer: ' và trình bày nội dung ngay sau đó."
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