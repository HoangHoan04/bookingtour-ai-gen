import os
from sqlalchemy import create_engine, text
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:root@localhost:5432/booking_tour_dev")

vector_store = None


def load_tours_from_db():
    try:
        engine = create_engine(DB_URL)
        connection = engine.connect()

        sql = text("""
                    SELECT 
                        t.id, 
                        t.title, 
                        t.slug, 
                        t.location, 
                        t.durations, 
                        t."shortDescription", 
                        t.highlights, 
                        t.rating,
                        t.status,
                        jsonb_agg(DISTINCT jsonb_build_object(
                            'startLocation', td."startLocation",
                            'startDay', td."startDay",
                            'endDay', td."endDay",
                            'prices', (
                                SELECT jsonb_agg(jsonb_build_object('type', tp."priceType", 'value', tp.price))
                                FROM tour_prices tp 
                                WHERE tp."tourDetailId" = td.id
                            )
                        )) as details
                    FROM tours t
                    JOIN tour_details td ON td."tourId" = t.id
                    WHERE t.status = 'active' OR t.status = 'ACTIVE'
                    GROUP BY t.id;
                """)
        result = connection.execute(sql)

        documents = []
        for row in result:
            detail_strings = []
            for detail in row.details:
                price_info = ", ".join([f"{p['type']}: {p['value']:,.0f}đ" for p in detail['prices']])

                detail_str = (
                    f"- Khởi hành từ {detail['startLocation']} "
                    f"({detail['startDay']} đến {detail['endDay']}): {price_info}"
                )
                detail_strings.append(detail_str)

            all_details_text = "\n".join(detail_strings)
            content = (
                f"Tên Tour: {row.title}. "
                f"Địa điểm: {row.location}. "
                f"Thời gian: {row.durations}. "
                f"Điểm nổi bật: {row.highlights or 'Không có'}. "
                f"Mô tả: {row.shortDescription}. "
                f"Đánh giá khách hàng: {row.rating}/5 sao."
                f"Thông tin chi tiết các đợt khởi hành:\n{all_details_text}"
            )

            doc = Document(
                page_content=content,
                metadata={
                    "tour_id": row.id,
                    "title": row.title,
                    "slug": row.slug,
                    "rating": row.rating
                }
            )
            documents.append(doc)

        connection.close()
        return documents
    except Exception as e:
        print(f"❌ Lỗi kết nối PostgreSQL: {e}")
        return []

def create_vector_db():
    print("⏳ Đang tải dữ liệu từ PostgreSQL...")
    docs = load_tours_from_db()

    if not docs:
        print("⚠️ Không tìm thấy tour nào hoặc lỗi kết nối DB!")
        return None

    print(f"✅ Đã tải {len(docs)} tours. Đang tạo vector...")

    embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")

    return FAISS.from_documents(docs, embeddings)


def init_knowledge():
    global vector_store
    vector_store = create_vector_db()


def find_relevant_tours(query: str, k=3):
    if not vector_store:
        return []
    docs = vector_store.similarity_search(query, k=k)
    return [d.page_content for d in docs]