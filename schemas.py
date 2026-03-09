from pydantic import BaseModel, Field
from typing import List, Optional

class RecommendRequest(BaseModel):
    user_vector: List[float]

class RecommendResponse(BaseModel):
    tour_id: int
    score: float

class TripRequest(BaseModel):
    destination: str = Field(..., description="Địa điểm du lịch")
    duration_days: int = Field(..., description="Số ngày đi")
    budget_level: str = Field(..., description="Mức ngân sách: Tiết kiệm, Trung bình, Sang trọng")
    interests: List[str] = Field(default=[], description="Sở thích: Ẩm thực, Check-in, Lịch sử...")




class ChatRequest(BaseModel):
    question: str = Field(..., description="Tin nhắn của người dùng")

class ChatResponse(BaseModel):
    answer: str
