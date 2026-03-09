# check_models.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

genai.configure(api_key="AIzaSyA4ps36fsCTR2KuLvIub7i_v8BrAt8X1vI")

print("Đang lấy danh sách models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Lỗi: {e}")