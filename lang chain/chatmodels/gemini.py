from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-1.5-chat",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

result = model.invoke("what is the capital of India ??")

print(result)