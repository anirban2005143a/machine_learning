# from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
# from dotenv import load_dotenv
# import os
# from typing import TypedDict

# load_dotenv()

# llm = HuggingFaceEndpoint(
#     repo_id='Qwen/Qwen3-4B-Instruct-2507',
#     max_new_tokens=128,
#     temperature=0.5,
#     huggingfacehub_api_token = os.environ["HF_TOKEN"],
# )

# model = ChatHuggingFace(llm = llm)

# class Review(TypedDict):
#     summary : str
#     sentiment : str

# # structured_model = model.with_structured_output(Review)
# structured_model = llm.with_structured_output(Review)

# result = structured_model.invoke("""The hardware is great, but the software feels bloated.
# There are too many pre-installed apps that I can't remove. Also, the UI looks outdated
# compared to other brands. Hoping for a software update to fix this.""")

# print(result)

from dotenv import load_dotenv
import os
from typing import TypedDict
from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

class Review(TypedDict):
    summary: str
    sentiment: str


format_instructions = {
  "summary": "Short summary of the review",
  "sentiment": "positive, neutral, or negative"
}

# 2. LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",
    huggingfacehub_api_token=os.environ["HF_TOKEN"],
    max_new_tokens=128,
    temperature=0.5,
)

model = ChatHuggingFace(llm = llm)

# 3. Prompt + parser
prompt = ChatPromptTemplate([
    ("system", "Extract structured data."),
    ("human", """Review text:
    {review}
    Follow this format:
    {format_instructions}
    """)
])

chain = prompt | model

# 4. Invoke
result: Review = chain.invoke({
    "review": """The hardware is great, but the software feels bloated.
There are too many pre-installed apps. UI looks outdated.""",
    "format_instructions": format_instructions
})

print(result)
