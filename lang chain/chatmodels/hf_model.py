# import os
# from dotenv import load_dotenv
# from huggingface_hub import InferenceClient

# load_dotenv()

# client = InferenceClient(
#     api_key=os.environ["HF_TOKEN"],
# )

# completion = client.chat.completions.create(
#     model="deepseek-ai/DeepSeek-V3.2",
#     messages=[
#         {
#             "role": "user",
#             "content": "write a short story about Malificent , a witch !"
#         }
#     ],
#     max_tokens=200
# )

# print(completion.choices[0].message.content)

from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='Qwen/Qwen3-4B-Instruct-2507',
    max_new_tokens=128,
    temperature=0.5,
    huggingfacehub_api_token = os.environ["HF_TOKEN"],
    # task="conversational"
)

model = ChatHuggingFace(
    llm=llm
)

result = model.invoke("what is the capital of india ?")

print(result)