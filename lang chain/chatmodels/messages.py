from langchain_core.messages import SystemMessage , AIMessage , HumanMessage
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os 

load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

messages = [
    SystemMessage(content="you are a helpful assistent "),
    HumanMessage(content='Tell me about langchain')
]

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3.2",
    messages=messages 
)

messages.append(AIMessage(content=completion.choices[0].message.content))

print(messages)