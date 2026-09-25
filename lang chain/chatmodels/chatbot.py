import numpy as np
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os 

load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)


chat_history = []

while True :
    user_input = input("you : ")
    if user_input == 'exit':
        break

    messages = []
    # Add previous AI responses if you want context
    for i in range(0, len(chat_history), 2):
        messages.append({"role": "user", "content": chat_history[i]})
        if i+1 < len(chat_history):
            messages.append({"role": "assistant", "content": chat_history[i+1]})

    messages.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3.2",
        messages=messages 
    )

    chat_history.append(user_input)
    chat_history.append(completion.choices[0].message.content)
    print(f"AI : {completion.choices[0].message.content}")


print(chat_history)