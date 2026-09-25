import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ["HF_TOKEN"],
)

result = client.sentence_similarity(
    sentence= "That is a happy person",
    other_sentences=[
            "That is a happy dog",
            "That is a very happy person",
            "Today is a sunny day"
        ],
    model="sentence-transformers/all-MiniLM-L6-v2",
)

print(result)