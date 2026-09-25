# from langchain_huggingface import HuggingFaceEmbeddings

# embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# doc = [
#      "That is a happy dog",
#      "That is a very happy person",
#      "Today is a sunny day"
# ]
# text = "That is a very happy person"

# vector = embedding.embed_query(text)
# # vector = embedding.embed_documents(doc)

# print(str(vector))

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os 

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
     model='google/embeddinggemma-300m',
     huggingfacehub_api_token= os.environ['HF_TOKEN'],
)

vector = embedding.embed_query("That is a happy dog")

print(len(vector))