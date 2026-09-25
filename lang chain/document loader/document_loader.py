from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V3.2',
    temperature=0.5,
    max_new_tokens=128,
    huggingfacehub_api_token=os.environ['HF_TOKEN']
)

model = ChatHuggingFace(llm = llm)

loader = TextLoader( file_path='./document.txt' )

doc = loader.load()

print(doc)