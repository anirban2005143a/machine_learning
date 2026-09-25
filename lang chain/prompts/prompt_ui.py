import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient
import streamlit as st
from langchain_core.prompts import PromptTemplate , load_prompt

load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)


result = "some random text"

st.header("Research Tool")

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template = load_prompt('template.json')

prompt = template.format(
    paper_input=paper_input,
    style_input=style_input,
    length_input=length_input
)

if st.button("summarize"):
    completion = client.chat.completions.create(
        # model="EssentialAI/rnj-1-instruct",
        model="deepseek-ai/DeepSeek-V3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],  
    )

    st.write(completion.choices[0].message.content)