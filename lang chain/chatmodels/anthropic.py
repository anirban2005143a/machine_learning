from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

claude = ChatAnthropic(
    model = ""
)

result = claude.invoke("what is the capital of india ?")

print(result)