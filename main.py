from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv

load_dotenv()
MODEL = "meta-llama/llama-3.2-3b-instruct:free"


def generate_pet_name():
    llm = ChatOpenAI(
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base=getenv("OPENROUTER_BASE_URL"),
        model=MODEL,
        temperature=1,
    )

    response = llm.invoke(
        "I have a dog pet and I want a cool name for it. Suggest me five cool names."
    )

    return response.content


if __name__ == "__main__":
    print(generate_pet_name())
