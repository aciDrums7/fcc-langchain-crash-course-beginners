from tempfile import template
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from os import getenv
from dotenv import load_dotenv

load_dotenv()
MODEL = "meta-llama/llama-3.2-3b-instruct:free"


def generate_pet_name(animal_type, pet_color):
    llm = ChatOpenAI(
        openai_api_key=getenv("OPENROUTER_API_KEY"),
        openai_api_base=getenv("OPENROUTER_BASE_URL"),
        model=MODEL,
        temperature=1,
    )

    prompt_template = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color. Suggest me five cool names.",
    )

    chain = LLMChain(llm=llm, prompt=prompt_template, output_key="pet_name")
    response = chain.invoke({"animal_type": animal_type, "pet_color": pet_color})

    return response


if __name__ == "__main__":
    print(generate_pet_name("dog", "blue"))
