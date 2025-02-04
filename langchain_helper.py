from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

load_dotenv()
# MODEL = "meta-llama/llama-3.2-3b-instruct:free"

llm = OpenAI(
    temperature=1,
)


def generate_pet_name(animal_type, pet_color):
    prompt_template = PromptTemplate(
        input_variables=["animal_type", "pet_color"],
        template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color. Suggest me five cool names.",
    )

    chain = LLMChain(llm=llm, prompt=prompt_template, output_key="pet_name")
    response = chain.invoke({"animal_type": animal_type, "pet_color": pet_color})

    return response


def langchain_agent():
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    result = agent.invoke("What is the average age of a dog? Multiply the age by 3")

    return result


if __name__ == "__main__":
    # print(generate_pet_name("dog", "blue"))
    print(langchain_agent())
