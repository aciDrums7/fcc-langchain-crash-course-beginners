from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()


def create_vector_db_from_youtube_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)

    vector_db = FAISS.from_documents(docs, embeddings)
    return vector_db


def get_response_from_query(vector_db, query: str, k=5) -> str:
    # ? so we can send 5 relevant docs at max
    docs = vector_db.similarity_search(query, k)
    docs_page_content = " ".join([doc.page_content for doc in docs])

    llm = OpenAI(temperature=0.5)
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful Youtube assistant that can answers questions about videos based on the video's transcript.

        Answer the following question: {question}
        By searching the following video transcription: {docs}

        Only use the factual information from the transcript to answer the question.

        If you feel like don't have enough information to answer the question, just say "I don't know".

        Your answer should be detailed.
        """,
    )
    chain = LLMChain(llm=llm, prompt=prompt)

    response = chain.invoke({"question": query, "docs": docs_page_content})
    response = (
        response["text"].replace("\n", "")
        if isinstance(response, dict)
        else str(response).replace("\n", "")
    )

    return response, docs
