import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Youtube Assistant")

with st.sidebar:
    with st.form(key="yt_form"):
        youtube_url = st.sidebar.text_area("Enter the YouTube URL", max_chars=100)
        query: str = st.sidebar.text_area(
            label="Ask me about the video?",
            max_chars=100,
            key="query",
        )

        submit_buton = st.form_submit_button(label="Submit")

if query and youtube_url:
    vector_db = lch.create_vector_db_from_youtube_url(youtube_url)
    response, docs = lch.get_response_from_query(vector_db, query)
    st.subheader("Response:")
    st.text(
        textwrap.fill(
            response,
            width=100,
        )
    )
