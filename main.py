import langchain_helper as lch
import streamlit as st

st.title("Pet Name Generator")

animal_type = st.sidebar.selectbox("What is your pet?", ["Dog", "Cat", "Bird", "Fish"])

if animal_type == "Cat":
    pet_color = st.sidebar.text_area(label="What color is your cat?", max_chars=15)

if animal_type == "Dog":
    pet_color = st.sidebar.text_area(label="What color is your dog?", max_chars=15)

if animal_type == "Bird":
    pet_color = st.sidebar.text_area(label="What color is your bird?", max_chars=15)

if animal_type == "Fish":
    pet_color = st.sidebar.text_area(label="What color is your fish?", max_chars=15)

if pet_color:
    response = lch.generate_pet_name(animal_type, pet_color)
    st.text(response["pet_name"])
