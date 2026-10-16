import streamlit as st


def render_question_input():
    return st.text_area("Question", placeholder="Enter a customer support question")


def render_response(message="Your response will appear here."):
    st.info(message)