# import_streamlit.py
import streamlit as st
import cohere

# Set your Cohere API key here
co = cohere.Client('jiOXVoCCtYZHImJ8tTpTYhNfX8zCnlqhpAVVWGkW')

# Custom CSS for background image
st.markdown(
    """
    <style>
        body {
            background-image: url('E:\Officeworks\Chatbot_cohere_LLM\logo.jpg');
            background-size: cover;
            background-repeat: no-repeat;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and User Input
st.title("Eviden personal assistant")
user_input = st.text_input("You:", "")

# Function to generate response using ChatGPT
def generate_response(input_text):
    response = co.generate(
       model='command',
       prompt=input_text,
       max_tokens=300,
       temperature=0.9,
       k=0,
       stop_sequences=[],
       return_likelihoods='NONE')
    return response.generations[0].text

# Display the response
if user_input:
    response = generate_response(user_input)
    st.text("ChatGPT:")
    st.write(response)
