import streamlit as st
from transformers import pipeline, TFAutoModelForQuestionAnswering, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "deepset/roberta-base-squad2"
model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, framework="tf")

# Streamlit app layout
st.title("LegalEase - Legal Document Question Answering Tool")

st.write("### Instructions")
st.write("1. Paste a legal document or contract in the text area below.")
st.write("2. Ask specific questions based on the document.")
st.write("3. Click 'Get Answer' to retrieve the most relevant response.")

# Text input area for legal document
context = st.text_area("Paste the legal document or contract here:", height=250)

# Text input area for question
question = st.text_input("Ask a question:")

# Button to get the answer
if st.button("Get Answer"):
    if context and question:
        result = qa_pipeline(question=question, context=context)
        st.write("### Answer:")
        st.write(result['answer'])
    else:
        st.write("Please provide both the document and the question.")
