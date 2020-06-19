import streamlit as st
import wikipedia as wiki
from qasystem import DocumentReader, MODEL_PATHS
import argparse


def main():

    model_choice = st.sidebar.selectbox(
        "Choose a Transformer model:",
        list(MODEL_PATHS.keys())
    )

    st.title("Basic QA with Wikipedia")
    st.write("Model is vanilla BERT trained on SQuAD 1.0")


    try:
        # Points to fine-tuned weights that we've trained
        reader = DocumentReader(MODEL_PATHS[model_choice])
    except:
        # Downloads pretrained weights from Huggingface
        reader = DocumentReader()

    question = st.text_input('Ask a Question', 'Why is the sky blue?')

    results = wiki.search(question)

    page = wiki.page(results[0])
    print(f"Top result: {page}")

    text = page.content

    reader.tokenize(question, text)

    st.write("Answer:", reader.get_answer())

    st.markdown('###### Found from this Wikipedia page:')
    st.markdown(f'<a target="_blank" href="{page.url}">{page.title}</a>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
