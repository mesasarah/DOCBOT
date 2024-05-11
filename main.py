import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceInstructEmbeddings
import torch
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
from chromadb.config import Settings

device = torch.device('cpu')


def qa_pipeline():
    llm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    tokenizers = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    return llm, tokenizers


def chroma_settings():
    settings = Settings(
        chroma_api_impl="chromadb.api.fastapi.FastAPI",
        persist_directory='db',
        chroma_server_host='localhost',
        chroma_server_http_port='8000',
        anonymized_telemetry=False
    )
    return settings


def textsplitter(uploaded_text):
    texts = []
    for txt in uploaded_text:
        doc = Document(page_content=txt)
        texts.append(doc)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    tests = text_splitter.split_documents(texts)
    return tests


def embeddings():
    sent_embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2")
    return sent_embeddings


def database(uploaded_text):
    settings = chroma_settings()
    tests = textsplitter(uploaded_text)
    sent_embeddings = embeddings()
    client = chromadb.Client()
    cdb = Chroma.from_documents(tests, sent_embeddings, persist_directory="db", client_settings=settings, client=client)
    cdb.persist()
    return cdb


@st.cache
def qa_llm():
    llm, tokenizer = qa_pipeline()
    db = database(uploaded_text)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever, return_source_documents=True)
    return qa


def process_answer(uploaded_text):
    qa = qa_llm()
    generated_text = qa(uploaded_text)
    answer = generated_text['result']
    return answer


st.set_page_config(
    page_title="DOCBOT",
    page_icon="👾",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(f"""
    <style>
    .sidebar.fixed {{
        background-image: linear-gradient(to right, #8E236A, #4A148C);
    }}
    </style>
    """, unsafe_allow_html=True)
with st.expander("About Docbot"):
    st.markdown(
        """
        DOCBOT can read and understand any type of document including PDFs, Word Documents and many more.
        DOCBOT is still under development, this is just a demo of communication with multiple documents.
        """
    )

st.header("CHAT WITH DOCBOT 👾")
user_input = st.text_area("ASK YOUR QUERY....")
st.write("Query:" + user_input)
st.write("DocBot:")


def handle_send_button_click():
    if not user_input:
        st.error("Please enter a query to proceed.")
    return


if user_input:
    uploaded_text = user_input.split(".")  # Assuming each sentence ends with a period for text splitting
    answer = process_answer(uploaded_text)
    st.write("DocBot:", answer)
if st.button("SEND"):
    handle_send_button_click()

with st.sidebar:
    st.sidebar.title("DOCBOT 👾")
    pdfs = st.file_uploader("Upload Your Documents", accept_multiple_files=True)

    if pdfs is not None:
        uploaded_text = []
        for pdf in pdfs:
            file_extension = pdf.name.split(".")[-1].lower()
            if file_extension == "pdf":
                pdf_reader = PdfReader(pdf)
                document = []
                for page in pdf_reader.pages:
                    document.append(page.extract_text())
                uploaded_text.extend(document)
                st.sidebar.text(f"File {pdf.name} processed.")
        st.sidebar.success("Files uploaded and processed.")

previously_asked_queries = []

st.sidebar.markdown("## Previously Asked Queries")
