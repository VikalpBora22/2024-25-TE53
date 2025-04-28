import streamlit as st
import os
import time
import concurrent.futures

# LangChain imports
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer

# ------------------------ Setup ------------------------
os.makedirs('pdfFiles', exist_ok=True)
os.makedirs('vectorDB', exist_ok=True)

st.title("Chatbot - Talk to PDFs (Llama 3)")

# ------------------------ Load Embedding Model Offline ------------------------
embedding_model_path = "local_embedding_model"
if not os.path.exists(embedding_model_path):
    with st.spinner("Downloading embedding model... (only once)"):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        model.save(embedding_model_path)

embedding_function = HuggingFaceEmbeddings(model_name=embedding_model_path)

# ------------------------ Initialize Session State ------------------------
if 'template' not in st.session_state:
    st.session_state.template = """You are a chatbot that strictly answers based on the provided documents.

    Context: {context}
    History: {history}

    User: {question}
    Chatbot:
    - If the information is in the document, provide a clear answer.
    - If the answer is not found, respond with: "I couldn't find relevant information in the provided documents.".
    """

if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )

if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question",
    )

if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(
        base_url="http://localhost:11434",
        model="llama3",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if os.path.exists('vectorDB') and 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(
        persist_directory='vectorDB',
        embedding_function=embedding_function,
    )

# ------------------------ Multi-file PDF Uploader ------------------------
uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

def process_pdf(file):
    """Processes a single PDF file into text chunks."""
    file_path = os.path.join('pdfFiles', file.name)
    
    with open(file_path, 'wb') as f:
        f.write(file.read())

    loader = PyPDFLoader(file_path)
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=100, length_function=len
    )
    return text_splitter.split_documents(data)

# ------------------------ Process PDFs & Build Vectorstore ------------------------
if uploaded_files:
    st.text(f"Processing {len(uploaded_files)} files...")

    with st.status("Processing files..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            all_splits = []
            future_to_file = {executor.submit(process_pdf, file): file for file in uploaded_files}
            for future in concurrent.futures.as_completed(future_to_file):
                all_splits.extend(future.result())

        st.session_state.vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=embedding_function
        )
        st.session_state.vectorstore.persist()

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=st.session_state.llm,
            chain_type='stuff',
            retriever=st.session_state.retriever,
            verbose=True,
            chain_type_kwargs={
                "verbose": True,
                "prompt": st.session_state.prompt,
                "memory": st.session_state.memory,
            }
        )

# ------------------------ Chat Interface ------------------------
if 'qa_chain' in st.session_state:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["message"])

    if user_input := st.chat_input("You:", key="user_input"):
        user_message = {"role": "user", "message": user_input}
        st.session_state.chat_history.append(user_message)

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(user_input)
                result_text = response.get('result', '').strip()

                # If no relevant information found, respond with the correct message
                if not result_text or result_text.lower().startswith("i couldn't find"):
                    result_text = "I couldn't find relevant information in the provided documents."

            message_placeholder = st.empty()
            full_response = ""

            for chunk in result_text.split():
                time.sleep(0.05)
                full_response += chunk + " "
                message_placeholder.markdown(full_response + " ▌")

            message_placeholder.markdown(full_response.strip())

        chatbot_message = {"role": "assistant", "message": full_response.strip()}
        st.session_state.chat_history.append(chatbot_message)

else:
    st.write("Please upload one or more PDF files to start the chatbot.")



# python -m streamlit run main5.py --server.fileWatcherType none
