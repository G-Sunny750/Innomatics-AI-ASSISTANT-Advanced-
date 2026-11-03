import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# --- 1. PAGE SETUP & UI BEAUTIFICATION ---
st.set_page_config(
    page_title="Innomatics Student Assistant", 
    page_icon="🎓", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #c9d1d9;
    }
    .stChatInput input {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }
    h1, h2, h3 {
        color: #58a6ff !important;
    }
    .stAlert {
        background-color: #1f6feb22;
        color: #58a6ff;
        border: 1px solid #1f6feb;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. SIDEBAR INFORMATION PANEL ---
with st.sidebar:
    st.markdown("### 🎓 Innomatics Hub")
    st.markdown("Your official AI assistant for prospective and enrolled students.")
    st.markdown("---")
    st.markdown("### 📚 Quick Navigation")
    st.markdown("- 📊 **Data Science & GenAI**")
    st.markdown("- 📈 **Data Analytics**")
    st.markdown("- 💻 **Full Stack Development**")
    st.markdown("- 🤖 **Advanced GenAI**")
    st.markdown("- 🤝 **Trainers & Mentors**")
    st.markdown("- 🏢 **Branch Locations**")
    st.markdown("---")
    st.info("💡 **Tip**: Ask direct questions like *'Give me the detailed syllabus for Data Analytics with all sub-topics'!")

st.markdown("""
    <h1 style='text-align: center; font-size: 2.5rem;'>🎓 Innomatics Student Query AI Assistant</h1>
    <p style='text-align: center; color: #8b949e; font-size: 1.1rem;'>
        Unified Intelligence Portal for <b>Prospective Applicants</b> & <b>Enrolled Associates</b>.
    </p>
""", unsafe_allow_html=True)
st.markdown("---")

# --- 3. LOAD ALL LOCAL PDF & JSON FILES ---
@st.cache_resource
def initialize_vector_db():
    docs = []
    search_dirs = ["data", "."]
    
    for d in search_dirs:
        if os.path.exists(d):
            for file in os.listdir(d):
                file_path = os.path.join(d, file)
                if os.path.isfile(file_path):
                    if file.endswith(".pdf"):
                        try:
                            loader = PyPDFLoader(file_path)
                            pdf_docs = loader.load()
                            for doc in pdf_docs:
                                doc.metadata["source_file"] = file.lower()
                            docs.extend(pdf_docs)
                        except Exception as e:
                            st.warning(f"Error loading PDF {file}: {e}")
                    elif file.endswith(".json"):
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                docs.append(Document(page_content=content, metadata={"source_file": file.lower()}))
                        except Exception as e:
                            st.warning(f"Error loading JSON {file}: {e}")

    if not docs:
        docs = [Document(page_content="Innomatics Research Labs offers Advanced Data Science with GenAI, Data Analytics, and Full Stack Web Development.")]

    # Larger chunk size ensures complete module topics are retrieved together
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(split_docs, embeddings)
    return vectorstore

vectorstore = initialize_vector_db()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# --- 4. DETAILED & STRICT PROMPT ---
template = """You are the official, authoritative Student Assistant for Innomatics Research Labs. 
Your responses must be direct, highly detailed, accurate, and based strictly on the retrieved institute records below.

CRITICAL RULES:
1. When a user asks for a **course syllabus** (e.g., Data Analytics or Data Science), you MUST list out every module along with their **complete sub-topics and bullet points** as found in the text. Do not just summarize module titles; provide the full breakdown of topics under Python, EDA, MySQL, Power BI, etc.
2. DO NOT add weak disclaimers like "this might not be comprehensive" or "check the official website". Speak with absolute confidence.
3. If the user asks about **staff, trainers, mentors, or placement officers**, use the JSON staff records.
4. If information is missing from the retrieved context, state clearly and directly: "I do not have that specific record in my current database."

Retrieved Records:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)
llm = Ollama(model="llama3.2")

def format_docs(docs):
    return "\n\n".join(f"\n\n{doc.page_content}" for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- 5. STREAMLIT CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your Innomatics Student Assistant. How can I assist you with detailed course syllabi, fee structures, or staff details today?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("Ask about detailed syllabus with topics, fees, trainers, or mentors..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Retrieving detailed institute records..."):
        retrieved_docs = retriever.invoke(user_query)
        bot_reply = rag_chain.invoke(user_query)
        
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
            
            with st.expander("🔍 View Verified Source Files & Documents"):
                for doc in retrieved_docs:
                    source_name = doc.metadata.get('source_file', 'Institute Record')
                    st.markdown(f"**Source File:** `{source_name}`")
                    st.text(doc.page_content[:350] + "...")
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})