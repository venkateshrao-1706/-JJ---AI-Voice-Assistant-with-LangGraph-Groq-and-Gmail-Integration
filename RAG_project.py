from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate ,ChatPromptTemplate
load_dotenv()

model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
parser = StrOutputParser()
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

file_path = ("C:/Users/venka/OneDrive/Documents/NetFlix.csv")

loader = CSVLoader(file_path=file_path,encoding="utf-8")

docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(

    chunk_size = 1000,
    chunk_overlap = 200,

)

chunks = text_splitter.split_documents(docs)


prompt = ChatPromptTemplate.from_template(
    """You are helpful movie Assistant,
    answer the questions based upon the below context only, the information stored in dictionary format (eg.{{ "title" : "value","director":"value","genre":"value","release_date":"value","rating":"value","description":"value","language":"value","country":"value",}})
    if you don't the answer just say i cannot find the related information in the context.
    Question:{question}
    """
)

vector_store = Chroma.from_documents(documents=chunks,
embedding=embedding,persist_directory="./vector_db")

retriever = vector_store.as_retriever(search_kwargs ={"k":2})

bm25r = BM25Retriever.from_documents(chunks)
bm25r.k=2

Hybrid_search = EnsembleRetriever(retrievers=[bm25r,retriever],weights=[0.5,0.5])

result = Hybrid_search.invoke("Who is director of the movie 3 idiots")

for r in result:
    print(r.page_content)
    print("\n")




# import streamlit as st

# st.write("This is my test stremalit ")

# st.text_input("Enter the idea here")

# st.text_input("Enter the name here")

# st.selectbox("Jai",["Ho","Hi","jai","kites","Hello"])

# # uv run streamlit run f.py

# st.file_uploader("Upload a file",type=["pdf","txt"])

# st.spinner("Loading...")
# st.sidebar.text_input("Enter the details")
# st.sidebar.text_input("Enter the father details")
# st.sidebar.text_input("Enter the mother details")
# st.sidebar.text_input("Enter the sibling details")
# st.sidebar.text_input("Enter the friend details")
# st.sidebar.text_input("Enter the relative details")

# rating,feedback= st.columns(2)

# with rating:
#     st.slider("Rate the app",0,10,7)
# with feedback:
#     st.text_area("Give feedback")

# st.expander("More details")

# @st.cache_data
# def load_data():
#     return pd.read_csv("data.csv")


