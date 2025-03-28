
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text =""
        for page in reader.pages:
            text += page.extract_text() +"\n"
        
        return text
    except Exception as e:
        print(f"Error: {e}")
        
def get_text_chuncks(text):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=1000)
        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error: {e}")
        
def get_vector_store(chunk_text):
    try:
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        print("Hi")
        vector = FAISS.from_texts(chunk_text ,embedding=embedding)
        vector.save_local("faiss-lib")
    except Exception as e:
        print(f"Error: {e}")
        

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss-lib", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    """ models = genai.list_models()
    for model in models:
        print(model.name)
    """
   
    st.set_page_config(page_title="Chat PDF")
    st.header("Interactive RAG-based LLM for Multi-PDF Document Analysis", divider='rainbow')

    user_question = st.chat_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)
        
    #text = extract_text("PREM RAJ--Resume.pdf")
    #chunks = get_text_chuncks(text)
   # get_vector_store(chunks)
    #print(text)

if __name__ == "__main__":
    main()
