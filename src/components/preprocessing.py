
from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.exception.exception_base import ProjectException
import os
import sys
from langchain_google_genai import GoogleGenerativeAIEmbeddings;
import json



def __path__():
    return os.path.join(os.getcwd(),'faiss-lib')

def extract_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text =""
        for page in reader.pages:
            text += page.extract_text() +"\n"
        
        return text
    except Exception as e:
        ProjectException(e,sys)
        print(f"Error: {e}")
        
def load_json(path):
    try:
        with open(path, 'r', encoding="utf-8") as file:
            data = file.read().strip()
            if not data:
                raise ValueError(f"Error: The file {path} is empty.")
            
            data = json.loads(data)
            
            if isinstance(data, dict):
                text = json.dumps(data, indent=2)
            elif isinstance(data, list):
                text = "\n".join(json.dumps(item) for item in data)
            else:
                raise ValueError("Unexpected JSON format.")

            if not text:
                raise ValueError("Extracted text is empty.")

            return text
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        ProjectException(e, sys)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        ProjectException(e, sys)
    except ValueError as e:
        print(f"Value error: {e}")
        ProjectException(e, sys)
    except Exception as e:
        ProjectException(e, sys)
        print(f"Unexpected error: {e}")


    
def get_text_chunks(text):
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Invalid text input for chunking.")
            
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        
        if not chunks:
            raise ValueError("Text chunking resulted in an empty list.")
        
        return chunks
    except Exception as e:
        ProjectException(e, sys)
        print(f"Error: {e}")
        return []
    

def get_vector_store(chunk_text):
    try:
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector = FAISS.from_texts(chunk_text ,embedding=embedding)
        vector.save_local(__path__())

    except Exception as e:
        ProjectException(e,sys)
        print(f"Error: {e}")
        
    
def load(data_path="src/components/data_set/datas.json",file_type="json"):
    try:
        if file_type == "pdf":
            text = extract_text(data_path)
        elif file_type == "json":
            text = load_json(data_path)
        
    
        chunks = get_text_chunks(text)
        get_vector_store(chunks)

    except Exception as e:
        print("err ",e)
        ProjectException(e,sys)
        
