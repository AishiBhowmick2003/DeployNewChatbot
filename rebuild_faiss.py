
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings



# Path to the single PDF file
PDF_FILE = "./medical_book.pdf"  # Replace with your PDF file name
DB_FAISS_PATH = "./chatbot_model/db_faiss"

# Check if the PDF file exists
if not os.path.exists(PDF_FILE):
    raise FileNotFoundError(f" PDF file '{PDF_FILE}' not found!")

# Load PDF document
loader = PyMuPDFLoader(PDF_FILE)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Convert to embeddings and store in FAISS
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.from_documents(docs, embedding_model)

# Save FAISS index
db.save_local(DB_FAISS_PATH)
print("FAISS index rebuilt successfully from the single PDF!")
