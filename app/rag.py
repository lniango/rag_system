# Sources
# https://github.com/pixegami/langchain-rag-tutorial
# https://www.datacamp.com/tutorial/building-a-rag-system-with-langchain-and-fastapi?utm_cid=23552157100&utm_aid=188237542690&utm_campaign=230119_1-ps-other~dsa-tofu~ai_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9196856-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~emea-en~dsa~tofu~tutorial~artificial-intelligence&gad_source=1&gad_campaignid=23552157100&gbraid=0AAAAADQ9WsGgFWG_y4lyQ0i4PMQtYlyJG&gclid=Cj0KCQjw9ZLSBhCcARIsAEhGKgN7WOsPl0gVKXCeFTrRwlKyi3bIsaBK0pMMaLvMR8tAIJ-d7TkWL9EaAsMPEALw_wcB
# Embedding : https://campus.datacamp.com/courses/introduction-to-embeddings-with-the-openai-api/what-are-embeddings?ex=4
# Pgvector : https://www.datacamp.com/tutorial/pgvector-tutorial
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline, Ollama
from transformers import pipeline
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") # Openai token
hgface_token = os.getenv("HF_TOKEN")

# Initialize the LLM (using TinyLlama/TinyLlama-1.1B-Chat-v1.0)
llm_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

pipe = pipeline(
  "text-generation",
  model = llm_model, #"google/flan-t5-base", # "google/flan-t5-base"
  max_new_tokens=256,
  temperature=0.1,
  do_sample=False,
  return_full_text=False
  #device=-1
  #token = hgface_token
)
llm = HuggingFacePipeline(pipeline=pipe)

# Function to set up the RAG system
def setup_rag_system():
  # Load documents
  documents = []
  for pdf in Path("data/papers").glob("*.pdf"):
    loader = PyPDFLoader(str(pdf))
    documents.extend(loader.load())
    
  #documents = loader.load()
  
  # Split the document into chunks
  splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
            )
  document_chunk = splitter.split_documents(documents)
  
  # Initialize embeddings 
  # Tuto RAG : https://huggingface.co/learn/cookbook/advanced_rag
  model_name = "sentence-transformers/all-mpnet-base-v2"
  model_kwargs = {"device": "cpu"}
  encode_kwargs = {"normalize_embeddings": False}
  embeddings = HuggingFaceEmbeddings(
               model_name = model_name,
               model_kwargs = model_kwargs,
               encode_kwargs = encode_kwargs
  )

  # Create FAISS vector store from document chunks and embeddings
  # FAISS helps us find the most similar vectors really fast.
  vector_store = FAISS.from_documents(
                 document_chunk, 
                 embeddings) # create a vector store

  # Return the retriever for document retrieval with specified search_type
  # k=5 means, Give me the top 5 most relevant documents.
  # It helps reduce the noise.
  retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 3}
  )
  
  return retriever


# Function to get the response from the RAG system
retriever = setup_rag_system()
async def get_rag_response(query : str):
  
  # Retrieve the relevant documents using 'get_relevant_documents' method
  retrieved_docs = retriever.get_relevant_documents(query)
  
  # Prepare the input for the LLM: Combine the query and the retrieved documents into a single string
  context = "\n".join([doc.page_content for doc in retrieved_docs])
  
  # LLM expects a list of strings (prompts), so we create one by combining the query with retrieved context
  prompt = [f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}"]

  # Generate the final response using the language model (LLM)
  generated_response = llm.generate(prompt)
  
  return generated_response


"""
QUESTIONS:
What are the differences between DiT and LaVin-DiT?
What datasets are used by these models?
How does 3D RoPE work?
"""
