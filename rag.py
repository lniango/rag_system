# Sources
# https://github.com/pixegami/langchain-rag-tutorial
# https://www.datacamp.com/tutorial/building-a-rag-system-with-langchain-and-fastapi?utm_cid=23552157100&utm_aid=188237542690&utm_campaign=230119_1-ps-other~dsa-tofu~ai_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9196856-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~emea-en~dsa~tofu~tutorial~artificial-intelligence&gad_source=1&gad_campaignid=23552157100&gbraid=0AAAAADQ9WsGgFWG_y4lyQ0i4PMQtYlyJG&gclid=Cj0KCQjw9ZLSBhCcARIsAEhGKgN7WOsPl0gVKXCeFTrRwlKyi3bIsaBK0pMMaLvMR8tAIJ-d7TkWL9EaAsMPEALw_wcB
# Embedding : https://campus.datacamp.com/courses/introduction-to-embeddings-with-the-openai-api/what-are-embeddings?ex=4
# Pgvector : https://www.datacamp.com/tutorial/pgvector-tutorial
from dotenv import load_dotenv
import os

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY") # Openai token
hgface_token = os.getenv("HF_TOKEN")

# Documents loader
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader, PyPDFLoader
from pathlib import Path
"""loader = UnstructuredPDFLoader(
  file_path= "data/LaVin-DiT.pdf",
  mode= 'elements',
)"""
#loader = PyPDFLoader(
#  file_path= "data/LaVin-DiT.pdf"
#)

#loader = TextLoader(
#  file_path="data/text.txt"
#)

documents = []
for pdf in Path("data/papers").glob("*.pdf"):
  loader = PyPDFLoader(str(pdf))
  documents.extend(loader.load())

#loader2 = TextLoader('data/text.txt')
#documents2 = loader2.load()

documents = loader.load()
#print(len(documents))
#print(documents[0].page_content[:200])

# Chunking the text
"""
  Large documents are often split into smaller chunks to make them easier to index and retrieve
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=100
            )
document_chunk = splitter.split_documents(documents)

#print(document_chunk)

#Embeddings
"""
LangChain supports creating vector embeddings using various models, such as OpenAI or HuggingFace models. 
These embeddings represent the semantic meaning of the text chunks, 
which makes them suitable for similarity searches. 
"""
# Tuto RAG : https://huggingface.co/learn/cookbook/advanced_rag
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#embeddings = OpenAIEmbeddings()
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
embeddings = HuggingFaceEmbeddings(
             model_name = model_name,
             model_kwargs = model_kwargs,
             encode_kwargs = encode_kwargs
)

# Vector stores
"""
  After generating the embeddings, the next step is to store them in a vector store like PGVector, FAISS, 
  or any other supported by LangChain.
  This allows for fast and accurate retrieval of relevant documents when a query is made.
"""
from langchain_community.vectorstores import FAISS
# FAISS helps us find the most similar vectors really fast.
vector_store = FAISS.from_documents(
               document_chunk, 
               embeddings) # create a vector store


# Retrieval
"""
It's the component that goes through the indexed documents and finds the ones most relevant to a user’s query. 
"""
# k=5 means, Give me the top 5 most relevant documents.
# It helps reduce the noise.
retriever = vector_store.as_retriever(
  search_type = "similarity",
  search_kwargs = {"k": 2}
)


# Querying
"""
Add in the LLM and using it to actually generate the response based on the documents retrieved.
"""
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline, Ollama
from transformers import pipeline

# Ollama Models: 
# OpenAI: This model will be responsible for generating text responses.
#llm = OpenAI(openai_api_key=openai_api_key)
#llm_model = Ollama(
#  model= "tinyllama"#"llama3.2:1b"
#)

llm_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" #"meta-llama/Llama-3.2-3B" #"gpt2"
#"microsoft/phi-2" #"microsoft/Phi-3-mini-4k-instruct" #'google/flan-t5-base'   #"TinyLlama/TinyLlama-1.1B-Chat-v1.0"#"gpt2" #"microsoft/phi-2" #"mistralai/Mistral-7B-Instruct-v0.1"

pipe = pipeline(
  "text-generation",
  model = llm_model, #"google/flan-t5-base", # "google/flan-t5-base"
  max_new_tokens=64,
  temperature=0.1,
  do_sample=False,
  return_full_text=False
  #device=-1
  #token = hgface_token
)
llm = HuggingFacePipeline(pipeline=pipe)

# RetrievalQA: This is the special LangChain feature that combines retrieval 
# and QA (question answering). It connects the retriever (which finds the relevant documents) 
# to the LLM (which generates the answer).
qa_chain = RetrievalQA.from_chain_type(
          llm = llm,
          chain_type = "stuff",
          retriever = retriever, 
          return_source_documents=True
)

#query = "Are polar bears in danger?"
query = "What are the differences between DiT and LaVin-DiT?"
response = qa_chain.invoke(
  {"query" : query}
)

# Response
print(f"Response : {response['result']}")

"""print("\nDocuments utilisés:")
for doc in response["source_documents"]:
    print("-" * 80)
    print(doc.page_content[:500])"""
    
    
"""pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=100,
    return_full_text=False
)

response = pipe(
    "<|user|>\nAre polar bears in danger?\n<|assistant|>\n"
)

print(response[0]["generated_text"])"""

"""
QUESTIONS:
What are the differences between DiT and LaVin-DiT?
What datasets are used by these models?
How does 3D RoPE work?
"""
