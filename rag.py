# Sources
# https://github.com/pixegami/langchain-rag-tutorial
# https://www.datacamp.com/tutorial/building-a-rag-system-with-langchain-and-fastapi?utm_cid=23552157100&utm_aid=188237542690&utm_campaign=230119_1-ps-other~dsa-tofu~ai_2-b2c_3-emea_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na&utm_loc=9196856-&utm_mtd=-c&utm_kw=&utm_source=google&utm_medium=paid_search&utm_content=ps-other~emea-en~dsa~tofu~tutorial~artificial-intelligence&gad_source=1&gad_campaignid=23552157100&gbraid=0AAAAADQ9WsGgFWG_y4lyQ0i4PMQtYlyJG&gclid=Cj0KCQjw9ZLSBhCcARIsAEhGKgN7WOsPl0gVKXCeFTrRwlKyi3bIsaBK0pMMaLvMR8tAIJ-d7TkWL9EaAsMPEALw_wcB

from dotenv import load_dotenv
import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Documents loader
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
loader = UnstructuredPDFLoader(
  file_path= "data/LaVin-DiT.pdf",
  mode= 'element',
)

documents = loader.load()

print(documents)