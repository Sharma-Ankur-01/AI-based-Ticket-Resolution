from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

class KnowledgeBaseEngine:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2:1b")
        self.embeddings = OllamaEmbeddings(model="llama3.2:1b")
        self.vector_store = None
        self._initialize_dummy_knowledge_base()

    def _initialize_dummy_knowledge_base(self):
        # Simulating a Knowledge Base
        df = pd.read_csv('knowledge_base.csv')
        documents = list(df['article'])
        self.vector_store = FAISS.from_texts(documents, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def recommend_articles(self, ticket_content: str):
        # RAG for recommendations
        docs = self.retriever.invoke(ticket_content)
        return [doc.page_content for doc in docs]

kb_engine = KnowledgeBaseEngine()

app = FastAPI(title="Knowledge Management Platform API")

class Ticket(BaseModel):
    content: str

@app.post("/recommend")
def recommend_content(ticket: Ticket):
    try:
        recommendations = kb_engine.recommend_articles(ticket.content)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "active", "model": "llama3.2:1b"}
