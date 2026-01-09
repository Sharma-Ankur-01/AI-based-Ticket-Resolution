from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class KnowledgeBaseEngine:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2:1b")
        self.embeddings = OllamaEmbeddings(model="llama3.2:1b")
        self.vector_store = None
        self._initialize_dummy_knowledge_base()

    def _initialize_dummy_knowledge_base(self):
        # Simulating a Knowledge Base
        documents = [
            "To reset your password, go to settings > security > reset password.",
            "If the system is slow, try clearing your browser cache and cookies.",
            "For billing inquiries, please contact finance@example.com.",
            "VPN connection issues can often be resolved by restarting the Cisco AnyConnect client.",
            "To request new software, submit a ticket to IT with the software name and business justification.",
            "Meeting room booking is done via the Outlook Calendar integration.",
            "Slack notifications can be managed in Preferences > Notifications.",
            "The coffee machine on the 3rd floor is serviced every Tuesday.",
        ]
        self.vector_store = FAISS.from_texts(documents, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

    def recommend_articles(self, ticket_content: str):
        # RAG for recommendations
        docs = self.retriever.invoke(ticket_content)
        return [doc.page_content for doc in docs]

    def detect_gaps(self, queries: list):
        valid_topics = ["password", "billing", "vpn", "software", "meeting", "slack"]
        gaps = []
        for q in queries:
            if not any(topic in q.lower() for topic in valid_topics):
                gaps.append(q)
        return list(set(gaps))

kb_engine = KnowledgeBaseEngine()

app = FastAPI(title="Knowledge Management Platform API")

class Ticket(BaseModel):
    content: str

class GapRequest(BaseModel):
    queries: list[str]

@app.post("/recommend")
def recommend_content(ticket: Ticket):
    try:
        recommendations = kb_engine.recommend_articles(ticket.content)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/gaps")
def analyze_gaps(request: GapRequest):
    try:
        gaps = kb_engine.detect_gaps(request.queries)
        return {"content_gaps": gaps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "active", "model": "llama3.2:1b"}