from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from index import get_vectorstore   # reuse the same store and embedding settings as step 1


# ---------- 1) State ----------
class RAGState(TypedDict):
    question: str
    documents: list[Document]   # chunks returned by the retriever
    answer: str


# ---------- 2) Retriever and model ----------
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60, max_retries=2)


# ---------- 3) Helpers ----------
def format_docs(docs: list[Document]) -> str:
    """Join chunks into one context string, each labelled with its source,
    so the model can cite where each fact came from."""
    parts = []
    for doc in docs:
        source = f"[{doc.metadata['paper']}, p.{doc.metadata['page']}]"
        parts.append(f"{source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


ANSWER_PROMPT = ChatPromptTemplate.from_template(
    "You are an assistant answering questions about research papers on LLM post-training and RAG.\n"
    "Use ONLY the context below. If the context does not contain the answer, say you don't know.\n"
    "After each claim, cite its source in the form [paper, p.N] exactly as shown in the context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n"
    "Answer:"
)

answer_chain = ANSWER_PROMPT | llm | StrOutputParser()


# ---------- 4) Nodes ----------
def retrieve(state: RAGState):
    documents = retriever.invoke(state["question"])
    return {"documents": documents}


def generate(state: RAGState):
    context = format_docs(state["documents"])
    answer = answer_chain.invoke({"context": context, "question": state["question"]})
    return {"answer": answer}


# ---------- 5) Graph ----------
builder = StateGraph(RAGState)

# Nodes
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

#Edges
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)


graph = builder.compile()


# ---------- 6) Run ----------
if __name__ == "__main__":
    questions = [
        "How does QLoRA reduce memory usage during fine-tuning?",
        "What is the difference between DPO and PPO?",
        "What is the capital of France?",   # not in the papers: the model should say it doesn't know
    ]
    for q in questions:
        result = graph.invoke({"question": q})
        print("\n" + "=" * 80)
        print("Q:", q)
        print("Retrieved from:", [f"{d.metadata['paper']} p.{d.metadata['page']}" for d in result["documents"]])
        print("\nA:", result["answer"])