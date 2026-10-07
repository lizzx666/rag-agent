from dotenv import load_dotenv
load_dotenv()

from typing import TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from index import get_vectorstore   # reuse the same store and embedding settings as step 1


# ---------- 1) State ----------
class RAGState(TypedDict):
    question: str
    queries: list[str]          # NEW: rewritten search queries (multi-query version only)
    documents: list[Document]
    answer: str


# ---------- 2) Retriever and model ----------
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60, max_retries=2)

TOP_N = 5   # chunks passed to the answer step; same as the baseline k, so the comparison is fair


# ---------- 3) Answer generation (unchanged from step 2) ----------
def format_docs(docs: list[Document]) -> str:
    """Join chunks into one context string, each labelled with its source."""
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


# ---------- 4) Multi-query generation (structured output) ----------
class SearchQueries(BaseModel):
    queries: list[str] = Field(description="Alternative search queries for a vector database of research papers")


QUERY_PROMPT = ChatPromptTemplate.from_template(
    "You help retrieve passages from research papers on LLM post-training and RAG.\n"
    "Write {n} different search queries for the question below. Each query should approach it "
    "from a different angle or use different terminology.\n"
    "If the question compares several methods, write at least one query focused on each method.\n\n"
    "Question: {question}"
)

query_chain = QUERY_PROMPT | llm.with_structured_output(SearchQueries)


# ---------- 5) Reciprocal Rank Fusion ----------
def doc_key(doc: Document) -> str:
    """A unique id for a chunk: which paper, which page, and where on the page it starts."""
    m = doc.metadata
    return f"{m['paper']}|{m['page']}|{m.get('start_index')}"


def reciprocal_rank_fusion(results: list[list[Document]], k: int = 60, top_n: int = TOP_N) -> list[Document]:
    """Fuse several ranked lists into one. A chunk scores higher if it appears
    in more lists and near the top of them."""
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            key = doc_key(doc)
            if key not in scores:
                scores[key] = 0
                docs_by_key[key] = doc
            scores[key] += 1 / (rank + k)
    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [docs_by_key[key] for key in ranked_keys[:top_n]]


# ---------- 6) Nodes ----------
def retrieve(state: RAGState):
    """Baseline: one search with the original question."""
    return {"documents": retriever.invoke(state["question"])}


def generate_queries(state: RAGState):
    generated_queries = query_chain.invoke({"question": state["question"], "n": 4})
    return {"queries": [state["question"]] + generated_queries.queries}


def retrieve_fused(state: RAGState):
    """Multi-query: search with every query (in parallel), then fuse with RRF."""
    results = retriever.batch(state["queries"])
    return {"documents": reciprocal_rank_fusion(results)}


def generate(state: RAGState):
    context = format_docs(state["documents"])
    answer = answer_chain.invoke({"context": context, "question": state["question"]})
    return {"answer": answer}


# ---------- 7) Graph ----------
def build_graph(multi_query: bool):
    builder = StateGraph(RAGState)

    # Nodes ("retrieve" is the same node name in both versions, backed by a different function)
    builder.add_node("retrieve", retrieve_fused if multi_query else retrieve)
    builder.add_node("generate", generate)
    if multi_query:
        builder.add_node("generate_queries", generate_queries)

    # Edges
    if multi_query:
        builder.add_edge(START, "generate_queries")
        builder.add_edge("generate_queries", "retrieve")
    else:
        builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


# ---------- 8) Run: baseline vs multi-query ----------
if __name__ == "__main__":
    baseline = build_graph(multi_query=False)
    multi = build_graph(multi_query=True)

    questions = [
        "What is the difference between DPO and PPO?",
        "How does QLoRA reduce memory usage during fine-tuning?",
    ]
    for q in questions:
        for name, graph in [("BASELINE", baseline), ("MULTI-QUERY", multi)]:
            result = graph.invoke({"question": q})
            print("\n" + "=" * 80)
            print(f"[{name}] Q: {q}")
            if "queries" in result:
                print("Queries:")
                for query in result["queries"]:
                    print("  -", query)
            print("Retrieved from:", [f"{d.metadata['paper']} p.{d.metadata['page']}" for d in result["documents"]])
            print("\nA:", result["answer"])