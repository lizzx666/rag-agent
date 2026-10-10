from dotenv import load_dotenv
load_dotenv()

from typing import Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

from index import get_vectorstore   # reuse the same store and embedding settings as step 1


# ---------- 1) State ----------
class RAGState(TypedDict):
    question: str               # the user's original question (never changes)
    search_query: str           # the query actually used for retrieval (CRAG may rewrite it)
    queries: list[str]          # rewritten search queries (multi-query version only)
    documents: list[Document]   # chunks passed to the answer step
    retrieved_documents: list[Document]   # NEW: chunks retrieved before CRAG grading (for evaluation)
    rewrite_count: int
    answer: str


# ---------- 2) Retriever, model, limits ----------
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60, max_retries=2)

TOP_N = 5          # chunks passed to the answer step
MAX_REWRITES = 1   # NEW: CRAG may rewrite the query at most this many times before giving up


# ---------- 3) Answer generation (unchanged) ----------
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


# ---------- 4) Multi-query generation (unchanged) ----------
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


# ---------- 5) Reciprocal Rank Fusion (unchanged) ----------
def doc_key(doc: Document) -> str:
    """A unique id for a chunk: which paper, which page, and where on the page it starts."""
    m = doc.metadata
    return f"{m['paper']}|{m['page']}|{m.get('start_index')}"


def reciprocal_rank_fusion(results: list[list[Document]], k: int = 60, top_n: int = TOP_N) -> list[Document]:
    """Fuse several ranked lists into one."""
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


# ---------- 6) CRAG: grader and rewriter (NEW) ----------
class GradeDocument(BaseModel):
    relevant: Literal["yes", "no"] = Field(
        description="'yes' if the passage helps answer the question, otherwise 'no'"
    )


GRADE_PROMPT = ChatPromptTemplate.from_template(
    "You are grading whether a retrieved passage is relevant to a question.\n"
    "Answer 'yes' if the passage contains information that helps answer the question, even partially.\n"
    "Answer 'no' if it is off-topic.\n\n"
    "Question: {question}\n\n"
    "Passage:\n{document}"
)

grade_chain = GRADE_PROMPT | llm.with_structured_output(GradeDocument)


REWRITE_PROMPT = ChatPromptTemplate.from_template(
    "Rewrite the question below into a better search query for a vector database: "
    "use precise technical terms and drop filler words. Keep the same meaning.\n"
    "Return only the new query.\n\n"
    "Question: {question}\n"
    "Previous search query: {search_query}"
)

rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()

NO_ANSWER = "I couldn't find relevant information about this in the papers I have access to."


# ---------- 7) Nodes ----------
def current_query(state: RAGState) -> str:
    """The query to search with: the rewritten one if CRAG produced it, otherwise the original question."""
    return state.get("search_query") or state["question"]


def retrieve(state: RAGState):
    """Baseline: one search."""
    return {"documents": retriever.invoke(current_query(state))}


def generate_queries(state: RAGState):
    q = current_query(state)
    generated = query_chain.invoke({"question": q, "n": 4})
    return {"queries": [q] + generated.queries}


def retrieve_fused(state: RAGState):
    """Multi-query: search with every query (in parallel), then fuse with RRF."""
    results = retriever.batch(state["queries"])
    return {"documents": reciprocal_rank_fusion(results)}


def grade_documents(state: RAGState):
    docs = state["documents"]
    grades = grade_chain.batch([{"question": state["question"], "document": d.page_content} for d in docs])
    relevant = [d for d, g in zip(docs, grades) if g.relevant == "yes"]
    dropped = [f"{d.metadata['paper']} p.{d.metadata['page']}" for d, g in zip(docs, grades) if g.relevant == "no"]
    print(f"  [grade] kept {len(relevant)}/{len(docs)} chunks, dropped: {dropped}")
    # keep the unfiltered list too, so the evaluation can tell "never retrieved" from "dropped by the grader"
    return {"documents": relevant, "retrieved_documents": docs}


def rewrite_question(state: RAGState):
    rewritten = rewrite_chain.invoke({"question": state["question"], "search_query": current_query(state)})   
    new_query = rewritten.strip()
    print(f"  [rewrite] new search query: {new_query}")
    return {"search_query": new_query, "rewrite_count": state.get("rewrite_count", 0) + 1}


def generate(state: RAGState):
    context = format_docs(state["documents"])
    answer = answer_chain.invoke({"context": context, "question": state["question"]})
    return {"answer": answer}


def no_answer(state: RAGState):
    """Fixed reply set by code, instead of hoping the model says 'I don't know'."""
    return {"answer": NO_ANSWER}


# ---------- 8) Routing ----------
def route_after_grading(state: RAGState) -> str:
    if state["documents"]:                                  # at least one relevant chunk survived grading
        return "generate"
    if state.get("rewrite_count", 0) < MAX_REWRITES:        # nothing relevant, but we can still try again
        return "rewrite_question"
    return "no_answer"                                      # nothing relevant and no rewrites left


# ---------- 9) Graph ----------
def build_graph(multi_query: bool, crag: bool = False):
    builder = StateGraph(RAGState)

    # Nodes
    builder.add_node("retrieve", retrieve_fused if multi_query else retrieve)
    builder.add_node("generate", generate)
    if multi_query:
        builder.add_node("generate_queries", generate_queries)
    if crag:
        builder.add_node("grade_documents", grade_documents)
        builder.add_node("rewrite_question", rewrite_question)
        builder.add_node("no_answer", no_answer)

    # Edges
    search_start = "generate_queries" if multi_query else "retrieve"   # where a (new) search begins
    builder.add_edge(START, search_start)
    if multi_query:
        builder.add_edge("generate_queries", "retrieve")

    if crag:
        builder.add_edge("retrieve", "grade_documents")
        builder.add_conditional_edges(
            "grade_documents", route_after_grading, ["generate", "rewrite_question", "no_answer"]
        )
        # TODO: after rewriting, start a new search: rewrite_question -> search_start
        builder.add_edge("rewrite_question", search_start)
        builder.add_edge("no_answer", END)
    else:
        builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()


# ---------- 10) Run ----------
if __name__ == "__main__":
    crag_graph = build_graph(multi_query=True, crag=True)

    questions = [
        "What is the capital of France?",                          # off-topic: should end in no_answer
        "What is the difference between DPO and PPO?",
        "How does QLoRA reduce memory usage during fine-tuning?",
    ]
    for q in questions:
        print("\n" + "=" * 80)
        print("Q:", q)
        result = crag_graph.invoke({"question": q})
        print("Search query used:", result.get("search_query", q))
        print("Rewrites:", result.get("rewrite_count", 0))
        print("Kept chunks from:", [f"{d.metadata['paper']} p.{d.metadata['page']}" for d in result["documents"]])
        print("\nA:", result["answer"])