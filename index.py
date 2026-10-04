from dotenv import load_dotenv
load_dotenv()
import time

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")
PERSIST_DIR = "chroma_db"      # the vector store is saved here, so we only embed once
COLLECTION = "papers"


# ---------- 1) Load ----------
def load_papers():
    """Load every PDF in data/. PyPDFLoader returns one Document per page."""
    docs = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        pages = PyPDFLoader(str(pdf_path)).load()
        for page in pages:
            # PyPDFLoader already adds "source" and "page"; add a short paper name too,
            # so later we can see (and filter by) which paper a chunk came from
            page.metadata["paper"] = pdf_path.stem
        docs.extend(pages)
        print(f"Loaded {pdf_path.name}: {len(pages)} pages")
    return docs


# ---------- 2) Split ----------
def split_docs(docs):
    """Split pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        
        chunk_overlap=150,     
        add_start_index=True,  # store each chunk's position in its page (useful for merging neighbours later)
    )
    splits = splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks")
    return splits


# ---------- 3) Build once, load afterwards ----------
# def get_vectorstore():
#     #embeddings = OpenAIEmbeddings(model="text-embedding-3-small", timeout=30, max_retries=2)
#     embeddings = OpenAIEmbeddings(
#         model="text-embedding-3-small",
#         timeout=30,
#         max_retries=2,
#         chunk_size=100,   # send 100 chunks per request instead of up to 1000
#         )

#     if Path(PERSIST_DIR).exists():
#         vectorstore = Chroma(
#             collection_name=COLLECTION,
#             embedding_function=embeddings,
#             persist_directory=PERSIST_DIR,
#         )
#         print("Loading existing vector store")
#         return vectorstore

#     print("Building vector store (first run only)")
#     splits = split_docs(load_papers())
#     vectorstore = Chroma.from_documents(
#         documents=splits,
#         embedding=embeddings,
#         collection_name=COLLECTION,
#         persist_directory=PERSIST_DIR,
#     )
#     return vectorstore
def get_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=512,      # smaller vectors (default 1536): about 3x less data to download
        timeout=30,
        max_retries=2,
    )

    if Path(PERSIST_DIR).exists():
        vectorstore = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
        print("Loading existing vector store")
        return vectorstore

    print("Building vector store (first run only)")
    splits = split_docs(load_papers())

    # Create an empty store, then add chunks in small batches with progress output,
    # so we can see whether it is moving and how long each batch takes
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    batch_size = 50
    for i in range(0, len(splits), batch_size):
        batch = splits[i:i + batch_size]
        start = time.time()
        vectorstore.add_documents(batch)
        done = min(i + batch_size, len(splits))
        print(f"Embedded {done}/{len(splits)} chunks ({time.time() - start:.1f}s)")
    return vectorstore


# ---------- 4) Sanity check ----------
if __name__ == "__main__":
    vectorstore = get_vectorstore()
    print("Chunks in store:", vectorstore._collection.count())

    test_questions = [
        "How does QLoRA reduce memory usage during fine-tuning?",   
        "What is the difference between DPO and PPO?",
        "How does HyDE improve the quality of generated answers?",
    ]
    for q in test_questions:
        print("\n" + "=" * 80)
        print("Q:", q)
        results = vectorstore.similarity_search_with_score(q, k=4)
        for doc, score in results:
            print(f"  [{doc.metadata['paper']} p.{doc.metadata['page']}] score={score:.3f}")
            print("   ", doc.page_content[:150].replace("\n", " "))