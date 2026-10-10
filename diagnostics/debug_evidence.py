"""Check whether evidence snippets exist inside single chunks, and where they rank for the eval question."""
from dotenv import load_dotenv
load_dotenv()

from index import get_vectorstore
from check_questions import normalize

# PAPER = "lora"
# SNIPPETS = [   # copy exactly from questions.json
#     "we constrain its update by representing the latter with a low-rank decomposition",
#     "is frozen and does not receive gradient updates",
#     "We use a random Gaussian initialization for A",
# ]
# QUESTION = "In LoRA, how is the update to a pre-trained weight matrix represented, and how are the low-rank matrices initialized?"

PAPER = "self_rag"
SNIPPETS = [
    "Reflection tokens are categorized into retrieval and critique tokens",
]
QUESTION = "What special tokens does Self-RAG introduce, and what are they used for?"

MAX_K = 30

vs = get_vectorstore()
data = vs.get(where={"paper": PAPER}, include=["documents", "metadatas"])
results = vs.similarity_search(QUESTION, k=MAX_K)   # search once, reuse for every snippet

for snippet in SNIPPETS:
    target = normalize(snippet)
    print(f"\nSnippet: {snippet}")

    # 1) Is the snippet fully inside any single chunk of this paper?
    matches = [(m["page"], m.get("start_index")) for d, m in zip(data["documents"], data["metadatas"])
               if target in normalize(d)]
    if not matches:
        print("  Not inside any single chunk (split across chunks or garbled by PDF extraction)")
        continue
    print(f"  Found in chunks (page, start_index): {matches}")

    # 2) Where does that chunk rank for the eval question?
    rank = next((i for i, doc in enumerate(results, start=1) if target in normalize(doc.page_content)), None)
    print(f"  Rank for the eval question: {rank}" if rank else f"  Not in the top {MAX_K}")