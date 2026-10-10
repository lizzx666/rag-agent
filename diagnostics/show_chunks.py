"""Print the full text of every chunk on one page of one paper."""
from dotenv import load_dotenv
load_dotenv()

from index import get_vectorstore

PAPER = "hyde"
PAGE = 0   # 0-based, same as metadata["page"]

vs = get_vectorstore()
data = vs.get(where={"$and": [{"paper": PAPER}, {"page": PAGE}]}, include=["documents", "metadatas"])

# Sort by start_index so chunks appear in reading order
for doc, meta in sorted(zip(data["documents"], data["metadatas"]), key=lambda x: x[1].get("start_index", 0)):
    print("=" * 80)
    print(f"{PAPER} p.{PAGE}  start_index={meta.get('start_index')}")
    print(doc)