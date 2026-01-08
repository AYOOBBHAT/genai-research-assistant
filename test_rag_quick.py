
from app.rag.loader import load_text_file
from app.rag.chunker import chunk_text
from app.rag.pipeline import build_index_from_texts, rag_answer, default_hf_generator
from app.rag.retriever import Retriever
from sentence_transformers import SentenceTransformer
import numpy as np

# load simple text
texts = ["AI helps doctors diagnose diseases."]

# embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
embed_fn = lambda t: np.array(embed_model.encode(t))

# build index
store = build_index_from_texts(texts, embed_fn)
retriever = Retriever(embed_fn, store)

# llm wrapper (small test)
llm = default_hf_generator('gpt2')

res = rag_answer("How does AI help in healthcare?", retriever, llm_generate=llm)
print(res)
