"""
Project 6: Clinical Trial Protocol & Pharmacovigilance Assistant
RAG Pipeline Implementation using Google GenAI & ChromaDB.
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env or environment.")

# Initialize Google GenAI client
client = genai.Client(api_key=API_KEY)

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "cardiovex_protocol"
PDF_PATH = "cardiovex_clinical_trial_protocol_50p.pdf"
EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"


def get_embedding(text: str) -> list[float]:
    """Generates an embedding vector for a single string with fallback."""
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )
        return response.embeddings[0].values
    except Exception:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        return response.embeddings[0].values


def get_embeddings_batch(texts: list[str], batch_size: int = 20) -> list[list[float]]:
    """Generates embeddings for a batch of strings."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
            )
            embeddings.extend([e.values for e in response.embeddings])
        except Exception:
            # Try item-by-item with fallback
            for item in batch:
                embeddings.append(get_embedding(item))
    return embeddings


def initialize_vector_store():
    """Ingests PDF, chunks, embeds, and persists to ChromaDB if not already indexed."""
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    doc_count = collection.count()
    if doc_count > 0:
        print(f"[RAG Pipeline] Loaded existing ChromaDB collection '{COLLECTION_NAME}' ({doc_count} documents).")
        return collection

    pdf_file = Path(PDF_PATH)
    if not pdf_file.exists():
        raise FileNotFoundError(f"Protocol PDF not found at {PDF_PATH}")

    print(f"[RAG Pipeline] Ingesting {PDF_PATH} page-by-page...")
    reader = PdfReader(str(pdf_file))
    total_pages = len(reader.pages)
    print(f"[RAG Pipeline] Total pages found: {total_pages}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    documents = []
    metadatas = []
    ids = []
    global_chunk_idx = 0

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        page_label = f"Page {page_num}"
        raw_text = page.extract_text() or ""
        cleaned_text = raw_text.strip()

        if not cleaned_text:
            continue

        chunks = text_splitter.split_text(cleaned_text)
        for chunk_idx, chunk_text in enumerate(chunks):
            global_chunk_idx += 1
            chunk_id = f"{page_label}_chunk_{chunk_idx + 1}"
            documents.append(chunk_text)
            metadatas.append(
                {
                    "page_number": page_label,
                    "page_int": page_num,
                    "chunk_id": chunk_id,
                }
            )
            ids.append(f"doc_{global_chunk_idx}")

    print(f"[RAG Pipeline] Generated {len(documents)} chunks across {total_pages} pages.")
    print(f"[RAG Pipeline] Computing embeddings using {EMBEDDING_MODEL}...")

    embeddings = get_embeddings_batch(documents)

    # Insert into ChromaDB in batches
    add_batch_size = 100
    for i in range(0, len(documents), add_batch_size):
        collection.add(
            ids=ids[i : i + add_batch_size],
            documents=documents[i : i + add_batch_size],
            embeddings=embeddings[i : i + add_batch_size],
            metadatas=metadatas[i : i + add_batch_size],
        )

    print(f"[RAG Pipeline] Successfully indexed and persisted {collection.count()} chunks to {CHROMA_PATH}.")
    return collection


# Initialize collection on module load
collection = initialize_vector_store()

SYSTEM_PROMPT = """You are a strict, objective Clinical Trial Protocol & Pharmacovigilance Assistant.
Your ONLY knowledge source is the provided context from the CardioVex Phase III clinical trial protocol.

CRITICAL INSTRUCTIONS:
1. Answer the question relying ONLY and EXCLUSIVELY on the explicit statements in the retrieved context.
2. NEVER use external clinical knowledge, prior knowledge, assumptions, extrapolation, or speculation.
3. If the answer to the question is not present, cannot be found, or is not directly supported by the context, respond with EXACTLY:
[not found]
4. If the answer is found in the context:
   - Provide a direct, concise, factual answer based strictly on the text.
   - Accurately state the source page number(s) where the answer was found (e.g., Page 14).
   - Format your response strictly as:
Answer: <concise grounded answer>
Sources: <comma-separated page labels, e.g., Page 14>
"""


def answer(question: str) -> dict:
    """Answers a question using RAG over the CardioVex clinical trial protocol.

    Args:
        question: The natural language question to be answered.

    Returns:
        dict: {"answer": <str>, "sources": [<page_or_chunk_id>, ...]}
    """
    # 1. Embed query
    query_vector = get_embedding(question)

    # 2. Retrieve top k=5 chunks
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=5,
    )

    retrieved_docs = results["documents"][0] if results.get("documents") else []
    retrieved_metas = results["metadatas"][0] if results.get("metadatas") else []

    if not retrieved_docs:
        return {"answer": "[not found]", "sources": []}

    # Format context blocks
    context_blocks = []
    for idx, (doc, meta) in enumerate(zip(retrieved_docs, retrieved_metas)):
        page_lbl = meta.get("page_number", "Unknown")
        chunk_id = meta.get("chunk_id", f"chunk_{idx}")
        context_blocks.append(f"--- Document Chunk {idx + 1} [{page_lbl} | ID: {chunk_id}] ---\n{doc}")

    context_str = "\n\n".join(context_blocks)

    user_prompt = f"""Retrieved Context:
{context_str}

Question: {question}

Follow the system instructions strictly. If the answer is not in the context, output exactly "[not found]".
Otherwise, format your response as:
Answer: <concise grounded answer>
Sources: <comma-separated page labels, e.g. Page 14>
"""

    response = None
    candidate_models = [LLM_MODEL, "gemini-3.8-flash", "gemini-2.0-flash"]
    for model_name in candidate_models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.0,
                ),
            )
            if response:
                break
        except Exception as e:
            if model_name == candidate_models[-1]:
                raise e
            continue

    raw_text = response.text.strip() if response.text else ""

    # Exact Not-Found Rule check
    if not raw_text or "[not found]" in raw_text.lower():
        return {"answer": "[not found]", "sources": []}

    answer_val = ""
    sources_val = []

    for line in raw_text.splitlines():
        line_clean = line.strip()
        if line_clean.lower().startswith("answer:"):
            answer_val = line_clean.split(":", 1)[1].strip()
        elif line_clean.lower().startswith("sources:"):
            src_str = line_clean.split(":", 1)[1].strip()
            found_pages = re.findall(r"Page\s+\d+", src_str, re.IGNORECASE)
            sources_val.extend([f"Page {re.search(r'\d+', p).group()}" for p in found_pages])

    if not answer_val:
        # If no "Answer:" prefix was provided, extract non-source lines
        ans_lines = [l for l in raw_text.splitlines() if not l.strip().lower().startswith("sources:")]
        answer_val = "\n".join(ans_lines).strip()

    if not sources_val:
        # Search for page citations in the text
        found_pages = re.findall(r"Page\s+\d+", raw_text, re.IGNORECASE)
        if found_pages:
            sources_val = [f"Page {re.search(r'\d+', p).group()}" for p in found_pages]
        else:
            # Fallback to retrieved metadata pages that match key terms or all retrieved pages
            sources_val = [m["page_number"] for m in retrieved_metas if "page_number" in m]

    def page_sort_key(p):
        match = re.search(r"\d+", str(p))
        return int(match.group()) if match else 0

    deduped_sources = sorted(list(set(sources_val)), key=page_sort_key)

    return {"answer": answer_val, "sources": deduped_sources}


if __name__ == "__main__":
    test_q = "What is the minimum required estimated Glomerular Filtration Rate (eGFR) threshold for a patient to be included in the CardioVex Phase III trial?"
    print(f"Testing Question: {test_q}")
    res = answer(test_q)
    print("Result:", res)
