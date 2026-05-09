from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ---------- CONFIG ----------
COLLECTION_NAME = "islamic_knowledge"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
QDRANT_CLOUD_URL = os.getenv("QDRANT_CLOUD_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ---------- LOAD MODELS (once at startup) ----------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Connecting to Qdrant Cloud...")
client = QdrantClient(url=QDRANT_CLOUD_URL, api_key=QDRANT_API_KEY)
print("✅ Ready!")

# ---------- FASTAPI SETUP ----------
app = FastAPI()

# Allow frontend (localhost:5173) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- REQUEST MODEL ----------
class QueryRequest(BaseModel):
    query: str

# ---------- FUNCTION: GET ANSWER FROM OPENROUTER ----------
def get_answer(query, context):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3-8b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an Islamic assistant. Answer ONLY from the given context.\n"
                            "If unsure, say you are not sure.\n"
                            "Keep answers short and clear."
                        )
                    },
                    {
                        "role": "user",
                        "content": f"Question: {query}\n\nContext:\n{context}"
                    }
                ]
            }
        )
        data = response.json()
        print("\n🔍 API RAW RESPONSE:\n", data)

        if "choices" not in data:
            return "❌ API error or no response from model."
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ---------- API ENDPOINT ----------
@app.post("/api/ask")
def ask(request: QueryRequest):
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Step 1: Embed query
    print(f"\n🔎 Searching for: {query}")
    query_vector = model.encode(query).tolist()

    # Step 2: Search Qdrant
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=10,
        with_payload=True,
    )
    print(f"✅ Found {len(results)} results")

    # Step 3: Build context
    context_list = []
    for res in results:
        print(f"Score: {res.score}")
        if res.score < 0.2:
            continue
        context_list.append(res.payload["combined_text"])

    if not context_list:
        raise HTTPException(status_code=404, detail="No relevant Islamic knowledge found for this query.")

    context = "\n\n".join(context_list[:5])
    print(f"\n📦 Context preview:\n{context[:300]}")

    # Step 4: Get answer from OpenRouter
    print("\n🤖 Calling OpenRouter...")
    answer = get_answer(query, context)
    print(f"\n📖 Answer: {answer}")

    return {"answer": answer}

# ---------- HEALTH CHECK ----------
@app.get("/")
def health():
    return {"status": "NurAI backend is running ✅"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
