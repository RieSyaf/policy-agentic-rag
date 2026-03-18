import json
import os
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
# If your script is inside 'src/', you need to step out to root.
# If this script is in the root directory alongside your json files, use current_dir
# root_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(current_dir) # Change to os.path.dirname(current_dir) if in src/

DB_PATH = os.path.join(root_dir, "chroma_db_data")

def index_json_to_collection(json_filename: str, collection_name: str, client, embed_fn):
    input_file = os.path.join(root_dir, json_filename)
    
    # 1. Check if input exists
    if not os.path.exists(input_file):
        print(f"❌ Error: '{input_file}' not found. Run ingestion first.")
        return

    # 2. Load JSON Chunks
    print(f"\n📂 Loading chunks from '{json_filename}'...")
    with open(input_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"   Found {len(chunks)} chunks to index.")

    # 5. Create or Get Collection
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"} 
    )

    # 6. Prepare Data for Upsert
    ids = []
    documents = []
    metadatas = []

    print("🔄 Processing metadata and preparing batches...")
    
    for idx, chunk in enumerate(chunks):
        chunk_id = f"{chunk['source']}_{chunk['page']}_{idx}"
        
        flat_metadata = {
            "source": chunk["source"],
            "page": str(chunk["page"]), # Ensure it's string format
            "clause_number": chunk["clause_number"] if chunk["clause_number"] else "",
            "header_level": chunk["header_level"],
            "heading_path": " > ".join(chunk["heading_path"]) if chunk["heading_path"] else "General",
            "combined_citation": chunk["metadata"]["combined_citation"]
        }

        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append(flat_metadata)

    # 7. Upsert to Database (Batching)
    batch_size = 100
    total_batches = (len(ids) + batch_size - 1) // batch_size

    print(f"🚀 Indexing {len(ids)} documents into ChromaDB Collection: '{collection_name}'...")
    
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i : i + batch_size]
        batch_docs = documents[i : i + batch_size]
        batch_meta = metadatas[i : i + batch_size]
        
        collection.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )
        print(f"   ✅ Indexed batch {i // batch_size + 1}/{total_batches}")

    print(f"🎉 Success! Database saved to '{DB_PATH}'.")
    print(f"   Collection: {collection_name}")
    print(f"   Total Count: {collection.count()}")

def main():
    # 3. Initialize ChromaDB
    print("⚙️  Initializing Vector Database...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # 4. Setup Embedding Function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Execute for QBE
    index_json_to_collection("qbe_chunks.json", "qbe_policies", client, embedding_fn)
    
    # Execute for General Insurance
    index_json_to_collection("general_chunks.json", "general_insurance", client, embedding_fn)
    
    print("\n✅ All indexing operations complete! Phase 1 is officially done.")

if __name__ == "__main__":
    main()