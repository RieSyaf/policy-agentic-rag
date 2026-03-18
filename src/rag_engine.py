import chromadb
from chromadb.utils import embedding_functions
import os

# --- Optional: Import for Local LLM ---
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# --- Configuration ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
DB_PATH = os.path.join(root_dir, "chroma_db_data")
COLLECTION_NAME = "policy_docs"

# ---  1. GLOBAL SYSTEM RULES  ---
SYSTEM_RULES = """
You are a strict but confident Policy Assistant. Answer based ONLY on the provided context.

<rules>
1. **Analyze:** Does the context contain the specific answer or highly relevant related information?
2. **If EXACT MATCH:** Answer clearly without stating references. End with a "Sources:" footer.
3. **If PARTIAL MATCH (Related Info):** Answer confidently based strictly on what the policy *does* say about the topic. Do not use hesitant language or apologize. End with a "Sources:" footer.
4. **If MISSING (Near Miss/Negative):** Do NOT say "I cannot find a definitive answer." Directly state what the policy covers in the provided text and clearly state that the user's specific scenario is not mentioned. End with a "Closest Related Clauses:" footer.
5. **No Database Speak:** NEVER use words like "Source Text", "Section Path", "Page", "Context", or "Document" in the main body of your answer. Just speak naturally.
6. **Zero Hallucination:** Never invent amounts, clauses, or rules. If the text doesn't explicitly state it, you don't know it.
7. **Citation Format:** NEVER put citations in the middle of your sentences. You must ONLY place the exact "Required Footer Citation" strings at the very end of your response in the footer. Separate multiple sources with a semicolon (;).
8. **DO NOT COPY EXAMPLES:** The filenames in the examples below (e.g., Policy_A.pdf) are FAKE. NEVER output them. ONLY use the actual Required Footer Citations provided in the CONTEXT.
</rules>

<examples>
Example 1 (Exact Match):
User: What is the cooling-off period?
Context: 
Text: "You may cancel this policy within 21 days." 
Required Footer Citation: Policy_A.pdf “General Conditions”, p.4
Answer:
The cooling-off period is 21 days from the purchase date.

Sources: Policy_A.pdf “General Conditions”, p.4

Example 3 (Missing / Near Miss):
User: What is the deductible for alien abduction?
Context: 
Text: "We cover theft and fire." 
Required Footer Citation: Policy_A.pdf “Section 3”, p.10
Answer:
The policy lists covered events such as theft and fire, but it does not mention coverage or deductibles for alien abduction.

Closest Related Clauses: Policy_A.pdf “Section 3”, p.10
</examples>

Now, answer the real user question below using the same format/tone as the examples.
"""

# --- 2. PLUGGABLE INTERFACE ---
class LLMInterface:
    def generate(self, prompt: str, system_rules: str) -> str:
        raise NotImplementedError("Subclasses must implement generate()")

# --- 3. BACKEND: MOCK (For Testing/No-Install) ---
class MockLLM(LLMInterface):
    def generate(self, prompt: str, system_rules: str) -> str:
        return (
            "🤖 [MOCK ANSWER]\n"
            "The policy lists covered events such as theft and fire, but it does not mention alien abduction.\n\n"
            "Closest Related Clauses: Policy_A.pdf “Section 3”, p.12"
        )

# --- 4. BACKEND: LOCAL OLLAMA ---
class LocalLLM(LLMInterface):
    def __init__(self, model_name="llama3.1"):
        if not OLLAMA_AVAILABLE: raise ImportError("Ollama not installed")
        self.model = model_name

    def generate(self, prompt: str, system_rules: str) -> str:
        try:
            print(f"🦙 Thinking... (Model: {self.model})")
            
            # pass the GLOBAL system_rules to the model
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': system_rules},
                {'role': 'user', 'content': prompt},
            ], options={'num_ctx': 8192})
            return response['message']['content']
        except Exception as e:
            return f"❌ Error: {str(e)}"

# --- 5. THE RAG ENGINE ---
class RAGEngine:
    def __init__(self, llm_backend: LLMInterface, collection_name: str = "qbe_policies"):
        self.llm = llm_backend
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        try:
            self.collection = self.client.get_collection(
                name=collection_name, 
                embedding_function=self.embedding_fn
            )
        except:
            print(f"⚠️ Warning: Collection '{collection_name}' not found.")
            
    def retrieve(self, query: str, k: int = 5):
        results = self.collection.query(query_texts=[query], n_results=k)
        chunks = []
        if not results['ids']: return chunks
        
        ids = results['ids'][0]
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        
        for i in range(len(ids)):
            chunks.append({
                "text": docs[i],
                "source": metas[i]['source'],
                "page": metas[i]['page'],
                "path": metas[i]['heading_path'],
                "citation": metas[i]['combined_citation']
            })
        return chunks

    def construct_prompt(self, query: str, chunks: list) -> str:
        context_text = ""
        for i, chunk in enumerate(chunks, 1):
            # We pre-bake the citation string so the LLM doesn't have to think about formatting!
            context_text += (
                f"--- SOURCE TEXT {i} ---\n"
                f"Text: {chunk['text']}\n"
                f"Required Footer Citation: {chunk['source']} “{chunk['path']}”, p.{chunk['page']}\n\n"
            )
            
        return (
            f"CONTEXT PROVIDED:\n{context_text}\n\n"
            f"USER QUESTION: {query}\n\n"
        )

    def answer_question(self, query: str):
        chunks = self.retrieve(query)
        if not chunks: return "No relevant documents found."
        
        prompt = self.construct_prompt(query, chunks)
        
        return self.llm.generate(prompt, system_rules=SYSTEM_RULES)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        backend = LocalLLM(model_name="llama3.1")
    except ImportError:
        print("⚠️ Ollama not found. Falling back to MockLLM.")
        backend = MockLLM()
    
    print(f"🚀 Initializing RAG with {backend.__class__.__name__}...")
    bot = RAGEngine(llm_backend=backend)
    
    while True:
        q = input("\nAsk a policy question (or 'exit'): ")
        if q.lower() in ['exit', 'quit']: break
        
        print(bot.answer_question(q))