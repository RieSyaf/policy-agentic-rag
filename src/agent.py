from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

# Ensure this matches your project structure
try:
    from src.rag_engine import RAGEngine, LocalLLM
except ImportError:
    from rag_engine import RAGEngine, LocalLLM

print("🚀 Initializing Agentic Backend...")

# 1. Initialize the shared LLM backend
rag_backend = LocalLLM(model_name="llama3.1") 

# 2. Initialize TWO separate RAG engines
print("   -> Connecting to QBE Database...")
qbe_bot = RAGEngine(llm_backend=rag_backend, collection_name="qbe_policies")

print("   -> Connecting to General Insurance Database...")
general_bot = RAGEngine(llm_backend=rag_backend, collection_name="general_insurance")

# 3. STRICT SCHEMAS FOR THE TOOLS 
class SearchInput(BaseModel):
    query: str = Field(description="Pass the user's EXACT, literal question. DO NOT add filler words like 'in QBE policies' or 'according to'.")

# 4. UPDATED TOOLS WITH SCHEMAS
@tool("search_qbe_specific_policy", args_schema=SearchInput)
def search_qbe_specific_policy(query: str) -> str:
    """Use this tool to answer questions about QBE policies."""
    print(f"\n[System] 🛠️ Routing to QBE Database for query: {query}")
    return qbe_bot.answer_question(query)

@tool("search_general_insurance_knowledge", args_schema=SearchInput)
def search_general_insurance_knowledge(query: str) -> str:
    """Use this tool to answer questions about general insurance industry standards."""
    print(f"\n[System] 🛠️ Routing to General Knowledge Database for query: {query}")
    return general_bot.answer_question(query)

# 5. Define the Agent's Core Logic (Simple and Explicit)
system_directive = """You are an Agentic Routing Assistant.
You have two tools: search_qbe_specific_policy and search_general_insurance_knowledge.

CRITICAL RULES:
1. If the user asks about specific policy clauses, coverage scenarios, or explicitly mentions "QBE", use the search_qbe_specific_policy tool.
2. If the user asks broad definitional questions, general concepts, or mentions "general/industry", use the search_general_insurance_knowledge tool.
3. Output the tool's exact answer. Do not add explanations or your own thoughts.
"""

# 6. Build the Agent "Brain"
agent_llm = ChatOllama(model="llama3.1", temperature=0)
tools = [search_qbe_specific_policy, search_general_insurance_knowledge]
agent = create_react_agent(agent_llm, tools)

# --- MAIN EXECUTION (CLI Chat for Testing) ---
if __name__ == "__main__":
    print("\n🤖 Agentic Router Online! I am ready to route your questions.")
    print("   Type 'exit' to quit.\n")
    
    chat_history = [SystemMessage(content=system_directive)]
    
    while True:
        user_input = input("You: ")
        
        if not user_input.strip():
            continue
            
        if user_input.lower() in ['exit', 'quit']:
            break
            
        chat_history.append(HumanMessage(content=user_input))
        agent_reply = "I'm sorry, I couldn't process that request."
        
        # Stream the agent's thought process
        for step in agent.stream({"messages": chat_history}):
            if "tools" in step:
                # 🎯 A tool was executed!
                tool_messages = [msg for msg in step["tools"]["messages"] if msg.type == "tool"]
                if tool_messages:
                    agent_reply = tool_messages[-1].content
                    break 
            elif "agent" in step:
                # Agent asks a clarification question
                agent_reply = step["agent"]["messages"][-1].content
                
        print(f"\nAgent: {agent_reply}\n")
        chat_history.append(AIMessage(content=agent_reply))
        
        # --- THE MEMORY WIPE (Fixes Context Bleed) ---
        # If the agent actually gave a database answer, the transaction is done.
        # We wipe the memory so the next question is treated fresh!
        clarification_text = "Are you asking about the specific QBE policy"
        if clarification_text not in agent_reply:
            chat_history = [SystemMessage(content=system_directive)]