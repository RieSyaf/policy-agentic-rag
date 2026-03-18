import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Import your working agent and directive from your backend
from src.agent import agent, system_directive

# --- Page Configuration ---
st.set_page_config(
    page_title="QBE Policy Assistant",
    page_icon="🛡️",
    layout="centered"
)

# --- Main Chat UI ---
st.title("🛡️ QBE Policy Assistant")
st.markdown("Ask a question, and the agent will route it to the correct legal database.")

# A handy reset button for your live presentation
if st.button("🗑️ Clear Chat"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Policy Router. Are you looking for information on a specific QBE policy or general insurance standards?"}
    ]
    st.session_state.lc_history = [SystemMessage(content=system_directive)]
    st.rerun()

st.divider()

# --- Initialize Dual Session State ---
# 1. VISUAL HISTORY (What the user sees on screen)
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI Policy Router. Are you looking for information on a specific QBE policy or general insurance standards?"}
    ]

# 2. AGENT HISTORY (What the LLM reads)
if "lc_history" not in st.session_state:
    st.session_state.lc_history = [SystemMessage(content=system_directive)]

# --- Display Visual Chat History ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input & Execution ---
if prompt := st.chat_input("Ask a policy question..."):
    # 1. Display User Message on UI
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 2. Add to both memories
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.lc_history.append(HumanMessage(content=prompt))

    # 3. Stream the Agent's Thought Process
    with st.chat_message("assistant"):
        with st.spinner("Analyzing intent and routing to databases..."):
            
            agent_reply = "I'm sorry, I couldn't process that request."
            
            # Watch the agent step-by-step
            for step in agent.stream({"messages": st.session_state.lc_history}):
                if "tools" in step:
                    # 🎯 A tool was executed! Grab its output and stop.
                    tool_messages = [msg for msg in step["tools"]["messages"] if msg.type == "tool"]
                    if tool_messages:
                        agent_reply = tool_messages[-1].content
                        break 
                elif "agent" in step:
                    # Agent asks a clarification question
                    agent_reply = step["agent"]["messages"][-1].content
                    
            # Display the final answer
            st.markdown(agent_reply)
            
    # 4. Save the agent's answer to both memories
    st.session_state.messages.append({"role": "assistant", "content": agent_reply})
    st.session_state.lc_history.append(AIMessage(content=agent_reply))
    
    # --- 5. THE MEMORY WIPE (Fixes Context Bleed) ---
    # We wipe the AGENT'S memory so the next question is fresh, 
    # but the VISUAL memory stays on the screen!
    clarification_text = "Are you asking about the specific QBE policy"
    if clarification_text not in agent_reply:
        # Wipe the LLM's brain back to default
        st.session_state.lc_history = [SystemMessage(content=system_directive)]