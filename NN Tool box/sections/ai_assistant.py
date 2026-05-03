import streamlit as st
from core.ai_helper import ask_ai, init_ai

def show():
    # Page Header
    st.markdown(
        """
        <div style='background: white; padding: 2.5rem; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.03); margin-bottom: 2rem; border-left: 6px solid #6C63FF;'>
            <h1 style='margin: 0; font-size: 2.5rem; background: linear-gradient(135deg, #0f172a 0%, #6C63FF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Neuronix AI Tutor</h1>
            <p style='color: #64748b; font-size: 1.1rem; margin-top: 0.5rem;'>Your intelligent companion for mastering Neural Networks and Deep Learning concepts.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Silent init check
    if not init_ai():
        st.error("⚠️ AI Assistant is not configured. Please check your API key in secrets.toml.")
        return

    # AI Quick Actions
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([1,1,1,0.5])
    with col1:
        if st.button("📖 Backprop?", use_container_width=True):
            st.session_state.ai_chat_history.append({"role": "user", "content": "Explain backpropagation simply."})
            st.rerun()
    with col2:
        if st.button("🧠 LSTMs?", use_container_width=True):
            st.session_state.ai_chat_history.append({"role": "user", "content": "What are LSTMs?"})
            st.rerun()
    with col3:
        if st.button("👁️ CNNs?", use_container_width=True):
            st.session_state.ai_chat_history.append({"role": "user", "content": "How do CNNs work?"})
            st.rerun()
    with col4:
        if st.button("🗑️", help="Clear Chat", use_container_width=True):
            st.session_state.ai_chat_history = [{"role": "assistant", "content": "Chat cleared. How can I help you today?"}]
            st.rerun()


    # Chat Interface
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = [
            {"role": "assistant", "content": "Hello! I'm Neuronix AI. Ask me anything about Neural Networks, Deep Learning, or Machine Learning!"}
        ]

    # Display chat history
    for msg in st.session_state.ai_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    if prompt := st.chat_input("Ask a question about Neural Networks..."):
        # Display user msg
        st.session_state.ai_chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = "User is in the dedicated Neuronix AI chat interface. Provide a helpful, general explanation."
                response = ask_ai(prompt, context)
                st.markdown(response)
        st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
