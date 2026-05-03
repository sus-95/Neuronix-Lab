import streamlit as st
import google.generativeai as genai
import os

def init_ai():
    # Attempt to load API key from secrets, environment, or session state
    api_key = st.session_state.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    try:
        if not api_key and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
        
    if api_key:
        genai.configure(api_key=api_key)
        return True
    return False

def ask_ai(question, context=""):
    """
    Ask the AI a question with a given context.
    Restricts the AI to Neural Networks, Deep Learning, and Machine Learning topics.
    """
    if not init_ai():
        return "⚠️ API Key not found. Please add your Gemini API Key in the 'Neuronix AI' page."
        
    system_prompt = f"""You are Neuronix AI, an expert assistant for the 'Neuronix Lab' application.
Your domain is strictly restricted to Neural Networks, Deep Learning, and Machine Learning.
If the user asks a question unrelated to these topics, you MUST respond exactly with:
"I specialize in Neural Networks and Deep Learning topics."

Current Application Context:
{context}

Please provide clear, beginner-friendly explanations. Keep your response concise, well-structured, and use markdown.
"""
    
    model = genai.GenerativeModel('gemini-flash-latest')
    
    full_prompt = f"{system_prompt}\n\nUser Question: {question}"
    
    try:
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        # Fallback to gemini-pro-latest if flash fails
        try:
            model_alt = genai.GenerativeModel('gemini-pro-latest')
            response = model_alt.generate_content(full_prompt)
            return response.text
        except:
            return f"⚠️ Error communicating with AI: {str(e)}"

def render_section_ai(context_info):
    """
    Renders a small AI helper box for a specific section.
    """
    st.markdown("---")
    st.markdown("<div class='ai-helper-box'>", unsafe_allow_html=True)
    st.markdown("#### 🤖 Neuronix AI Helper")
    
    with st.expander("Ask AI about this section", expanded=False):
        user_q = st.text_input("Ask a question:", key=f"ai_q_{context_info[:10].replace(' ', '_')}")
        if st.button("Ask", key=f"ai_btn_{context_info[:10].replace(' ', '_')}"):
            if user_q:
                with st.spinner("Neuronix AI is thinking..."):
                    answer = ask_ai(user_q, context_info)
                st.markdown(f"<div class='ai-response-box'>{answer}</div>", unsafe_allow_html=True)
            else:
                st.warning("Please enter a question.")
    st.markdown("</div>", unsafe_allow_html=True)
