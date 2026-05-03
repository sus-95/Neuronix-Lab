import sys
import os
sys.path.append(os.path.dirname(__file__))

import streamlit as st

from sections import home, knowledge, ai_assistant, learn_math, playground, visual_learning, rnn_lab, cnn_lab, model_comparison, hopfield_lab

# ✅ Page config
st.set_page_config(page_title="Neuronix Lab", layout="wide")

# ✅ Load Custom CSS
def load_css(file_name):
    # Try to find the file relative to THIS script
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS: {e}")

load_css("style.css")


# ✅ Title
st.markdown(
    "<h1 style='text-align:center;' class='gradient-text'>🧠 Neuronix Lab</h1>",
    unsafe_allow_html=True
)

# ✅ Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ✅ Sidebar navigation
st.sidebar.markdown("<div class='sidebar-header'>🚀 Neuronix Lab</div>", unsafe_allow_html=True)
st.sidebar.markdown("<hr style='border-color: rgba(0,0,0,0.08); margin-top: 0; margin-bottom: 1rem;'>", unsafe_allow_html=True)

def nav_button(label, icon, page_name):
    # Use primary type to highlight the active page
    btn_type = "primary" if st.session_state.page == page_name else "secondary"
    if st.sidebar.button(f"{icon} \u00A0 {label}", type=btn_type, use_container_width=True):
        st.session_state.page = page_name
        st.rerun()

st.sidebar.markdown("<div class='nav-section'>MAIN</div>", unsafe_allow_html=True)
nav_button("Home", "🏠", "Home")
nav_button("Knowledge Base", "📚", "Knowledge Base")
nav_button("Neuronix AI", "🤖", "Neuronix AI")

st.sidebar.markdown("<div class='nav-section'>TOOLS</div>", unsafe_allow_html=True)
nav_button("Learn Math", "📐", "Learn Math")
nav_button("Playground", "🧠", "Playground")
nav_button("Visual Learning", "📊", "Visual Learning")

st.sidebar.markdown("<div class='nav-section'>ADVANCED</div>", unsafe_allow_html=True)
nav_button("RNN Lab", "🔁", "RNN Lab")
nav_button("CNN Lab", "👁️", "CNN Lab")
nav_button("Model Comparison", "⚖️", "Model Comparison")
nav_button("Hopfield Lab", "🧠", "Hopfield Lab")
nav_button("Model Insights", "📈", "Model Insights")
nav_button("Settings", "⚙️", "Settings")
nav_button("About", "ℹ️", "About")

# ✅ Routing
if st.session_state.page == "Home":
    home.show()
elif st.session_state.page == "Knowledge Base":
    knowledge.show()
elif st.session_state.page == "Neuronix AI":
    ai_assistant.show()
elif st.session_state.page == "Learn Math":
    learn_math.show()
elif st.session_state.page == "Playground":
    playground.show()
elif st.session_state.page == "Visual Learning":
    visual_learning.show()
elif st.session_state.page == "RNN Lab":
    rnn_lab.show()
elif st.session_state.page == "CNN Lab":
    cnn_lab.show()
elif st.session_state.page == "Model Comparison":
    model_comparison.show()
elif st.session_state.page == "Hopfield Lab":
    hopfield_lab.show()
elif st.session_state.page in ["Model Insights", "Settings", "About"]:
    st.markdown(f"<h2 class='gradient-text'>{st.session_state.page}</h2>", unsafe_allow_html=True)
    st.info("🚀 This premium feature is currently in development and will be available soon!")
