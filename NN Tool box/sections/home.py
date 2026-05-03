import streamlit as st

def show():
    st.markdown("""
    <div class="glass-card">
        <h2 class="gradient-text" style="margin-top:0;">🚀 Explore Neural Networks Like Never Before</h2>
        <p style="font-size:1.1rem; color:#4a4a4a;">
            Welcome to Neuronix Lab, your interactive AI dashboard. 
            Dive deep into the mathematics, experiment with models in the playground, 
            and visualize how neural networks learn in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<div class='glass-card' style='text-align:center;'><h4>📐 Math Lab</h4><p>Understand the core calculus and algebra.</p></div>", unsafe_allow_html=True)
        if st.button("Go to Math Lab", key="btn_math", use_container_width=True):
            st.session_state.page = "Learn Math"
            st.rerun()

    with col2:
        st.markdown("<div class='glass-card' style='text-align:center;'><h4>🧠 Playground</h4><p>Train your own custom neural networks.</p></div>", unsafe_allow_html=True)
        if st.button("Go to Playground", key="btn_play", use_container_width=True):
            st.session_state.page = "Playground"
            st.rerun()

    with col3:
        st.markdown("<div class='glass-card' style='text-align:center;'><h4>📊 Visual Learning</h4><p>See weights and gradients in action.</p></div>", unsafe_allow_html=True)
        if st.button("Go to Visuals", key="btn_vis", use_container_width=True):
            st.session_state.page = "Visual Learning"
            st.rerun()