import streamlit as st
import numpy as np
from math_utils.activations import sigmoid, tanh
from core.ai_helper import render_section_ai

def show():
    st.markdown("<h2 class='gradient-text'>📐 Neural Network Math Lab</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4a4a4a;'>Understand the core calculus behind backpropagation step-by-step.</p>", unsafe_allow_html=True)

    # ✅ Initialize session state
    if "computed" not in st.session_state:
        st.session_state.computed = False

    if "result" not in st.session_state:
        st.session_state.result = {}

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4>🧾 Model Inputs</h4>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        x_input = st.text_input("Inputs (X) - comma separated", "1,2,3")
        w_input = st.text_input("Weights (W) - comma separated", "0.2,0.4,0.6")
        y_true = st.number_input("True Value (y)", 1.0)
        
    with col2:
        b = st.number_input("Bias", 0.0)
        act = st.selectbox("Activation Function", ["Sigmoid", "Tanh"])
        lr = st.slider("Learning Rate", 0.001, 0.1, 0.01)
        
    st.markdown("</div>", unsafe_allow_html=True)

    # 🚀 Compute
    st.markdown("<div style='text-align:center; margin: 1.5rem 0;'>", unsafe_allow_html=True)
    compute_btn = st.button("⚙️ Compute Step-by-Step", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if compute_btn:
        try:
            X = np.array(list(map(float, x_input.split(","))))
            W = np.array(list(map(float, w_input.split(","))))

            if len(X) != len(W):
                st.error("❌ Inputs and Weights must have the same length")
                return

            # 1️⃣ Weighted sum
            z = np.dot(X, W) + b

            # 2️⃣ Activation
            if act == "Sigmoid":
                a = sigmoid(z)
                grad = a * (1 - a)
            else:
                a = tanh(z)
                grad = 1 - a**2

            # 3️⃣ Error
            error = (y_true - a) ** 2

            # 4️⃣ Updated weights
            new_W = W - lr * grad * X

            # Save state
            st.session_state.result = {
                "z": z,
                "a": a,
                "error": error,
                "grad": grad,
                "new_W": new_W
            }

            st.session_state.computed = True

        except Exception as e:
            st.error(f"Error computing values: {e}")

    # 📊 Display Results (Persistent)
    if st.session_state.computed:
        res = st.session_state.result

        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>📊 Calculation Results</h4>", unsafe_allow_html=True)
        
        r_col1, r_col2, r_col3, r_col4 = st.columns(4)
        
        with r_col1:
            st.metric(label="1️⃣ Weighted Sum (z)", value=f"{res['z']:.4f}")
            
        with r_col2:
            st.metric(label="2️⃣ Activation (a)", value=f"{res['a']:.4f}")
            
        with r_col3:
            st.metric(label="3️⃣ Error (Loss)", value=f"{res['error']:.4f}")
            
        with r_col4:
            st.metric(label="4️⃣ Gradient", value=f"{res['grad']:.4f}")

        st.write("---")
        st.markdown("<h5>5️⃣ Updated Weights (W_new)</h5>", unsafe_allow_html=True)

        w_cols = st.columns(len(res["new_W"]))

        for i, (col, w) in enumerate(zip(w_cols, res["new_W"])):
            col.metric(label=f"W{i+1}", value=f"{w:.4f}")
            
        st.markdown("</div>", unsafe_allow_html=True)
        
    render_section_ai("Learn Math lab: Neural Network Math, formulas, activation functions (sigmoid, tanh), backpropagation, weights, loss.")