import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.mlp import MLP
from core.ai_helper import render_section_ai

def show():
    st.markdown("<h2 class='gradient-text'>🧠 Playground - Train Your Model</h2>", unsafe_allow_html=True)

    # Wrap top section in a glass card
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<h4>📁 Upload Dataset</h4>", unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        
        if uploaded:
            df = pd.read_csv(uploaded)
            st.success("Dataset Loaded ✅")
        else:
            st.info("Using default generated dataset")
            df = pd.DataFrame({
                "Feature 1 (x1)": np.random.rand(100),
                "Feature 2 (x2)": np.random.rand(100),
                "Target (y)": np.random.randint(0, 2, 100)
            })

    with col2:
        st.markdown("<h4>📊 Dataset Preview</h4>", unsafe_allow_html=True)
        st.dataframe(df.head(), use_container_width=True)
        
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("---")

    # Configuration Section
    st.markdown("### ⚙️ Model Configuration")
    
    col_feat, col_params = st.columns([1, 1])
    
    with col_feat:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>🎯 Feature Selection</h4>", unsafe_allow_html=True)
        columns = df.columns.tolist()
        features = st.multiselect("Select Features", columns, default=columns[:-1])
        target = st.selectbox("Select Target", columns, index=len(columns)-1)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_params:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>Hyperparameters</h4>", unsafe_allow_html=True)
        hidden = st.slider("Hidden neurons", 2, 20, 5)
        lr = st.slider("Learning Rate", 0.001, 0.1, 0.01)
        epochs = st.slider("Epochs", 10, 300, 100)
        st.markdown("</div>", unsafe_allow_html=True)

    if len(features) == 0:
        st.warning("Select at least one feature to train the model.")
        return

    # Train Section
    st.markdown("<div style='text-align:center; margin: 2rem 0;'>", unsafe_allow_html=True)
    train_clicked = st.button("🚀 Start Training", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if train_clicked:
        try:
            X = df[features].values
            y = df[target].values.reshape(-1, 1)

            # Normalize data
            X = (X - X.mean()) / (X.std() + 1e-8)

            with st.spinner("Training model..."):
                model = MLP([X.shape[1], hidden, 1], lr=lr, epochs=epochs)
                losses = model.train(X, y)

            st.success("Training Complete! 🎉")

            # Results Section
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("<h4>📉 Loss Curve</h4>", unsafe_allow_html=True)
                fig, ax = plt.subplots(facecolor='none')
                ax.set_facecolor('none')
                ax.plot(losses, color="#6C63FF", linewidth=2.5)
                ax.set_xlabel("Epochs", color="#475569")
                ax.set_ylabel("Loss", color="#475569")
                ax.tick_params(colors='#475569')
                for spine in ax.spines.values():
                    spine.set_color('#cbd5e1')
                ax.grid(True, linestyle='--', alpha=0.05, color='#000000')
                st.pyplot(fig)
                st.markdown("</div>", unsafe_allow_html=True)

            with res_col2:
                st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                st.markdown("<h4>📊 Sample Predictions</h4>", unsafe_allow_html=True)
                preds = model.predict(X)
                pred_df = pd.DataFrame({"Actual": y[:5].flatten(), "Predicted": preds[:5].flatten()})
                st.dataframe(pred_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred during training: {e}")

    render_section_ai("Playground lab: Neural network training, datasets, features, target, hidden neurons, learning rate, epochs, loss curve, predictions.")