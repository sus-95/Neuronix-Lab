import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import tensorflow as tf
import os
import time
from core.handwriting_model import get_hopfield_patterns, CLASSES, center_image
from core.ai_helper import render_section_ai

# Load Model
@st.cache_resource
def load_cnn_model():
    model_path = 'core/letter_cnn.h5'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

def train_hopfield(patterns):
    N = len(patterns[0])
    W = np.zeros((N, N))
    for p in patterns:
        W += np.outer(p, p)
    np.fill_diagonal(W, 0)
    W /= len(patterns)
    return W

def recall_step(state, W):
    new_state = np.sign(W @ state)
    new_state[new_state == 0] = 1
    return new_state

def show():
    st.markdown("<h2 class='gradient-text'>🧠 Neural Handwriting Lab</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; margin-bottom: 2rem;'>The ultimate synergy: <b>CNN</b> for recognition and <b>Hopfield</b> for reconstruction. Draw a letter and watch the neural convergence!</p>", unsafe_allow_html=True)

    # Initialize Models
    cnn_model = load_cnn_model()
    
    # Hopfield parameters
    H_SIZE = 14
    if "hf_W_14" not in st.session_state:
        patterns = get_hopfield_patterns(size=H_SIZE)
        st.session_state.hf_W_14 = train_hopfield(patterns)
        st.session_state.hf_patterns_14 = patterns
    
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = 0
    if "hf_recalled" not in st.session_state:
        st.session_state.hf_recalled = None
    if "raw_input_pattern" not in st.session_state:
        st.session_state.raw_input_pattern = None

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4>🎨 Drawing Canvas</h4>", unsafe_allow_html=True)
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=20,
            stroke_color="#6366f1",
            background_color="#f8fafc",
            height=300,
            width=300,
            drawing_mode="freedraw",
            key="smooth_canvas",
            update_streamlit=True,
        )
        
        st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
        
        if st.button("🚀 Analyze Handwriting", type="primary", use_container_width=True, key="analyze_btn"):
            if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
                with st.spinner("Neural networks thinking..."):
                    # 1. Process for CNN
                    img_data = canvas_result.image_data[:, :, 0:3]
                    img = Image.fromarray(img_data.astype('uint8')).convert('L')
                    img_array_np = 255 - np.array(img)
                    img_inverted = Image.fromarray(img_array_np.astype('uint8'))
                    img_centered = center_image(img_inverted)
                    
                    img_final = np.array(img_centered).astype('float32') / 255.0
                    if cnn_model:
                        img_final_thresh = np.where(img_final > 0.1, img_final, 0)
                        preds = cnn_model.predict(img_final_thresh.reshape(1, 28, 28, 1), verbose=0)
                        idx = np.argmax(preds)
                        st.session_state.prediction = CLASSES[idx]
                        st.session_state.confidence = float(preds[0][idx])
                    
                    # 2. Process for Hopfield
                    img_14 = img_centered.resize((H_SIZE, H_SIZE), Image.Resampling.LANCZOS)
                    binary = np.where(np.array(img_14) > 50, 1, -1).flatten()
                    st.session_state.raw_input_pattern = binary.reshape(H_SIZE, H_SIZE)
                    st.session_state.hf_recalled = np.copy(st.session_state.raw_input_pattern)
                    st.rerun()
            else:
                st.warning("Please draw a letter on the canvas first!")

        if st.button("🧹 Clear Canvas", use_container_width=True, key="clear_btn"):
            st.session_state.prediction = None
            st.session_state.confidence = 0
            st.session_state.hf_recalled = None
            st.session_state.raw_input_pattern = None
            st.rerun()
            
        st.markdown("</div>", unsafe_allow_html=True)


    with col2:
        if st.session_state.prediction:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4>🤖 Recognition Result</h4>", unsafe_allow_html=True)
            
            conf_val = st.session_state.confidence * 100
            st.markdown(f"""
            <div style='text-align: center; padding: 10px;'>
                <p style='color: #64748b; font-size: 1rem; margin-bottom: 0;'>CNN predicted this letter as:</p>
                <h1 style='font-size: 6rem; margin: 0; color: #6366f1;'>{st.session_state.prediction}</h1>
                <p style='color: #6366f1; font-weight: bold;'>Confidence: {conf_val:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Animation Section
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4>🧠 Hopfield Reconstruction</h4>", unsafe_allow_html=True)
            
            viz_placeholder = st.empty()
            
            if st.button("▶️ Start Neural Convergence", use_container_width=True):
                state = st.session_state.raw_input_pattern.flatten()
                for i in range(8): # Reduced steps for snappier feel
                    state = recall_step(state, st.session_state.hf_W_14)
                    st.session_state.hf_recalled = state.reshape(H_SIZE, H_SIZE)
                    
                    with viz_placeholder.container():
                        v1, v2 = st.columns(2)
                        with v1:
                            st.markdown("<p style='text-align:center; font-size: 0.7rem;'>Drawn (Centered)</p>", unsafe_allow_html=True)
                            fig = px.imshow(st.session_state.raw_input_pattern, color_continuous_scale=[[0, '#0f172a'], [1, '#6366f1']])
                            fig.update_layout(width=150, height=150, coloraxis_showscale=False, margin=dict(l=5,r=5,t=5,b=5))
                            st.plotly_chart(fig, use_container_width=True, key=f"in_{i}")
                        with v2:
                            st.markdown(f"<p style='text-align:center; font-size: 0.7rem;'>Reconstructing... {i+1}</p>", unsafe_allow_html=True)
                            fig = px.imshow(st.session_state.hf_recalled, color_continuous_scale=[[0, '#0f172a'], [1, '#10b981']])
                            fig.update_layout(width=150, height=150, coloraxis_showscale=False, margin=dict(l=5,r=5,t=5,b=5))
                            st.plotly_chart(fig, use_container_width=True, key=f"rec_{i}")
                    time.sleep(0.25)
                st.success("Reconstruction complete!")
            else:
                with viz_placeholder.container():
                    v1, v2 = st.columns(2)
                    with v1:
                        st.markdown("<p style='text-align:center; font-size: 0.7rem;'>Drawn (Centered)</p>", unsafe_allow_html=True)
                        fig = px.imshow(st.session_state.raw_input_pattern, color_continuous_scale=[[0, '#0f172a'], [1, '#6366f1']])
                        fig.update_layout(width=150, height=150, coloraxis_showscale=False, margin=dict(l=5,r=5,t=5,b=5))
                        st.plotly_chart(fig, use_container_width=True, key="static_in")
                    with v2:
                        st.markdown("<p style='text-align:center; font-size: 0.7rem;'>Hopfield Result</p>", unsafe_allow_html=True)
                        fig = px.imshow(st.session_state.hf_recalled, color_continuous_scale=[[0, '#0f172a'], [1, '#10b981']])
                        fig.update_layout(width=150, height=150, coloraxis_showscale=False, margin=dict(l=5,r=5,t=5,b=5))
                        st.plotly_chart(fig, use_container_width=True, key="static_rec")
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='glass-card' style='display:flex; flex-direction:column; align-items:center; justify-content:center; height:350px; color:#94a3b8; text-align:center;'>", unsafe_allow_html=True)
            st.markdown("<h3 style='color:#94a3b8;'>Ready for Drawing</h3>", unsafe_allow_html=True)
            st.markdown("<p>Please draw a letter on the left to see the AI analysis and neural animation.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # Patterns Bank
    with st.expander("📚 Stored Neural Patterns (A-Z)"):
        patterns = st.session_state.hf_patterns_14
        cols = st.columns(13)
        for i in range(13):
            with cols[i]:
                st.markdown(f"<p style='text-align:center; font-size:0.6rem; margin-bottom:0;'>{CLASSES[i]}</p>", unsafe_allow_html=True)
                fig = px.imshow(patterns[i].reshape(H_SIZE, H_SIZE), color_continuous_scale=[[0, '#0f172a'], [1, '#64748b']])
                fig.update_layout(width=50, height=50, coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True, key=f"stored_{i}")
        
        cols = st.columns(13)
        for i in range(13, 26):
            with cols[i-13]:
                st.markdown(f"<p style='text-align:center; font-size:0.6rem; margin-bottom:0;'>{CLASSES[i]}</p>", unsafe_allow_html=True)
                fig = px.imshow(patterns[i].reshape(H_SIZE, H_SIZE), color_continuous_scale=[[0, '#0f172a'], [1, '#64748b']])
                fig.update_layout(width=50, height=50, coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0))
                st.plotly_chart(fig, use_container_width=True, key=f"stored_{i}")

    render_section_ai("Hopfield lab: Hopfield networks, associative memory, pattern reconstruction, neural convergence, energy function.")
