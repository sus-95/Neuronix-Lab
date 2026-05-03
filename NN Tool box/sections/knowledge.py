import streamlit as st

def show():
    st.markdown("<h2 class='gradient-text'>📚 Knowledge Base</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#4a4a4a; margin-bottom: 2rem;'>Your ultimate guide to understanding neural networks.</p>", unsafe_allow_html=True)

    # 🧠 Intro
    st.markdown("""
    <div class='glass-card'>
        <h4 style='margin-top:0;'>🧠 What is a Neural Network?</h4>
        <p>A <b>Neural Network</b> is a system inspired by the human brain that learns patterns from data.</p>
        <p>It consists of:</p>
        <ul style='color:#334155;'>
            <li><b>Input Layer</b>: Receives the initial data.</li>
            <li><b>Hidden Layers</b>: Performs computations and extracts features.</li>
            <li><b>Output Layer</b>: Produces the final prediction.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # 📜 History
        st.markdown("""
        <div class='glass-card' style='height: 100%;'>
            <h4 style='margin-top:0;'>📜 Brief History</h4>
            <ul style='color:#334155;'>
                <li><b>1958 → Perceptron</b>: First neural model (Frank Rosenblatt)</li>
                <li><b>1986 → Backpropagation</b>: Enabled deep learning (Rumelhart, Hinton)</li>
                <li><b>2010+ → Deep Learning Boom</b>: Image, speech, AI revolution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # 🧩 Types
        st.markdown("""
        <div class='glass-card' style='height: 100%;'>
            <h4 style='margin-top:0;'>🧩 Network Types</h4>
            <ul style='color:#334155;'>
                <li><b>1. ANN</b>: Basic fully connected network (structured data).</li>
                <li><b>2. CNN</b>: Convolutional Network (image processing).</li>
                <li><b>3. RNN</b>: Recurrent Network (sequential data, text).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # ⚙️ Activation & Loss
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div class='glass-card' style='height: 100%;'>
            <h4 style='margin-top:0;'>⚙️ Activation Functions</h4>
            <p><b>Sigmoid</b>: Range (0,1) → used for probability</p>
            <div style='background:#f8fafc; padding:10px; border-radius:5px; text-align:center; margin-bottom:10px; color:#1e293b; font-family:monospace;'>σ(x) = 1 / (1 + e^(-x))</div>
            <p><b>Tanh</b>: Range (-1,1) → zero centered</p>
            <div style='background:#f8fafc; padding:10px; border-radius:5px; text-align:center; margin-bottom:10px; color:#1e293b; font-family:monospace;'>tanh(x)</div>
            <p><b>ReLU</b>: Most popular → removes negative values</p>
            <div style='background:#f8fafc; padding:10px; border-radius:5px; text-align:center; color:#1e293b; font-family:monospace;'>f(x) = max(0, x)</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class='glass-card' style='height: 100%;'>
            <h4 style='margin-top:0;'>📉 Loss Functions</h4>
            <p><b>Mean Squared Error (MSE)</b></p>
            <div style='background:#f8fafc; padding:10px; border-radius:5px; text-align:center; margin-bottom:10px; color:#1e293b; font-family:monospace;'>MSE = (1/n) * Σ(y - ŷ)²</div>
            <p><b>Cross Entropy</b></p>
            <p>Used for classification problems where output is a probability distribution.</p>
        </div>
        """, unsafe_allow_html=True)

    # 🔄 Backpropagation & Applications
    st.markdown("""
    <div class='glass-card'>
        <h4 style='margin-top:0;'>🔄 Backpropagation</h4>
        <p>Backpropagation is the process of updating weights using error.</p>
        <p><b>Steps:</b></p>
        <ol style='color:#334155;'>
            <li>Forward pass (compute prediction)</li>
            <li>Calculate loss (compare to true value)</li>
            <li>Compute gradients (chain rule)</li>
            <li>Update weights</li>
        </ol>
        <div style='background:#f8fafc; padding:10px; border-radius:5px; text-align:center; color:#1e293b; font-family:monospace;'>W = W - η * (∂L/∂W)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='glass-card'>
        <h4 style='margin-top:0;'>🌍 Real World Applications</h4>
        <p style='text-align:center; font-weight:600;'>
        🤖 Chatbots (ChatGPT) &nbsp;&nbsp;|&nbsp;&nbsp; 🎥 Image Recognition (Face ID) &nbsp;&nbsp;|&nbsp;&nbsp; 📈 Stock Prediction &nbsp;&nbsp;|&nbsp;&nbsp; 🩺 Healthcare Diagnosis &nbsp;&nbsp;|&nbsp;&nbsp; 🚗 Self Driving Cars
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.success("🎯 You now understand the basics of Neural Networks!")