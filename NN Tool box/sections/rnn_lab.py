import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, SimpleRNN, GRU
from tensorflow.keras.optimizers import Adam
from core.ai_helper import render_section_ai

def generate_data(seq_len=10):
    # Create a nice clean sine wave for predictable training
    t = np.linspace(0, 30, 300)
    data = np.sin(t) + 0.05 * np.random.randn(300)
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X).reshape(-1, seq_len, 1), np.array(y), data

def show():
    st.markdown("<h2 class='gradient-text'>🔁 RNN Architecture Lab</h2>", unsafe_allow_html=True)
    st.markdown("<p>Interactive laboratory to train, visualize, and understand Recurrent Neural Networks.</p>", unsafe_allow_html=True)
    
    # PARAMETERS
    st.markdown("<h4>⚙️ Model Configuration</h4>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    model_type = col1.selectbox("Model Type", ["LSTM", "SimpleRNN", "GRU"])
    epochs = col2.slider("Epochs", 10, 100, 30, step=10)
    seq_len = col3.slider("Sequence Length", 3, 20, 10)
    lr = col4.selectbox("Learning Rate", [0.001, 0.01, 0.1])
    
    # Store history in session state so we don't lose it on widget interactions
    if 'rnn_history' not in st.session_state:
        st.session_state.rnn_history = None
        st.session_state.rnn_model = None
        st.session_state.rnn_data = None
        
    st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
    
    if st.button("🔮 Train Model", type="primary"):
        with st.spinner(f"Training {model_type} Model... This may take a few seconds."):
            X, y, raw_data = generate_data(seq_len)
            
            model = Sequential()
            if model_type == "LSTM":
                model.add(LSTM(16, activation='relu', input_shape=(seq_len, 1)))
            elif model_type == "SimpleRNN":
                model.add(SimpleRNN(16, activation='relu', input_shape=(seq_len, 1)))
            else:
                model.add(GRU(16, activation='relu', input_shape=(seq_len, 1)))
                
            model.add(Dense(1))
            model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
            
            # Use validation split to show validation curve
            history = model.fit(X, y, epochs=epochs, validation_split=0.2, verbose=0)
            
            preds = model.predict(X)
            
            st.session_state.rnn_history = history.history
            st.session_state.rnn_data = (X, y, preds, raw_data)
            st.session_state.rnn_model = model_type
            
    if st.session_state.rnn_history is not None:
        st.markdown("<br/>", unsafe_allow_html=True)
        tabs = st.tabs(["📉 Training", "📊 Predictions", "🔍 Sequence View", "📈 Advanced"])
        
        hist = st.session_state.rnn_history
        X, y, preds, raw_data = st.session_state.rnn_data
        
        # TAB 1: TRAINING
        with tabs[0]:
            st.markdown("<h4>Training Performance</h4>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hist['loss'], mode='lines', name='Training Loss', line=dict(color='#6C63FF', width=3)))
            if 'val_loss' in hist:
                fig.add_trace(go.Scatter(y=hist['val_loss'], mode='lines', name='Validation Loss', line=dict(color='#00C9A7', width=3)))
            
            fig.update_layout(
                xaxis_title="Epochs", yaxis_title="Mean Squared Error",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, b=0, t=20),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                hovermode='x unified',
                font=dict(color='#1e293b')
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
            st.plotly_chart(fig, use_container_width=True)
            
        # TAB 2: PREDICTIONS
        with tabs[1]:
            st.markdown("<h4>Actual vs Predicted (Time Series)</h4>", unsafe_allow_html=True)
            
            plot_y = y.flatten()
            plot_preds = preds.flatten()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=plot_y, mode='lines', name='Actual Values', line=dict(color='#1e293b', width=2)))
            fig.add_trace(go.Scatter(y=plot_preds, mode='lines', name='Predicted Values', line=dict(color='#f43f5e', width=2, dash='dash')))
            
            fig.update_layout(
                xaxis_title="Time Step", yaxis_title="Value",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, b=0, t=20),
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
                hovermode='x unified',
                font=dict(color='#1e293b')
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
            st.plotly_chart(fig, use_container_width=True)
            
        # TAB 3: SEQUENCE VIEW
        with tabs[2]:
            st.markdown("<h4>Step-by-Step Sequence Processing</h4>", unsafe_allow_html=True)
            st.write("Visualizing how the model processes a single window of inputs to predict the next future value.")
            
            sample_idx = st.slider("Select Sequence Window", 0, len(X)-1, 0)
            x_seq = X[sample_idx].flatten()
            y_true = y[sample_idx]
            y_pred = preds[sample_idx][0]
            
            # Interactive Flow Diagram in HTML
            html = "<div style='display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-top:20px;'>"
            for i, val in enumerate(x_seq):
                html += f"<div style='background:#f8fafc; border:1px solid #cbd5e1; padding:8px 12px; border-radius:6px; text-align:center;'><b style='color:#475569;'>X<sub>{i}</sub></b><br/>{val:.2f}</div>"
                if i < len(x_seq) - 1:
                    html += "<div style='color:#94a3b8; font-size:18px;'>→</div>"
                    
            html += f"<div style='color:#6C63FF; font-weight:bold; font-size:24px; padding:0 10px;'>⇒</div>"
            html += f"<div style='background:rgba(108, 99, 255, 0.08); border:1px solid #6C63FF; padding:10px 15px; border-radius:8px; text-align:center;'><b style='color:#6C63FF;'>Predicted</b><br/><span style='font-size:1.1em; font-weight:bold;'>{y_pred:.2f}</span></div>"
            html += f"<div style='background:rgba(0, 201, 167, 0.08); border:1px solid #00C9A7; padding:10px 15px; border-radius:8px; text-align:center;'><b style='color:#00C9A7;'>Actual</b><br/><span style='font-size:1.1em; font-weight:bold;'>{y_true:.2f}</span></div>"
            html += "</div>"
            
            st.markdown(html, unsafe_allow_html=True)
            
        # TAB 4: ADVANCED
        with tabs[3]:
            st.markdown("<h4>Hidden State Evolution (Simulated Activation)</h4>", unsafe_allow_html=True)
            
            timesteps = len(x_seq)
            hidden_dim = 16
            # We use simulated states for quick visualization so we don't have to rebuild intermediate Keras models
            simulated_states = np.zeros((timesteps, hidden_dim))
            simulated_states[0] = np.random.randn(hidden_dim) * 0.1
            for t in range(1, timesteps):
                # Simulate state passing through time
                simulated_states[t] = simulated_states[t-1] * 0.7 + np.random.randn(hidden_dim) * 0.3
                
            fig_heatmap = px.imshow(simulated_states.T, aspect='auto', color_continuous_scale='Magma',
                            labels=dict(x="Timestep", y="Hidden Unit", color="Activation Magnitude"))
            fig_heatmap.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                margin=dict(l=0, r=0, b=0, t=20),
                font=dict(color='#1e293b')
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.markdown("<hr/>", unsafe_allow_html=True)
            st.markdown("<h4>Unrolled Sequence Flow Diagram</h4>", unsafe_allow_html=True)
            
            # Simple unrolled diagram using networkx
            G = nx.DiGraph()
            unroll_steps = min(5, timesteps)
            for t in range(unroll_steps):
                G.add_node(f"X{t}", pos=(t, 0), color='#00C9A7')
                G.add_node(f"H{t}", pos=(t, 1), color='#6C63FF')
                G.add_edge(f"X{t}", f"H{t}")
                if t > 0:
                    G.add_edge(f"H{t-1}", f"H{t}")
                if t == unroll_steps - 1:
                    G.add_node(f"Y{t}", pos=(t, 2), color='#f43f5e')
                    G.add_edge(f"H{t}", f"Y{t}")
                    
            pos = nx.get_node_attributes(G, 'pos')
            colors = [nx.get_node_attributes(G, 'color').get(node) for node in G.nodes()]
            
            fig, ax = plt.subplots(figsize=(10, 3), facecolor='none')
            ax.set_facecolor('none')
            nx.draw(G, pos, ax=ax, with_labels=True, node_color=colors, edge_color='#cbd5e1', 
                    node_size=1500, font_color='white', font_weight='bold', arrows=True, arrowsize=15)
            st.pyplot(fig)

    render_section_ai("RNN lab: Recurrent Neural Networks, LSTM, GRU, sequences, time series, hidden states, vanishing gradients, sequence length.")