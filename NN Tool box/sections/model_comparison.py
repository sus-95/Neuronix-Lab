import streamlit as st
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def show():
    # Initialize step
    if "mc_step" not in st.session_state:
        st.session_state.mc_step = 1

    st.markdown("<h2 class='gradient-text'>⚖️ Model Comparison Journey</h2>", unsafe_allow_html=True)
    st.markdown("<p>A step-by-step guided learning experience comparing ANN, CNN, and RNN architectures.</p>", unsafe_allow_html=True)
    
    # Progress Bar
    progress = (st.session_state.mc_step - 1) / 3.0
    st.progress(progress)
    st.markdown(f"<p style='text-align:right; font-weight:bold; color:#6C63FF; margin-top:5px;'>Step {st.session_state.mc_step} of 4</p>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin-top:0;'/>", unsafe_allow_html=True)

    # STEP 1: OVERVIEW
    if st.session_state.mc_step == 1:
        st.markdown("<h3>Step 1: The Big Picture 🌍</h3>", unsafe_allow_html=True)
        st.write("Artificial Intelligence uses different types of neural networks depending on the shape and type of data you are trying to process. Here is the high-level overview:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div style='background:#f8fafc; border:1px solid #cbd5e1; border-top:4px solid #6C63FF; padding:20px; border-radius:8px; height:220px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);'>
                <h3 style='margin-top:0; color:#1e293b;'>🧠 ANN</h3>
                <p style='color:#1e293b;'><b>Artificial Neural Network</b></p>
                <p style='color:#64748b;'>Works best on standard tabular data (CSV files, spreadsheets). Good for general prediction tasks.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div style='background:#f8fafc; border:1px solid #cbd5e1; border-top:4px solid #00C9A7; padding:20px; border-radius:8px; height:220px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);'>
                <h3 style='margin-top:0; color:#1e293b;'>👁️ CNN</h3>
                <p style='color:#1e293b;'><b>Convolutional Neural Network</b></p>
                <p style='color:#64748b;'>Works best on image and spatial data. Uses filters to detect patterns like edges and faces.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div style='background:#f8fafc; border:1px solid #cbd5e1; border-top:4px solid #f43f5e; padding:20px; border-radius:8px; height:220px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);'>
                <h3 style='margin-top:0; color:#1e293b;'>🔁 RNN</h3>
                <p style='color:#1e293b;'><b>Recurrent Neural Network</b></p>
                <p style='color:#64748b;'>Works best on sequential data (Time-series, text). Remembers past inputs to predict the future.</p>
            </div>
            """, unsafe_allow_html=True)

    # STEP 2: THEORY
    elif st.session_state.mc_step == 2:
        st.markdown("<h3>Step 2: Core Theory & Mechanics 📚</h3>", unsafe_allow_html=True)
        st.write("Let's dive a little deeper into how each of these networks actually process their information internally.")
        
        st.markdown("""
<div style='display:flex; flex-direction:column; gap:15px; margin-top: 15px;'>
<div style='background:#ffffff; padding:20px; border-radius:10px; border-left: 5px solid #6C63FF; box-shadow: 0 4px 12px rgba(0,0,0,0.03);'>
<h4 style='color:#1e293b; margin-top:0;'>🧠 ANN (Dense / Fully Connected)</h4>
<ul style='color:#475569; margin-bottom:0;'>
<li>Every neuron in one layer is connected to every single neuron in the next layer.</li>
<li>Learns global patterns across the entire dataset.</li>
<li>Struggles with very high-dimensional data (like 4K images) because it requires too many parameters.</li>
</ul>
</div>

<div style='background:#ffffff; padding:20px; border-radius:10px; border-left: 5px solid #00C9A7; box-shadow: 0 4px 12px rgba(0,0,0,0.03);'>
<h4 style='color:#1e293b; margin-top:0;'>👁️ CNN (Convolutional)</h4>
<ul style='color:#475569; margin-bottom:0;'>
<li>Uses sliding windows (Kernels/Filters) to scan the data block by block.</li>
<li>Excels at finding local spatial patterns (like an eye in a face).</li>
<li>Uses Pooling layers to shrink the image and save memory, keeping only the most important features.</li>
</ul>
</div>

<div style='background:#ffffff; padding:20px; border-radius:10px; border-left: 5px solid #f43f5e; box-shadow: 0 4px 12px rgba(0,0,0,0.03);'>
<h4 style='color:#1e293b; margin-top:0;'>🔁 RNN (Recurrent)</h4>
<ul style='color:#475569; margin-bottom:0;'>
<li>Contains internal loops that allow information to persist across time steps.</li>
<li>Maintains a 'Hidden State' (short-term memory) that is updated with each new piece of data.</li>
<li>Can suffer from 'vanishing gradients' on long sequences, which is why LSTMs were invented!</li>
</ul>
</div>
</div>
""", unsafe_allow_html=True)

    # STEP 3: VISUAL COMPARISON
    elif st.session_state.mc_step == 3:
        st.markdown("<h3>Step 3: Visual Architecture 🖼️</h3>", unsafe_allow_html=True)
        st.write("Visualizing the structural difference between the models.")
        
        tab1, tab2, tab3 = st.tabs(["🧠 ANN", "👁️ CNN", "🔁 RNN"])
        with tab1:
            st.markdown("<p style='text-align:center; color:#64748b;'>Dense matrix mapping inputs directly to hidden layers.</p>", unsafe_allow_html=True)
            G = nx.DiGraph()
            for i in range(3): G.add_node(f"I{i}", pos=(0, i), color='#00C9A7')
            for i in range(4): G.add_node(f"H{i}", pos=(1, i-0.5), color='#6C63FF')
            for i in range(2): G.add_node(f"O{i}", pos=(2, i+0.5), color='#f43f5e')
            for i in range(3):
                for j in range(4): G.add_edge(f"I{i}", f"H{j}")
            for j in range(4):
                for k in range(2): G.add_edge(f"H{j}", f"O{k}")
            
            fig, ax = plt.subplots(figsize=(6, 3), facecolor='none')
            ax.set_facecolor('none')
            colors = [nx.get_node_attributes(G, 'color').get(n) for n in G.nodes()]
            nx.draw(G, nx.get_node_attributes(G, 'pos'), ax=ax, node_color=colors, edge_color='#cbd5e1', node_size=800)
            st.pyplot(fig)
            
        with tab2:
            st.markdown("<p style='text-align:center; color:#64748b;'>Sliding filters condensing spatial grids into feature maps.</p>", unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_shape(type="rect", x0=0, y0=0, x1=3, y1=3, line=dict(color="#00C9A7", width=2), fillcolor="rgba(0, 201, 167, 0.1)")
            fig.add_shape(type="rect", x0=4, y0=0.5, x1=6, y1=2.5, line=dict(color="#00C9A7", width=2), fillcolor="rgba(0, 201, 167, 0.3)")
            fig.add_shape(type="rect", x0=7, y0=1, x1=8, y1=2, line=dict(color="#f43f5e", width=2), fillcolor="rgba(244, 63, 94, 0.5)")
            
            fig.add_annotation(x=1.5, y=3.5, text="<b>Input Image</b>", showarrow=False, font=dict(color="#1e293b", size=14))
            fig.add_annotation(x=5, y=3.0, text="<b>Feature Map</b>", showarrow=False, font=dict(color="#1e293b", size=14))
            fig.add_annotation(x=7.5, y=2.5, text="<b>Output</b>", showarrow=False, font=dict(color="#1e293b", size=14))
            
            # Draw arrows connecting them
            fig.add_annotation(x=3.5, y=1.5, ax=3, ay=1.5, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#94a3b8")
            fig.add_annotation(x=6.5, y=1.5, ax=6, ay=1.5, xref="x", yref="y", axref="x", ayref="y", showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#94a3b8")
            
            fig.update_layout(xaxis=dict(range=[-1, 9], showgrid=False, zeroline=False, visible=False),
                              yaxis=dict(range=[-1, 4], showgrid=False, zeroline=False, visible=False),
                              plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=300, font=dict(color='#1e293b'))
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.markdown("<p style='text-align:center; color:#64748b;'>Unrolled recurrent states passing memory over time.</p>", unsafe_allow_html=True)
            G = nx.DiGraph()
            for t in range(4):
                G.add_node(f"X{t}", pos=(t, 0), color='#00C9A7')
                G.add_node(f"H{t}", pos=(t, 1), color='#6C63FF')
                G.add_edge(f"X{t}", f"H{t}")
                if t > 0:
                    G.add_edge(f"H{t-1}", f"H{t}")
            
            pos = nx.get_node_attributes(G, 'pos')
            colors = [nx.get_node_attributes(G, 'color').get(n) for n in G.nodes()]
            fig, ax = plt.subplots(figsize=(8, 3), facecolor='none')
            ax.set_facecolor('none')
            nx.draw(G, pos, ax=ax, with_labels=True, node_color=colors, edge_color='#cbd5e1', node_size=1000, font_color='white', font_weight='bold')
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(-0.5, 1.5)
            st.pyplot(fig)

    # STEP 4: PERFORMANCE
    elif st.session_state.mc_step == 4:
        st.markdown("<h3>Step 4: Performance & Use Cases 📈</h3>", unsafe_allow_html=True)
        st.write("How do they stack up in the real world across different tasks? (Higher score is better)")
        
        categories = ['Tabular Data', 'Image Recognition', 'Time-Series & Text', 'Training Speed']
        
        fig = go.Figure(data=[
            go.Bar(name='ANN', x=categories, y=[10, 2, 4, 9], marker_color='#6C63FF'),
            go.Bar(name='CNN', x=categories, y=[2, 10, 3, 5], marker_color='#00C9A7'),
            go.Bar(name='RNN', x=categories, y=[3, 1, 10, 3], marker_color='#f43f5e')
        ])
        fig.update_layout(
            barmode='group', 
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#1e293b'),
            legend=dict(yanchor="top", y=1.0, xanchor="right", x=1.0)
        )
        fig.update_yaxes(title="Effectiveness Score (0-10)", showgrid=True, gridcolor='rgba(0,0,0,0.05)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.success("🎉 You've completed the Model Comparison Journey! You now understand the core differences between the three main pillars of Deep Learning.")

    st.markdown("<hr style='margin-bottom:15px;'/>", unsafe_allow_html=True)
    
    # Bottom Navigation Controls
    col_prev, col_space, col_next = st.columns([1, 2, 1])
    
    with col_prev:
        if st.session_state.mc_step > 1:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.mc_step -= 1
                st.rerun()
                
    with col_next:
        if st.session_state.mc_step < 4:
            if st.button("Next ➡️", type="primary", use_container_width=True):
                st.session_state.mc_step += 1
                st.rerun()
        else:
            if st.button("Restart Journey 🔄", use_container_width=True):
                st.session_state.mc_step = 1
                st.rerun()
