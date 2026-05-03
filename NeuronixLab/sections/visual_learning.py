import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import make_classification, make_regression, make_blobs

# Safely import activations
try:
    from math_utils.activations import sigmoid, tanh
except ImportError:
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    def tanh(x): return np.tanh(x)

def show():
    st.markdown("<h2 class='gradient-text'>📊 Visual Learning Lab</h2>", unsafe_allow_html=True)
    st.markdown("<p>Explore data visually, understand network structures, and view 2D/3D representations.</p>", unsafe_allow_html=True)

    # --- 1. DATASET SYSTEM ---
    st.markdown("<h4>📁 Dataset Configuration</h4>", unsafe_allow_html=True)
    
    data_source = st.radio("Choose Data Source", ["Sample Dataset", "Generate Custom Dataset", "Upload CSV"], horizontal=True)
    
    df = None
    
    if data_source == "Sample Dataset":
        sample_type = st.selectbox("Select Type", ["Classification", "Regression", "Clustering"])
        if sample_type == "Classification":
            X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)
            df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(4)])
            df["Target"] = y
        elif sample_type == "Regression":
            X, y = make_regression(n_samples=200, n_features=4, noise=0.1, random_state=42)
            df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(4)])
            df["Target"] = y
        else:
            X, y = make_blobs(n_samples=200, centers=3, n_features=4, random_state=42)
            df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(4)])
            df["Target"] = y
            
    elif data_source == "Generate Custom Dataset":
        col1, col2 = st.columns(2)
        n_samples = col1.slider("Number of Samples", 50, 1000, 200)
        noise = col2.slider("Noise Level", 0.0, 1.0, 0.1)
        # Fix the ValueError by explicitly setting informative features
        X, y = make_classification(n_samples=n_samples, n_features=3, n_informative=2, n_redundant=0, n_classes=2, flip_y=noise, random_state=42)
        df = pd.DataFrame(X, columns=["Feature_1", "Feature_2", "Feature_3"])
        df["Target"] = y
        
    else:
        uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            
    if df is not None:
        with st.expander("Preview Dataset"):
            st.dataframe(df.head(), use_container_width=True)
            
        # --- 2. FEATURE SELECTION ---
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<h4>⚙️ Feature Selection</h4>", unsafe_allow_html=True)
        all_cols = df.columns.tolist()
        
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            target_col = st.selectbox("Target Column", all_cols, index=len(all_cols)-1)
        with col_f2:
            feature_cols = st.multiselect("Features", [c for c in all_cols if c != target_col], default=[c for c in all_cols if c != target_col][:3])
            
        if len(feature_cols) == 0:
            st.warning("Please select at least one feature.")
            return

        # --- 3. TABS LAYOUT ---
        st.markdown("<br/>", unsafe_allow_html=True)
        tabs = st.tabs(["📊 Heatmap", "📈 Pairplot", "⚙️ Activations", "🧬 Neural Net", "🔷 Decision Boundary", "⚖️ Weights", "🌐 3D Vis"])
        
        # TAB 1: HEATMAP
        with tabs[0]:
            st.markdown("<h4>Feature Correlation Heatmap</h4>", unsafe_allow_html=True)
            fig, ax = plt.subplots(facecolor='none')
            ax.set_facecolor('none')
            corr = df[feature_cols + [target_col]].corr()
            cmap = sns.diverging_palette(250, 170, as_cmap=True, center="light")
            sns.heatmap(corr, ax=ax, cmap=cmap, annot=True, fmt=".2f", linewidths=0.5, linecolor='#cbd5e1', square=True)
            ax.tick_params(colors='#475569')
            cbar = ax.collections[0].colorbar
            if cbar:
                cbar.ax.yaxis.set_tick_params(colors='#475569')
            st.pyplot(fig)

        # TAB 2: PAIRPLOT
        with tabs[1]:
            st.markdown("<h4>Pairwise Relationships</h4>", unsafe_allow_html=True)
            if st.checkbox("Generate Pairplot (May take a moment)"):
                with st.spinner("Generating..."):
                    plot_df = df[feature_cols + [target_col]].copy()
                    fig = sns.pairplot(plot_df, hue=target_col, palette="husl", plot_kws={'alpha': 0.6})
                    fig.fig.patch.set_facecolor('none')
                    st.pyplot(fig.fig)
            else:
                st.info("Check the box above to generate the pairplot.")

        # TAB 3: ACTIVATIONS
        with tabs[2]:
            st.markdown("<h4>Activation Functions</h4>", unsafe_allow_html=True)
            act_func = st.selectbox("Select Function", ["Sigmoid", "Tanh", "ReLU"])
            x_val = np.linspace(-10, 10, 100)
            if act_func == "Sigmoid":
                y_val = sigmoid(x_val)
            elif act_func == "Tanh":
                y_val = tanh(x_val)
            else:
                y_val = np.maximum(0, x_val)
                
            fig, ax = plt.subplots(facecolor='none')
            ax.set_facecolor('none')
            ax.plot(x_val, y_val, color="#6C63FF", linewidth=2.5)
            ax.set_title(f"{act_func} Activation", color="#0f172a")
            ax.grid(True, linestyle='--', alpha=0.1, color='#000000')
            ax.tick_params(colors='#475569')
            for spine in ax.spines.values():
                spine.set_color('#cbd5e1')
            st.pyplot(fig)

        # TAB 4: NEURAL NETWORK
        # TAB 4: NEURAL NETWORK
        with tabs[3]:
            st.markdown("<h4>Network Architecture</h4>", unsafe_allow_html=True)
            col_n1, col_n2, col_n3 = st.columns(3)
            n_in = col_n1.number_input("Input Neurons", 1, 10, len(feature_cols))
            n_hid = col_n2.number_input("Hidden Neurons", 1, 15, 5)
            n_out = col_n3.number_input("Output Neurons", 1, 5, 1)
            
            if st.button("Generate Architecture Graph"):
                layers = [n_in, n_hid, n_out]
                node_x, node_y = [], []
                node_colors = []
                node_texts = []
                
                layer_x = [0, 1, 2]
                layer_names = ["Input Layer", "Hidden Layer", "Output Layer"]
                colors = ['#00C9A7', '#6C63FF', '#f43f5e']
                
                pos = {}
                node_idx = 0
                max_size = max(layers)
                
                for i, layer_size in enumerate(layers):
                    for j in range(layer_size):
                        x = layer_x[i]
                        y = j - (layer_size - 1) / 2.0
                        
                        pos[node_idx] = (x, y)
                        node_x.append(x)
                        node_y.append(y)
                        node_colors.append(colors[i])
                        node_texts.append(f"<b>{layer_names[i]}</b><br>Neuron {j+1}")
                        node_idx += 1
                        
                pos_edge_x, pos_edge_y = [], []
                neg_edge_x, neg_edge_y = [], []
                mid_x, mid_y, mid_texts = [], [], []
                idx1 = 0
                idx2 = n_in
                
                np.random.seed(42) # For consistent weights visually
                for i in range(len(layers)-1):
                    for u in range(idx1, idx1+layers[i]):
                        for v in range(idx2, idx2+layers[i+1]):
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            
                            weight = np.random.randn()
                            if weight >= 0:
                                pos_edge_x.extend([x0, x1, None])
                                pos_edge_y.extend([y0, y1, None])
                            else:
                                neg_edge_x.extend([x0, x1, None])
                                neg_edge_y.extend([y0, y1, None])
                            
                            # Add midpoint for hovering over the weight
                            mid_x.append((x0+x1)/2)
                            mid_y.append((y0+y1)/2)
                            mid_texts.append(f"Weight: {weight:.3f}")
                            
                    idx1 += layers[i]
                    idx2 += layers[i+1]
                    
                fig = go.Figure()
                
                # Positive Edges (Excitatory) - Teal
                fig.add_trace(go.Scatter(
                    x=pos_edge_x, y=pos_edge_y,
                    mode='lines',
                    line=dict(color='rgba(0, 201, 167, 0.4)', width=2),
                    name='Positive Weight',
                    hoverinfo='none'
                ))
                
                # Negative Edges (Inhibitory) - Rose
                fig.add_trace(go.Scatter(
                    x=neg_edge_x, y=neg_edge_y,
                    mode='lines',
                    line=dict(color='rgba(244, 63, 94, 0.4)', width=2),
                    name='Negative Weight',
                    hoverinfo='none'
                ))
                
                # Edge weights (invisible markers to trigger hover)
                fig.add_trace(go.Scatter(
                    x=mid_x, y=mid_y,
                    mode='markers',
                    marker=dict(size=8, color='rgba(30, 41, 59, 0.8)'), # small visible dot
                    text=mid_texts,
                    hoverinfo='text',
                    showlegend=False
                ))
                
                # Nodes
                fig.add_trace(go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    marker=dict(size=30, color=node_colors, line=dict(color='white', width=3)),
                    text=node_texts,
                    hoverinfo='text',
                    showlegend=False
                ))
                
                # Layer Labels as Annotations
                annotations = []
                for i in range(len(layers)):
                    annotations.append(
                        dict(
                            x=i, y=max_size/2.0 + 0.5,
                            text=f"<b>{layer_names[i]}</b>",
                            showarrow=False,
                            font=dict(size=16, color='#1e293b'),
                            xanchor='center', yanchor='bottom'
                        )
                    )
                
                fig.update_layout(
                    showlegend=True,
                    legend=dict(yanchor="top", y=1.0, xanchor="left", x=0.0),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 2.5]),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-max_size/2.0 - 0.5, max_size/2.0 + 1.5]),
                    margin=dict(l=20, r=20, b=20, t=60),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    annotations=annotations,
                    hovermode='closest',
                    font=dict(color='#1e293b')
                )
                
                st.plotly_chart(fig, use_container_width=True)

        # TAB 5: DECISION BOUNDARY
        with tabs[4]:
            st.markdown("<h4>Decision Boundary (2D)</h4>", unsafe_allow_html=True)
            if len(feature_cols) == 2:
                fig, ax = plt.subplots(facecolor='none')
                ax.set_facecolor('none')
                x_feat = df[feature_cols[0]]
                y_feat = df[feature_cols[1]]
                target = df[target_col]
                
                scatter = ax.scatter(x_feat, y_feat, c=target, cmap='coolwarm', edgecolors='k', alpha=0.7)
                ax.set_xlabel(feature_cols[0], color="#475569")
                ax.set_ylabel(feature_cols[1], color="#475569")
                ax.tick_params(colors='#475569')
                for spine in ax.spines.values():
                    spine.set_color('#cbd5e1')
                    
                # Mock decision boundary (linear)
                x_min, x_max = x_feat.min() - 1, x_feat.max() + 1
                y_min, y_max = y_feat.min() - 1, y_feat.max() + 1
                xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))
                # Simple mock boundary: w1*x1 + w2*x2 = 0
                Z = (xx - x_feat.mean()) + (yy - y_feat.mean()) > 0
                ax.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
                
                st.pyplot(fig)
            else:
                st.warning("Please select exactly 2 features to view a 2D decision boundary.")

        # TAB 6: WEIGHTS
        with tabs[5]:
            st.markdown("<h4>Simulated Weight Distribution</h4>", unsafe_allow_html=True)
            dist_type = st.selectbox("Initialization Method", ["Normal (Gaussian)", "Uniform", "Xavier/Glorot"])
            n_weights = st.slider("Number of Weights", 100, 10000, 1000)
            
            if dist_type == "Normal (Gaussian)":
                w = np.random.randn(n_weights)
            elif dist_type == "Uniform":
                w = np.random.uniform(-1, 1, n_weights)
            else:
                # Xavier
                variance = 2.0 / (len(feature_cols) + 1)
                w = np.random.randn(n_weights) * np.sqrt(variance)
                
            fig, ax = plt.subplots(facecolor='none')
            ax.set_facecolor('none')
            sns.histplot(w, kde=True, color="#00C9A7", ax=ax)
            ax.tick_params(colors='#475569')
            for spine in ax.spines.values():
                spine.set_color('#cbd5e1')
            st.pyplot(fig)

        # TAB 7: 3D VISUALIZATION
        with tabs[6]:
            st.markdown("<h4>🌐 Interactive 3D Scatter</h4>", unsafe_allow_html=True)
            if len(feature_cols) >= 3:
                fig_3d = px.scatter_3d(
                    df, x=feature_cols[0], y=feature_cols[1], z=feature_cols[2],
                    color=target_col, color_continuous_scale="Viridis",
                    opacity=0.8, template="plotly_white"
                )
                fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), font=dict(color='#1e293b'))
                st.plotly_chart(fig_3d, use_container_width=True)
                
                st.markdown("<h4>📈 3D Surface Plot (Loss Landscape Simulation)</h4>", unsafe_allow_html=True)
                # Simulated complex surface
                X_surf = np.linspace(-5, 5, 50)
                Y_surf = np.linspace(-5, 5, 50)
                X_surf, Y_surf = np.meshgrid(X_surf, Y_surf)
                Z_surf = np.sin(np.sqrt(X_surf**2 + Y_surf**2)) + np.random.normal(0, 0.1, X_surf.shape)
                
                fig_surf = go.Figure(data=[go.Surface(z=Z_surf, x=X_surf, y=Y_surf, colorscale='Viridis')])
                fig_surf.update_layout(margin=dict(l=0, r=0, b=0, t=0), template="plotly_white", font=dict(color='#1e293b'))
                st.plotly_chart(fig_surf, use_container_width=True)
                
            else:
                st.warning("Please select at least 3 features to view 3D Visualizations.")
    else:
        st.info("Please select or generate a dataset to begin.")