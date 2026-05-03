import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
import cv2
from core.ai_helper import render_section_ai

def create_dummy_data():
    # 100 random samples, 64x64, grayscale for quick testing
    X_dummy = np.random.randn(100, 64, 64, 1)
    # Simple arbitrary threshold to create binary classes
    y_dummy = (X_dummy.sum(axis=(1,2,3)) > 0).astype(int)
    return X_dummy, y_dummy

def get_cnn_model():
    inputs = Input(shape=(64, 64, 1))
    x = Conv2D(8, (3,3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(16, (3,3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def show():
    st.markdown("<h2 class='gradient-text'>👁️ CNN Vision Lab</h2>", unsafe_allow_html=True)
    st.markdown("<p>Explore how CNNs process images, from manual edge detection to learned feature extraction.</p>", unsafe_allow_html=True)
    
    # Session state
    if "cnn_model" not in st.session_state:
        st.session_state.cnn_model = None
    if "cnn_history" not in st.session_state:
        st.session_state.cnn_history = None
    if "cnn_image" not in st.session_state:
        st.session_state.cnn_image = None
    if "cnn_image_disp" not in st.session_state:
        st.session_state.cnn_image_disp = None
        
    st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)
    tabs = st.tabs(["📸 Input", "🔍 Edge Detection", "📊 Training", "🧩 Feature Maps", "🔍 Filters", "📈 Prediction"])
    
    # TAB 1: INPUT
    with tabs[0]:
        st.markdown("<h4>Upload & Process Image</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an Image (PNG, JPG)", type=["png", "jpg", "jpeg"], key="cnn_uploader")
        
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file).convert('L') # Convert to grayscale
                st.session_state.cnn_image_disp = img
                
                # Preprocess
                img_resized = img.resize((64, 64))
                img_array = np.array(img_resized) / 255.0
                st.session_state.cnn_image = img_array.reshape(1, 64, 64, 1)
                
                st.success("Image successfully uploaded and processed to 64x64 Grayscale!")
            except Exception as e:
                st.error(f"Error processing image: {e}")
                
        if st.session_state.cnn_image_disp is not None:
            col1, col2 = st.columns(2)
            col1.image(st.session_state.cnn_image_disp, caption="Original Image (Grayscale)", use_container_width=True)
            img_res = np.squeeze(st.session_state.cnn_image) * 255.0
            col2.image(img_res.astype(np.uint8), caption="Processed Model Input (64x64)", use_container_width=True)
        else:
            st.info("Upload an image to start visually exploring convolutional processing.")

    # TAB 2: EDGE DETECTION (NEW FEATURE)
    with tabs[1]:
        st.markdown("<h4>Edge Detection (Manual Feature Extraction)</h4>", unsafe_allow_html=True)
        st.markdown("""
        Before a CNN learns to 'see' automatically, computer vision used manual techniques like **Canny Edge Detection**. 
        This highlights boundaries and shapes, which is exactly what the first layers of a CNN try to do using filters.
        """)
        
        edge_file = st.file_uploader("Upload Image for Edge Detection", type=["png", "jpg", "jpeg"], key="edge_uploader")
        
        target_img = None
        if edge_file:
            target_img = Image.open(edge_file)
        elif st.session_state.cnn_image_disp:
            target_img = st.session_state.cnn_image_disp
            st.info("Using image from Input tab.")
            
        if target_img:
            # Controls
            col_c1, col_c2 = st.columns(2)
            t1 = col_c1.slider("Threshold 1 (Min)", 0, 255, 100)
            t2 = col_c2.slider("Threshold 2 (Max)", 0, 255, 200)
            
            # Processing
            img_cv = np.array(target_img.convert('RGB'))
            gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, t1, t2)
            
            # Display
            st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
            v_col1, v_col2, v_col3 = st.columns(3)
            v_col1.image(img_cv, caption="Original", use_container_width=True)
            v_col2.image(gray, caption="Grayscale", use_container_width=True)
            v_col3.image(edges, caption="Canny Edges", use_container_width=True)
            
            st.markdown("""
            <div class='ai-response-box'>
                <b>💡 How this connects to CNNs:</b><br>
                Notice how the Canny output shows only the outlines. In a CNN, the <b>Filters</b> you see in the next tabs perform a similar mathematical operation (convolution) to detect these same edges automatically!
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Please upload an image here or in the Input tab to see Edge Detection in action.")

    # TAB 3: TRAINING
    with tabs[2]:
        st.markdown("<h4>Train CNN Model</h4>", unsafe_allow_html=True)
        st.markdown("<p style='color:#64748b;'>We simulate training here on a lightweight dummy dataset (64x64 image shapes) to allow for rapid, real-time experimentation without freezing the dashboard.</p>", unsafe_allow_html=True)
        
        epochs = st.slider("Epochs", 1, 20, 5)
        
        if st.button("🚀 Train CNN", type="primary"):
            with st.spinner("Initializing CNN Architecture and Training..."):
                model = get_cnn_model()
                X_train, y_train = create_dummy_data()
                history = model.fit(X_train, y_train, epochs=epochs, verbose=0, validation_split=0.2)
                st.session_state.cnn_model = model
                st.session_state.cnn_history = history.history
                
        if st.session_state.cnn_history is not None:
            hist = st.session_state.cnn_history
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hist['loss'], mode='lines', name='Training Loss', line=dict(color='#6C63FF', width=3)))
            if 'val_loss' in hist:
                fig.add_trace(go.Scatter(y=hist['val_loss'], mode='lines', name='Validation Loss', line=dict(color='#00C9A7', width=3)))
            
            fig.update_layout(
                xaxis_title="Epochs", yaxis_title="Binary Crossentropy Loss",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, b=0, t=20),
                font=dict(color='#1e293b'), hovermode='x unified',
                legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.05)')
            st.plotly_chart(fig, use_container_width=True)

    # TAB 4: FEATURE MAPS
    with tabs[3]:
        st.markdown("<h4>Convolutional Feature Maps</h4>", unsafe_allow_html=True)
        st.write("Feature maps represent the intermediate 'vision' of the network after applying its learned filters.")
        if st.session_state.cnn_model is None:
            st.warning("Please train the model first!")
        elif st.session_state.cnn_image is None:
            st.warning("Please upload an image in the Input tab!")
        else:
            model = st.session_state.cnn_model
            # Get the first Conv2D layer output
            layer_outputs = [layer.output for layer in model.layers if isinstance(layer, Conv2D)]
            if len(layer_outputs) > 0:
                activation_model = Model(inputs=model.input, outputs=layer_outputs[0])
                activations = activation_model.predict(st.session_state.cnn_image)
                
                st.markdown("<h5>First Conv2D Layer Activations</h5>", unsafe_allow_html=True)
                # Plot 8 feature maps
                fig, axes = plt.subplots(2, 4, figsize=(10, 5), facecolor='none')
                fig.patch.set_facecolor('none')
                for i, ax in enumerate(axes.flat):
                    if i < activations.shape[-1]:
                        ax.imshow(activations[0, :, :, i], cmap='viridis')
                        ax.axis('off')
                        ax.set_title(f"Map {i+1}", color='#1e293b', fontsize=10)
                st.pyplot(fig)
            else:
                st.error("No Convolutional layers found in model.")

    # TAB 5: FILTERS
    with tabs[4]:
        st.markdown("<h4>Learned Conv2D Filters</h4>", unsafe_allow_html=True)
        st.write("These are the literal 3x3 weight matrices the network has learned to detect patterns.")
        
        if st.session_state.cnn_model is None:
            st.warning("Please train the model first!")
        else:
            model = st.session_state.cnn_model
            conv_layers = [layer for layer in model.layers if isinstance(layer, Conv2D)]
            if len(conv_layers) > 0:
                filters, biases = conv_layers[0].get_weights()
                # Normalize filters for visualization
                f_min, f_max = filters.min(), filters.max()
                filters = (filters - f_min) / (f_max - f_min + 1e-8)
                
                fig, axes = plt.subplots(2, 4, figsize=(8, 4), facecolor='none')
                fig.patch.set_facecolor('none')
                for i, ax in enumerate(axes.flat):
                    if i < filters.shape[-1]:
                        f = filters[:, :, 0, i]
                        ax.imshow(f, cmap='gray')
                        ax.axis('off')
                        ax.set_title(f"Filter {i+1}", color='#1e293b', fontsize=10)
                st.pyplot(fig)
            else:
                st.error("No Convolutional layers found in model.")

    # TAB 6: PREDICTION
    with tabs[5]:
        st.markdown("<h4>Model Prediction</h4>", unsafe_allow_html=True)
        if st.session_state.cnn_model is None or st.session_state.cnn_image is None:
            st.warning("Please ensure you have uploaded an image and trained the model.")
        else:
            if st.button("🔍 Run Inference", type="primary"):
                with st.spinner("Processing image through CNN..."):
                    pred = st.session_state.cnn_model.predict(st.session_state.cnn_image)[0][0]
                    confidence = pred if pred > 0.5 else 1 - pred
                    predicted_class = "Class 1" if pred > 0.5 else "Class 0"
                    
                    st.markdown("<div style='display:flex; justify-content:center; margin-top:20px;'>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='background:#ffffff; border:1px solid #e2e8f0; padding:30px 50px; border-radius:16px; text-align:center; box-shadow: 0 10px 30px rgba(0,0,0,0.04);'>
                        <h3 style='color:#475569; margin-bottom:15px;'>Classification Result</h3>
                        <h1 style='color:#6C63FF; margin:0; font-size:2.5rem;'>{predicted_class}</h1>
                        <p style='color:#64748b; margin-top:15px; font-size:1.2rem;'>Confidence: <b style='color:#0f172a;'>{confidence*100:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
    render_section_ai("CNN lab: Convolutional Neural Networks, OpenCV edge detection, filters, feature maps, vision.")

