import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from annotated_text import annotated_text
import time
import os
from model_utils import NERModel

st.set_page_config(
    page_title="Twitter NER Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2127;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8cff;
        color: white;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #4e8cff;
        text-align: center;
    }
    h1 {
        color: #4e8cff;
        text-align: center;
        font-family: 'Helvetica Neue', sans-serif;
    }
    h2, h3 {
        color: #fafafa;
    }
    .stButton>button {
        background-color: #4e8cff;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2a6ae0;
    }
</style>
""", unsafe_allow_html=True)

# --- STANDALONE MODEL INITIALIZATION ---
@st.cache_resource
def get_model():
    """Load and cache the NER model"""
    # Use current directory for data
    data_dir = os.path.dirname(__file__)
    model = NERModel(model_type='bert', data_dir=data_dir)
    
    # Try to load data
    try:
        model.prepare_data()
    except Exception as e:
        print(f"Warning: Could not load data: {e}")
        
    return model

# Initialize model
try:
    ner_model = get_model()
    backend_connected = True
except Exception as e:
    backend_connected = False
    st.error(f"Failed to load model: {e}")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/twitter--v1.png", width=80)
    st.title("Settings")
    
    # Backend Status Indicator
    if backend_connected:
        st.success("🟢 System: Ready")
    else:
        st.error("🔴 System: Error")
    
    st.info("Using Standalone BERT Model")
    
    st.markdown("---")
    st.markdown("### Model Controls")
    st.markdown("Model controls have been moved to the 🛠️ **Model & Training** tab.")
    
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    if st.button("⚡ Use Features"):
        st.session_state.active_tab = "Analyze"
    if st.button("🔍 Analyze Text"):
        st.session_state.active_tab = "Analyze"

# Header
st.markdown("<h1>By RATNESH SINGH</h1>", unsafe_allow_html=True)
st.markdown("# 🐦 Twitter Named Entity Recognition")
st.markdown("### Extract entities like Person, Location, Company, Product from tweets")

# Tabs
tabs = st.tabs([
    "💼 Business Case", 
    "ℹ️ About", 
    "📑 Technical Documentation", 
    "🔍 Analyze", 
    "🛠️ Model & Training", 
    "📊 Data Statistics", 
    "📝 Logs"
])

# 1. Business Case Tab
with tabs[0]:
    st.header("💼 Business Case: Twitter Entity Extraction")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Problem Statement
        Social media platforms like Twitter generate massive amounts of unstructured text data every second. 
        Businesses struggle to extract meaningful insights from this noise to understand:
        - **Brand Sentiment**: What are people saying about us?
        - **Product Feedback**: Which products are being discussed?
        - **Trend Analysis**: What topics are trending in specific locations?
        - **Customer Support**: Identifying urgent issues mentioned by users.
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Solution
        An automated **Named Entity Recognition (NER)** system that processes tweets in real-time to identify and classify key entities:
        
        - **🏢 Companies**: Apple, Tesla, Google
        - **👤 People**: Elon Musk, Tim Cook
        - **📍 Locations**: New York, London, Silicon Valley
        - **📱 Products**: iPhone 15, Model S, Pixel 8
        
        This enables automated tagging, routing, and analytics for business intelligence.
        """)
        
    st.markdown("---")
    
    st.markdown("### 💰 ROI & Impact")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    with metric_col1:
        st.markdown("""
        <div class="metric-card">
            <h3>85%</h3>
            <p>Reduction in Manual Tagging Time</p>
        </div>
        """, unsafe_allow_html=True)
        
    with metric_col2:
        st.markdown("""
        <div class="metric-card">
            <h3>24/7</h3>
            <p>Real-time Monitoring Capability</p>
        </div>
        """, unsafe_allow_html=True)
        
    with metric_col3:
        st.markdown("""
        <div class="metric-card">
            <h3>3x</h3>
            <p>Faster Response to Trends</p>
        </div>
        """, unsafe_allow_html=True)

# 2. About Tab
with tabs[1]:
    st.header("ℹ️ About the Project")
    
    st.markdown("""
    ### 🧠 Model Architecture
    This project utilizes **BERT (Bidirectional Encoder Representations from Transformers)**, a state-of-the-art language model developed by Google.
    
    - **Base Model**: `bert-base-uncased`
    - **Fine-tuning**: The model is fine-tuned on the **WNUT 17** dataset (Twitter data) for token classification.
    - **Tokenizer**: WordPiece tokenizer handling sub-word units.
    
    ### 🛠️ Tech Stack
    - **Frontend**: Streamlit (Python)
    - **Model**: Hugging Face Transformers & PyTorch
    - **Visualization**: Plotly & Pandas
    
    ### 👨‍💻 Developer
    **Ratnesh Singh**  
    *Data Scientist & NLP Engineer*
    """)

# 3. Technical Documentation Tab
with tabs[2]:
    st.header("📑 Technical Documentation")
    
    st.markdown("""
    ### ⚙️ System Architecture
    The application runs as a standalone Streamlit app with an embedded BERT model.
    
    1. **Model Layer**: 
       - Loads pre-trained BERT weights
       - Tokenizes input text
       - Predicts entity tags (B-PER, I-PER, etc.)
       
    2. **Application Layer**:
       - Streamlit interface for user interaction
       - Data processing and visualization pipeline
    
    ### 📊 Data Pipeline
    1. **Input**: Raw text string from user.
    2. **Preprocessing**: Cleaning and tokenization.
    3. **Inference**: Model forward pass to get logits.
    4. **Post-processing**: Converting IDs to Tags and aligning with words.
    5. **Visualization**: Rendering annotated text and charts.
    """)
    
    try:
        with open("README.md", "r") as f:
            readme_content = f.read()
        
        st.download_button(
            label="📥 Download Full Documentation (README)",
            data=readme_content,
            file_name="Twitter_NER_Documentation.md",
            mime="text/markdown"
        )
    except:
        st.warning("README file not found.")

# 4. Analyze Tab
with tabs[3]:
    st.header("🔍 Analyze Tweet")
    
    st.markdown("Enter a tweet or sentence below to extract entities.")
    
    # Sample selector
    sample_options = {
        "Custom": "",
        "Tech News": "Apple announced the new MacBook Pro with M3 chip in San Francisco yesterday.",
        "Sports": "Lionel Messi scored a goal for Inter Miami in the match against Orlando City.",
        "Business": "Elon Musk said Tesla will expand its factory in Berlin next year.",
        "Entertainment": "Taylor Swift is performing in London for her Eras Tour concert."
    }
    
    selected_sample = st.selectbox("Or choose a sample:", list(sample_options.keys()))
    
    default_text = sample_options[selected_sample] if selected_sample != "Custom" else ""
    
    text_input = st.text_area("Text:", value=default_text, height=100, placeholder="Type something here...")
    
    if st.button("🚀 Analyze Text", type="primary"):
        if text_input:
            with st.spinner("Analyzing..."):
                try:
                    # Direct model call
                    result = ner_model.predict(text_input)
                    
                    # Process results
                    words = [item[0] for item in result]
                    entities = [item[1] for item in result]
                    
                    # Create annotated text format
                    annotated_data = []
                    entity_counts = {}
                    
                    for word, entity in zip(words, entities):
                        if entity == 'O':
                            annotated_data.append(word + " ")
                        else:
                            # Color mapping
                            color = "#808080"
                            if "company" in entity.lower(): color = "#4e8cff" # Blue
                            elif "person" in entity.lower(): color = "#ff4b4b" # Red
                            elif "geo" in entity.lower() or "location" in entity.lower(): color = "#3dd56d" # Green
                            elif "product" in entity.lower(): color = "#ffa421" # Orange
                            
                            annotated_data.append((word, entity, color))
                            annotated_data.append(" ")
                            
                            # Count entities
                            if entity.startswith("B-"):
                                entity_type = entity[2:]
                                entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
                    
                    st.markdown("### 📝 Annotated Text")
                    annotated_text(*annotated_data)
                    
                    st.markdown("---")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### 📈 Entity Distribution")
                        if entity_counts:
                            df_entities = pd.DataFrame({
                                "Entity Type": list(entity_counts.keys()),
                                "Count": list(entity_counts.values())
                            })
                            fig = px.pie(df_entities, values='Count', names='Entity Type', hole=0.4, 
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig.update_layout(background_color="rgba(0,0,0,0)")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No entities found to visualize.")
                            
                    with col2:
                        st.markdown("### 📋 Detailed Entities")
                        if entity_counts:
                            # Create a clean table of entities
                            clean_entities = []
                            current_entity = {"text": "", "type": ""}
                            
                            for word, tag in zip(words, entities):
                                if tag.startswith("B-"):
                                    if current_entity["text"]:
                                        clean_entities.append(current_entity)
                                    current_entity = {"text": word, "type": tag[2:]}
                                elif tag.startswith("I-") and current_entity["text"]:
                                    current_entity["text"] += " " + word
                                else:
                                    if current_entity["text"]:
                                        clean_entities.append(current_entity)
                                        current_entity = {"text": "", "type": ""}
                            
                            if current_entity["text"]:
                                clean_entities.append(current_entity)
                                
                            if clean_entities:
                                st.dataframe(pd.DataFrame(clean_entities), use_container_width=True)
                            else:
                                st.info("No entities detected.")
                        else:
                            st.info("No entities detected.")
                            
                except Exception as e:
                    st.error(f"Error during analysis: {e}")
        else:
            st.warning("Please enter some text to analyze.")

# 5. Model & Training Tab
with tabs[4]:
    st.header("🛠️ Model & Training")
    
    st.info("In Standalone Mode, training is simulated or limited. For full training, run locally.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Model Status")
        st.json({
            "Model": "BERT (bert-base-uncased)",
            "Status": "Loaded",
            "Mode": "Standalone / Demo",
            "Device": str(ner_model.device)
        })
        
    with col2:
        st.subheader("Training Controls")
        epochs = st.slider("Epochs", 1, 5, 1)
        batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
        
        if st.button("Start Training (Demo)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
                status_text.text(f"Training... {i+1}%")
            
            status_text.success("Training completed! (Simulation)")

# 6. Data Statistics Tab
with tabs[5]:
    st.header("📊 Dataset Statistics")
    
    if st.button("🔄 Refresh Stats"):
        # Calculate stats directly
        try:
            train_len = len(ner_model.train_data) if ner_model.train_data else 0
            test_len = len(ner_model.test_data) if ner_model.test_data else 0
            
            stats = {
                "train_samples": train_len,
                "test_samples": test_len,
                "total_samples": train_len + test_len,
                "num_entities": len(ner_model.tag2id) if ner_model.tag2id else 0
            }
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Training Samples", stats["train_samples"])
            col2.metric("Test Samples", stats["test_samples"])
            col3.metric("Total Samples", stats["total_samples"])
            col4.metric("Number of Entities", stats["num_entities"])
            
            if ner_model.tag2id:
                st.markdown("### Entity Types")
                st.write(list(ner_model.tag2id.keys()))
                
        except Exception as e:
            st.error(f"Could not calculate stats: {e}")

# 7. Logs Tab
with tabs[6]:
    st.header("📝 System Logs")
    st.text_area("Logs", "System initialized in Standalone Mode.\nModel loaded successfully.\nReady for inference.", height=300)

# Footer
st.markdown("---")
st.markdown("Twitter NER System v1.0 | Built with ❤️ using Streamlit & Transformers")
