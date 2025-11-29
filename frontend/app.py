import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from annotated_text import annotated_text
import time

st.set_page_config(
    page_title="Twitter NER Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = "http://localhost:8000"

# Cached API call functions to improve performance
@st.cache_data(ttl=30)
def check_backend_health():
    """Check if backend is available"""
    try:
        response = requests.get(f"{API_URL}/", timeout=2)
        return response.status_code == 200
    except:
        return False

@st.cache_data(ttl=60)
def get_data_stats():
    """Fetch data statistics with caching"""
    try:
        response = requests.get(f"{API_URL}/data-stats", timeout=30)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def log_event(message, level="INFO"):
    """Send a log message to the backend"""
    try:
        requests.post(
            f"{API_URL}/log-client-event",
            json={"message": message, "level": level},
            timeout=1
        )
    except:
        pass  # Fail silently if logging fails

# Custom CSS
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1DA1F2, #9b5de5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Card/Box Styling */
    .css-1r6slb0, .stMarkdown, .stDataFrame {
        border-radius: 10px;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #21262d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #1DA1F2;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #21262d;
        border-radius: 5px;
        color: #c9d1d9;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1DA1F2;
        color: white;
    }
    
    /* Entity Legend Styling */
    .entity-legend {
        background: linear-gradient(135deg, #21262d, #161b22);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(90deg, #1DA1F2, #0d8bd9);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #0d8bd9, #1DA1F2);
        box-shadow: 0 0 10px rgba(29, 161, 242, 0.5);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #21262d;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('''
<div style="text-align: center; margin-bottom: 20px;">
    <h2 style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 48px;
        font-weight: 900;
        margin: 0;
        padding: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    ">By RATNESH SINGH</h2>
</div>
''', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🐦 Twitter Named Entity Recognition</h1>', unsafe_allow_html=True)
st.markdown("### Extract entities like Person, Location, Company, Product from tweets")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Backend Status Indicator
    backend_healthy = check_backend_health()
    if backend_healthy:
        st.success("🟢 Backend: Connected")
    else:
        st.error("🔴 Backend: Disconnected")
        st.caption("Start backend: `python -m uvicorn main:app --port 8000`")
    
    st.info("Using BERT (Transformer) Model")
    
    st.markdown("---")
    st.write("Model controls have been moved to the **🛠️ Model & Training** tab.")
    
    st.markdown("---")
    
    st.subheader("⚡ Use Features")
    
    st.markdown("""
    **🔍 Analyze Text**
    *   Extract entities from any text.
    *   Visualize entity distribution.
    
    **📊 Data Statistics**
    *   View dataset insights.
    *   Check tag distribution.
    
    **🛠️ Model & Training**
    *   Fine-tune the BERT model.
    *   Read technical documentation.
    
    **📝 Logs**
    *   Monitor system activity.
    """)

# Main content
tab_biz, tab_about, tab_tech_doc, tab_analyze, tab_model, tab_stats, tab_logs = st.tabs([
    "💼 Business Case", 
    "ℹ️ About", 
    "📚 Technical Documentation",
    "🔍 Analyze", 
    "🛠️ Model & Training",
    "📊 Data Statistics", 
    "📝 Logs"
])

with tab_analyze:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Enter Text to Analyze")
        
        # Sample texts
        sample_texts = {
            "Product": "Apple MacBook Pro is the best laptop in the world",
            "Location": "I visited New York City and saw the Empire State Building",
            "Person": "Elon Musk announced Tesla's new electric car model",
            "Mixed": "Google CEO Sundar Pichai spoke at the conference in San Francisco about Android updates"
        }
        
        selected_sample = st.selectbox("Or choose a sample:", ["Custom"] + list(sample_texts.keys()))
        
        if selected_sample == "Custom":
            text_input = st.text_area(
                "Text:",
                "Apple MacBook Pro is the best laptop in the world",
                height=150
            )
        else:
            text_input = st.text_area(
                "Text:",
                sample_texts[selected_sample],
                height=150
            )
        
        analyze_button = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Entity Legend")
        st.markdown("""
        <div class="entity-legend">
        <b>B-</b>: Beginning of entity<br>
        <b>I-</b>: Inside entity<br>
        <b>O</b>: Outside entity<br><br>
        
        <b>Entity Types:</b><br>
        🟥 person - Person names<br>
        🟩 geo-loc - Locations<br>
        🟦 company - Companies<br>
        🟨 product - Products<br>
        🟪 facility - Facilities<br>
        🟦 musicartist - Music Artists<br>
        🟧 tvshow - TV Shows<br>
        🟪 sportsteam - Sports Teams<br>
        ⬜ other - Other entities
        </div>
        """, unsafe_allow_html=True)
                    
    if analyze_button and text_input:
        log_event(f"User requested analysis for text: {text_input[:50]}...")
        with st.spinner("Analyzing..."):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": text_input, "model_type": "bert"},
                    timeout=60
                )
                process_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # --- Metrics Row ---
                    st.markdown("### 📊 Analysis Results")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Processing Time", f"{process_time:.2f}s")
                    
                    # Count actual entities (excluding 'O' tag)
                    found_entities = [item for item in result['annotated'] if item['entity'] != 'O']
                    m2.metric("Entities Found", len(found_entities))
                    m3.metric("Word Count", len(result['words']))
                    
                    st.divider()
                    
                    # --- Annotated Text ---
                    st.subheader("📝 Annotated Text")
                    
                    # Prepare annotated text
                    annotated_data = []
                    for item in result['annotated']:
                        if item['entity'] != 'O':
                            annotated_data.append((item['word'], item['entity'], item['color']))
                        else:
                            annotated_data.append(item['word'] + " ")
                            
                    annotated_text(*annotated_data)
                    
                    st.divider()
                    
                    # --- Visualizations & Details ---
                    col_viz, col_details = st.columns([1, 1])
                    
                    with col_viz:
                        st.subheader("📈 Entity Distribution")
                        # Count entity types
                        entity_counts = {}
                        for item in result['annotated']:
                            if item['entity'] != 'O':
                                label = item['entity']
                                # Strip B- and I- for cleaner chart
                                clean_label = label.replace("B-", "").replace("I-", "")
                                entity_counts[clean_label] = entity_counts.get(clean_label, 0) + 1
                        
                        if entity_counts:
                            df_counts = pd.DataFrame(list(entity_counts.items()), columns=['Entity Type', 'Count'])
                            fig = px.pie(df_counts, values='Count', names='Entity Type', hole=0.4, 
                                         color_discrete_sequence=px.colors.qualitative.Pastel)
                            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No entities found to visualize.")

                    with col_details:
                        st.subheader("📋 Detailed Entities")
                        # Create a clean table
                        table_data = []
                        for item in result['annotated']:
                            if item['entity'] != 'O':
                                table_data.append({
                                    "Entity": item['word'],
                                    "Type": item['entity']
                                })
                        
                        if table_data:
                            st.dataframe(table_data, use_container_width=True)
                        else:
                            st.write("No entities detected.")

                else:
                    st.error(f"Error: {response.text}")
                    log_event(f"Analysis failed: {response.text}", "ERROR")
            except requests.exceptions.Timeout:
                st.error("Request timed out. The model might be loading. Please try again.")
                log_event("Analysis timed out", "WARNING")
            except Exception as e:
                st.error(f"Connection error: {e}. Is the backend running on port 8000?")
                log_event(f"Analysis connection error: {e}", "ERROR")

with tab_stats:
    st.subheader("📊 Dataset Statistics")
    
    with st.spinner("⏳ Loading dataset statistics... This may take a moment on first load."):
        stats = get_data_stats()
    
    if stats:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Samples", stats['train_samples'])
        with col2:
            st.metric("Test Samples", stats['test_samples'])
        with col3:
            st.metric("Total Samples", stats['total_samples'])
        with col4:
            st.metric("Number of Entities", stats['num_entities'])
        
        st.markdown("---")
        
        # Entity distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Entity Types")
            entities_df = pd.DataFrame({
                'Entity': stats['entities']
            })
            st.dataframe(entities_df, use_container_width=True)
        
        with col2:
            st.subheader("Entity Distribution in Dataset")
            
            # Create bar chart
            entity_dist = stats['entity_distribution']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(entity_dist.keys()),
                    y=list(entity_dist.values()),
                    marker_color='lightblue'
                )
            ])
            
            fig.update_layout(
                title="Entity Frequency",
                xaxis_title="Entity Type",
                yaxis_title="Count",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Additional stats
        st.subheader("Additional Information")
        st.write(f"**Max Sequence Length:** {stats['max_sequence_length']}")
    else:
        st.error("⚠️ Could not load statistics. The backend may still be initializing.")
        st.info("💡 **Tip**: The backend downloads the BERT model (~400MB) on first startup. This can take 2-5 minutes depending on your internet connection.")
        
        if st.button("🔄 Retry"):
            st.cache_data.clear()
            st.rerun()

with tab_logs:
    st.subheader("📝 API Logs")
    
    # Auto-fetch logs
    try:
        with st.spinner("⏳ Loading logs..."):
            response = requests.get(f"{API_URL}/logs?lines=1000", timeout=30)
        
        if response.status_code == 200:
            logs_data = response.json()
            logs = logs_data.get('logs', [])
            
            if logs:
                # Join logs and display in a code block for better readability
                log_content = "".join(logs)
                st.code(log_content, language="log")
                
                # Download button for logs
                st.download_button(
                    label="📥 Download Logs",
                    data=log_content,
                    file_name="ner_api.log",
                    mime="text/plain"
                )
            else:
                st.info("No logs available yet.")
        else:
            st.error(f"Failed to fetch logs: {response.text}")
    
    except requests.exceptions.Timeout:
        st.warning("⏳ Backend is still initializing. Logs will be available once the server is ready.")
        st.info("💡 **Tip**: The backend is downloading the BERT model (~400MB). This is a one-time operation that takes 2-5 minutes.")
        if st.button("🔄 Retry Logs"):
            st.rerun()
    except Exception as e:
        st.error(f"Could not load logs: {e}")
        st.caption("Ensure the backend server is running on port 8000.")

with tab_about:
    st.subheader("ℹ️ About This Application")
    
    st.markdown("""
    ### Twitter Named Entity Recognition System
    
    This application uses BERT (Bidirectional Encoder Representations from Transformers) to identify and classify named entities in Twitter text.
    
    #### Features:
    - **BERT Model**: State-of-the-art transformer architecture
    - **Real-time Analysis**: Instant entity extraction from text
    - **Model Training**: Train the model on your data directly from the UI
    - **Comprehensive Statistics**: View dataset and entity distribution
    - **Logging**: Monitor API activity and training progress
    
    #### Supported Entities:
    - **Person**: Names of people
    - **Geo-location**: Geographic locations
    - **Company**: Company names
    - **Product**: Product names
    - **Facility**: Facilities and buildings
    - **Music Artist**: Names of musicians
    - **TV Show**: TV show titles
    - **Sports Team**: Sports team names
    - **Other**: Other named entities
    
    #### Model:
    **BERT (bert-base-uncased)**
    - Transformer-based architecture
    - ~110M parameters
    - Pre-trained on large text corpus
    - Fine-tuned for NER task
    
    #### Dataset:
    - Training data: `wnut 16.txt.conll`
    - Test data: `wnut 16test.txt.conll`
    - Format: CoNLL (BIO tagging scheme)
    
    ---
    
    **Built with:** FastAPI, Streamlit, TensorFlow, Transformers
    """)

with tab_biz:
    st.subheader("💼 Business Case: Twitter NER")
    
    st.markdown("""
    ### 🎯 Objective
    **Implementing Named Entity Recognition (NER) for Tweet Analysis**
    
    Twitter aims to enhance the understanding of trends and topics on its platform by automatically identifying named entities in tweets. This goes beyond simple hashtag analysis to understand the *content* of conversations.
    
    ### 🏢 The Challenge
    - **Volume**: Twitter generates ~500 million tweets per day.
    - **Noise**: User-generated content is noisy, informal, and unstructured.
    - **Limitations**: Hashtags are often inconsistent, inaccurate, or absent.
    
    ### 💡 The Solution
    Develop an NER system to automatically extract and categorize entities such as:
    - **Persons** (e.g., "Elon Musk")
    - **Locations** (e.g., "New York")
    - **Organizations** (e.g., "Google")
    - **Products** (e.g., "iPhone")
    
    ### 🛠️ Technical Approach
    
    #### 1. Data Processing
    - **Format**: CoNLL (BIO Tagging Scheme)
    - **Entities**: 10+ fine-grained categories
    - **Preprocessing**: Tokenization, Padding, Label Encoding
    
    #### 2. Models Implemented
    - **BERT (Transformer)**: 
        - Used in this application for production-grade accuracy.
        - Leverages pre-trained contextual embeddings.
    - **LSTM + CRF**: 
        - Explored in the analysis phase (Jupyter Notebook).
        - Good for sequence modeling tasks.
    
    #### 3. Impact
    - **Trend Analysis**: Better understanding of "who" and "what" is trending.
    - **Content Recommendation**: Improved relevance for users.
    - **Ad Targeting**: More precise targeting based on entity interests.
    """)
    
    st.info("This application represents the production deployment of the solution.")

with tab_model:
    st.subheader("🛠️ Model Management & Training")
    
    st.markdown("### 🤖 Model Controls")
    
    model_options = {
        "BERT (bert-base-uncased)": "bert-base-uncased",
        "DistilBERT (distilbert-base-uncased)": "distilbert-base-uncased",
        "RoBERTa (roberta-base)": "roberta-base",
        "XLM-RoBERTa (xlm-roberta-base)": "xlm-roberta-base",
        "LSTM + CRF (Legacy)": "lstm"
    }
    
    selected_model_label = st.selectbox(
        "Choose a model for training/inference:",
        list(model_options.keys()),
        index=0,
        help="Select a Transformer model or the legacy LSTM approach."
    )
    
    selected_model_name = model_options[selected_model_label]
    
    if selected_model_name == "lstm":
        st.warning("⚠️ LSTM + CRF support is experimental. Transformer models are recommended.")
    else:
        st.success(f"✅ Selected **{selected_model_name}**. This model will be used for the next training session.")
    
    st.markdown("---")
    
    # --- Training Section ---
    st.write("#### 🎓 Train Model")
    st.info(f"Fine-tune **{selected_model_label}** on the CoNLL dataset.")
    st.caption("Training Data: `wnut 16.txt.conll` | Validation Data: `wnut 16test.txt.conll`")
    
    epochs = st.slider("Epochs", 1, 10, 3, key="tab6_epochs")
    batch_size = st.slider("Batch Size", 8, 64, 32, key="tab6_batch")
    
    if st.button("🚀 Start Training", type="primary", key="tab6_train"):
        log_event(f"User started training: Epochs={epochs}, Batch Size={batch_size}")
        with st.spinner("Starting training..."):
            try:
                response = requests.post(
                    f"{API_URL}/train",
                    json={"model_type": selected_model_name, "epochs": epochs, "batch_size": batch_size},
                    timeout=30
                )
                if response.status_code == 200:
                    st.success("Training started! Check status below.")
                else:
                    st.error(f"Training failed: {response.text}")
                    log_event(f"Training failed to start: {response.text}", "ERROR")
            except Exception as e:
                st.error(f"Connection error: {e}")
                log_event(f"Training connection error: {e}", "ERROR")
    
    # Training status
    try:
        response = requests.get(f"{API_URL}/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            if status['is_training']:
                st.info(f"🔄 {status['message']}")
                st.progress(status['progress'] / 100)
            else:
                st.success(f"✅ {status['message']}")
    except:
        pass

with tab_tech_doc:
    st.subheader("📚 Technical Documentation")
    st.caption("Source: `tweeter-ner-nlp.pdf`")
    
    # Navigation Menu
    doc_section = st.radio(
        "Go to section:",
        [
            "Problem Statement",
            "Data Description",
            "Process",
            "LSTM + CRF Model training",
            "Training our model",
            "Let's load the best model",
            "BERT Model",
            "Loading the tokenizer",
            "Comparision",
            "Questions"
        ],
        key="doc_nav"
    )
    
    st.markdown("---")
    
    if doc_section == "Problem Statement":
        st.markdown("""
        ### Twitter NER By Ratnesh
        
        #### 🎯 Objective
        **Implementing Named Entity Recognition (NER) for Tweet Analysis**
        
        Twitter aims to enhance the understanding of trends and topics on its platform by automatically identifying named entities in tweets. This goes beyond simple hashtag analysis to understand the *content* of conversations.
        
        #### 🏢 The Challenge
        - **Volume**: Twitter generates ~500 million tweets per day.
        - **Noise**: User-generated content is noisy, informal, and unstructured.
        - **Limitations**: Hashtags are often inconsistent, inaccurate, or absent.
        
        #### 💡 The Solution
        Develop an NER system to automatically extract and categorize entities such as:
        - **Persons** (e.g., "Elon Musk")
        - **Locations** (e.g., "New York")
        - **Organizations** (e.g., "Google")
        - **Products** (e.g., "iPhone")
        """)
        
    elif doc_section == "Data Description":
        st.markdown("""
        #### Data Description
        - **Dataset**: Annotated with 10 fine-grained NER categories.
        - **Format**: CoNLL (One word per line, separated by empty lines).
        - **Tags**: `person`, `geo-location`, `company`, `facility`, `product`, `music artist`, `movie`, `sports team`, `tv show`, `other`.
        - **Example**:
            ```
            Harry       B-PER
            Potter      I-PER
            was         O
            living      O
            in          O
            London      B-geo-loc
            ```
        """)
        
    elif doc_section == "Process":
        st.markdown("""
        #### Process Overview
        1.  **Import Data**: Load train/test datasets.
        2.  **Exploratory Analysis**: Check structure & characteristics.
        3.  **Preprocessing**: Prepare data for models.
        4.  **Modeling**: Train LSTM+CRF and BERT models.
        5.  **Evaluation**: Compute metrics and compare results.
        """)
        
    elif doc_section == "LSTM + CRF Model training":
        st.markdown("""
        #### LSTM + CRF Model Training
        - **Embeddings**: Initialize using Word2Vec.
        - **Architecture**: Bidirectional LSTM with a CRF (Conditional Random Field) layer on top.
        - **Goal**: Capture sequential dependencies in the text.
        """)
        
    elif doc_section == "Training our model":
        st.markdown("""
        #### Training the Model
        - **Split**: Train/Test split the data.
        - **Hyperparameters**: Experiment with epochs, batch size, and optimizers.
        - **Metrics**: Monitor loss and accuracy during training.
        """)
        
    elif doc_section == "Let's load the best model":
        st.markdown("""
        #### Loading the Best Model
        - After training, save the model weights that achieved the best performance on the validation set.
        - Reload these weights for inference/prediction.
        """)
        
    elif doc_section == "BERT Model":
        st.markdown("""
        #### BERT Model
        - **Model**: `bert-base-uncased` from Hugging Face Transformers.
        - **Advantage**: Pre-trained on a massive corpus, providing rich contextual embeddings.
        - **Fine-tuning**: The model is fine-tuned specifically for the NER task on the tweet dataset.
        """)
        
    elif doc_section == "Loading the tokenizer":
        st.markdown("""
        #### Loading the Tokenizer
        - **Tokenizer**: `BertTokenizer` corresponding to `bert-base-uncased`.
        - **Sub-word Tokenization**: BERT uses WordPiece tokenization.
        - **Alignment**: Special care is taken to align the original NER labels with the sub-word tokens generated by BERT.
        """)
        
    elif doc_section == "Comparision":
        st.markdown("""
        #### Comparison
        - **LSTM+CRF**: Good baseline, faster to train.
        - **BERT**: State-of-the-art performance, better at handling context and unseen words, but computationally heavier.
        - **Conclusion**: BERT generally outperforms LSTM+CRF for this task.
        """)
        
    elif doc_section == "Questions":
        st.markdown("""
        #### Questions & Future Work
        - How to handle new entity types?
        - Can we improve inference speed?
        - How does the model perform on multi-lingual tweets?
        """)

    # Add download button for the PDF
    import os
    pdf_path = "../../tweeter-ner-nlp.pdf" # Relative path from project/frontend
    if os.path.exists(pdf_path):
        st.markdown("---")
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Original PDF Guide",
                data=pdf_file,
                file_name="tweeter-ner-nlp.pdf",
                mime="application/pdf"
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Twitter NER System v1.0 | "
    "Built with ❤️ using FastAPI & Streamlit</div>",
    unsafe_allow_html=True
)
