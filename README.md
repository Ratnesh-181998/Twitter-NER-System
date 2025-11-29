# 🐦 Twitter Named Entity Recognition System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-FF4B4B.svg)](https://streamlit.io/)
[![Transformers](https://img.shields.io/badge/Transformers-4.35.0-orange.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A production-ready Named Entity Recognition system for Twitter data using state-of-the-art Transformer models**

Built by **RATNESH SINGH**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Demo](#-demo)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [UI Sections](#-ui-sections)
- [API Documentation](#-api-documentation)
- [Model Training](#-model-training)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project implements a **Named Entity Recognition (NER)** system specifically designed for Twitter data. It automatically identifies and classifies named entities such as persons, locations, companies, products, and more from informal, noisy tweet text.

### Problem Statement

Twitter generates ~500 million tweets per day. Understanding trends and topics requires going beyond simple hashtag analysis to extract meaningful entities from the content itself. This system addresses:

- **Volume**: Processing large-scale social media data
- **Noise**: Handling informal, unstructured user-generated content
- **Accuracy**: Providing fine-grained entity classification (10+ categories)

### Solution

A full-stack web application powered by BERT (Bidirectional Encoder Representations from Transformers) that provides:
- Real-time entity extraction from text
- Interactive visualization of results
- Model training capabilities
- Comprehensive analytics dashboard

---

## ✨ Features

### 🔍 Core Functionality
- **Real-time NER**: Instant entity extraction from any text input
- **Multi-model Support**: BERT, DistilBERT, RoBERTa, XLM-RoBERTa
- **10+ Entity Types**: Person, Location, Company, Product, Facility, Music Artist, TV Show, Sports Team, and more
- **Visual Analytics**: Interactive charts and entity distribution graphs
- **Model Training**: Train custom models directly from the UI

### 🎨 User Interface
- **Business Case**: Comprehensive project overview and impact analysis
- **About**: System features and capabilities
- **Technical Documentation**: Detailed technical guide from the research paper
- **Analyze**: Real-time entity extraction with visual highlighting
- **Model & Training**: Model selection and training interface
- **Data Statistics**: Dataset insights and entity distribution
- **Logs**: Real-time API monitoring

### 🚀 Performance
- **Lazy Loading**: Optimized startup time (<1 second)
- **Caching**: Smart data caching for improved response times
- **Async Processing**: Non-blocking model operations
- **Health Monitoring**: Real-time backend status indicators

---

## 🎬 Demo

### Entity Extraction Example

**Input:**
```
Apple MacBook is the best laptop in the world
```

**Output:**
- **Apple** → Company
- **MacBook** → Product
- **world** → Geo-location

### Visual Interface

The application features:
- Color-coded entity highlighting
- Interactive entity distribution charts
- Real-time prediction results
- Detailed entity tables with counts

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (Streamlit)                     │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐  │
│  │Business  │  About   │Technical │ Analyze  │  Model   │  │
│  │  Case    │          │   Docs   │          │ Training │  │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │ REST API (HTTP)
┌─────────────────────────▼───────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              API Endpoints                            │  │
│  │  /predict  /train  /status  /data-stats  /logs       │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Model Layer                              │  │
│  │  • BERT/DistilBERT/RoBERTa                           │  │
│  │  • Tokenization & Alignment                          │  │
│  │  • Training & Inference                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                    Data Layer                                │
│  • wnut 16.txt.conll (Training)                             │
│  • wnut 16test.txt.conll (Testing)                          │
│  • Saved Models (PyTorch)                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Frontend
- **Streamlit** `1.28.1` - Interactive web application framework
- **Plotly** `5.17.0` - Interactive data visualization
- **Pandas** `2.1.3` - Data manipulation and analysis
- **annotated-text** `4.0.1` - Text annotation display

### Backend
- **FastAPI** `0.104.1` - Modern, high-performance web framework
- **Uvicorn** `0.24.0` - ASGI server
- **Pydantic** `2.5.0` - Data validation

### Machine Learning
- **PyTorch** `2.1.1` - Deep learning framework
- **Transformers** `4.35.0` - Hugging Face transformers library
- **NumPy** `1.26.2` - Numerical computing

### Supported Models
1. **BERT** (bert-base-uncased) - 110M parameters
2. **DistilBERT** (distilbert-base-uncased) - 66M parameters (faster)
3. **RoBERTa** (roberta-base) - 125M parameters (improved BERT)
4. **XLM-RoBERTa** (xlm-roberta-base) - Multilingual support

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (8GB recommended for training)
- Internet connection (for first-time model download)

### Step 1: Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/twitter-ner-system.git
cd twitter-ner-system
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation
```bash
python --version  # Should be 3.8+
pip list | grep transformers  # Verify transformers is installed
```

---

## 🚀 Usage

### Quick Start

1. **Start the Backend Server**
```bash
cd project/backend
python -m uvicorn main:app --port 8000 --reload
```

2. **Start the Frontend Application** (in a new terminal)
```bash
cd project/frontend
streamlit run app.py --server.port 8501
```

3. **Access the Application**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### First-Time Setup

On first run, the system will:
1. Download the BERT model (~400MB) - takes 2-5 minutes
2. Load and prepare the training data
3. Initialize the model for inference

**Note**: Subsequent runs are instant due to caching!

---

## 🎨 UI Sections

### 1. 💼 Business Case
- **Objective**: Understanding Twitter trends through NER
- **Challenge**: Processing 500M+ tweets/day with noisy data
- **Solution**: Automated entity extraction for trend analysis
- **Impact**: Improved content recommendation and ad targeting
<img width="2082" height="1476" alt="image" src="https://github.com/user-attachments/assets/8169eed8-2918-44c0-9305-23950ba6bf07" />

### 2. ℹ️ About
- System features and capabilities
- Supported entity types (10+ categories)
- Model architecture overview
- Dataset information
<img width="2236" height="1391" alt="image" src="https://github.com/user-attachments/assets/1636ff4e-91a6-4cd9-8f45-46af4240efe3" />

### 3. 📚 Technical Documentation
Interactive navigation through:
- Problem Statement
- Data Description (CoNLL format, BIO tagging)
- Process Overview
- LSTM + CRF Model Training
- BERT Model Implementation
- Tokenization & Alignment
- Model Comparison
- Future Work & Questions
<img width="2088" height="1382" alt="image" src="https://github.com/user-attachments/assets/2f15d399-d293-4db6-8c0a-f77899372253" />

### 4. 🔍 Analyze
- **Text Input**: Enter any text for entity extraction
- **Visual Output**: Color-coded entity highlighting
- **Analytics**:
  - Total entities found
  - Unique entity types
  - Entity distribution chart
  - Detailed entity table with counts
<img width="2868" height="1258" alt="image" src="https://github.com/user-attachments/assets/13a4f0a2-83e8-4ea2-b267-771000a94883" />
<img width="2824" height="1353" alt="image" src="https://github.com/user-attachments/assets/d5ea5dd2-d828-4743-9a67-2a4462689ac3" />

### 5. 🛠️ Model & Training
- **Model Selection**: Choose from 5 model architectures
- **Training Controls**:
  - Epochs (1-10)
  - Batch Size (8-64)
  - Real-time training progress
- **Dataset Info**: Training/validation split details
<img width="2762" height="1393" alt="image" src="https://github.com/user-attachments/assets/e8c15b58-9ff8-4431-96e3-6e1fb0e3157f" />

### 6. 📊 Data Statistics
- Training samples: Count and distribution
- Test samples: Validation data overview
- Entity distribution: Visual breakdown
- Max sequence length: Data characteristics
<img width="2841" height="1387" alt="image" src="https://github.com/user-attachments/assets/df6886f6-67f6-49c7-b543-79f508c8b8be" />

### 7. 📝 Logs
- Real-time API activity monitoring
- Error tracking and debugging
- Download logs functionality
<img width="2862" height="1399" alt="image" src="https://github.com/user-attachments/assets/c899acf7-04f2-439b-852c-6cdbb0422960" />

---

## 📡 API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### 1. Health Check
```http
GET /
```
Returns API status and available endpoints.

#### 2. Predict Entities
```http
POST /predict
Content-Type: application/json

{
  "text": "Apple MacBook is the best laptop in the world"
}
```

**Response:**
```json
{
  "words": ["Apple", "MacBook", "is", "the", "best", "laptop", "in", "the", "world"],
  "entities": ["B-company", "B-product", "O", "O", "O", "O", "O", "O", "B-geo-loc"],
  "annotated": [
    {"word": "Apple", "entity": "B-company", "color": "#2980B9"},
    {"word": "MacBook", "entity": "B-product", "color": "#D35400"},
    ...
  ]
}
```

#### 3. Train Model
```http
POST /train
Content-Type: application/json

{
  "model_type": "bert-base-uncased",
  "epochs": 3,
  "batch_size": 32
}
```

#### 4. Training Status
```http
GET /status
```

#### 5. Data Statistics
```http
GET /data-stats
```

#### 6. API Logs
```http
GET /logs?lines=100
```

---

## 🎓 Model Training

### Training Process

1. **Select Model**: Choose from BERT, DistilBERT, RoBERTa, or XLM-RoBERTa
2. **Configure Parameters**:
   - Epochs: Number of training iterations (recommended: 3-5)
   - Batch Size: Samples per batch (recommended: 16-32)
3. **Start Training**: Click "Start Training" button
4. **Monitor Progress**: View real-time training status
5. **Model Saved**: Automatically saved as `{model_name}_ner_model`

### Training Data
- **Format**: CoNLL (BIO tagging scheme)
- **Training Set**: `wnut 16.txt.conll`
- **Test Set**: `wnut 16test.txt.conll`
- **Entity Tags**: 10+ fine-grained categories

### Performance Tips
- Use **DistilBERT** for faster training (66M params)
- Use **BERT** for best accuracy (110M params)
- Increase batch size if you have more RAM
- Monitor logs for training progress

---

## 📊 Dataset

### WNUT-16 Dataset
- **Source**: Workshop on Noisy User-generated Text (WNUT) 2016
- **Domain**: Twitter/Social Media
- **Format**: CoNLL (one word per line, BIO tagging)
- **Entities**: 10 fine-grained types

### Entity Types
1. **person** - Names of people
2. **geo-loc** - Geographic locations
3. **company** - Company/organization names
4. **product** - Product names
5. **facility** - Buildings and facilities
6. **musicartist** - Musicians and bands
7. **tvshow** - TV show titles
8. **sportsteam** - Sports team names
9. **movie** - Movie titles
10. **other** - Other named entities

### Example Format
```
Harry       B-person
Potter      I-person
was         O
living      O
in          O
London      B-geo-loc
```

---

## 📁 Project Structure

```
project/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── model_utils.py          # NER model implementation
│   ├── train_initial.py        # Initial training script
│   └── ner_api.log            # API logs
├── frontend/
│   └── app.py                  # Streamlit application
├── wnut 16.txt.conll          # Training data
├── wnut 16test.txt.conll      # Test data
├── tweeter-ner-nlp.pdf        # Technical documentation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
└── .gitignore                 # Git ignore rules
```

---

## 📈 Performance

### Model Comparison

| Model | Parameters | Accuracy | Speed | Memory |
|-------|-----------|----------|-------|--------|
| BERT | 110M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 400MB |
| DistilBERT | 66M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 250MB |
| RoBERTa | 125M | ⭐⭐⭐⭐⭐ | ⭐⭐ | 500MB |
| XLM-RoBERTa | 125M | ⭐⭐⭐⭐⭐ | ⭐⭐ | 500MB |

### Optimization Features
- ✅ Lazy model loading (startup < 1 second)
- ✅ Data caching (60-second TTL)
- ✅ Async API endpoints
- ✅ Batch processing support
- ✅ GPU acceleration (if available)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

---

## 📞 Contact

**RATNESH SINGH**

- 📧 Email: [rattudacsit2021gate@gmail.com](mailto:rattudacsit2021gate@gmail.com)
- 💼 LinkedIn: [https://www.linkedin.com/in/ratneshkumar1998/](https://www.linkedin.com/in/ratneshkumar1998/)
- 🐙 GitHub: [https://github.com/Ratnesh-181998](https://github.com/Ratnesh-181998)
- 📱 Phone: +91-947XXXXX46

### Project Links
- 🌐 Live Demo: [Streamlit](https://twitter-ner-system-ab12c.streamlit.app/)
- 📖 Documentation: [GitHub Wiki](https://github.com/Ratnesh-181998/twitter-ner-system/wiki)
- 🐛 Issue Tracker: [GitHub Issues](https://github.com/Ratnesh-181998/twitter-ner-system/issues)

---

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **WNUT-16** for the dataset
- **FastAPI** and **Streamlit** communities
- **PyTorch** team for the deep learning framework

---

## 📚 References

1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT.
3. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
4. WNUT-16 Shared Task on Named Entity Recognition in Twitter.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by **RATNESH SINGH**

</div>
