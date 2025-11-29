# 🚀 Quick Start Guide - Twitter NER Project

## ⚡ 5-Minute Setup

### Step 1: Verify Installation
```bash
# Check if dependencies are installed
pip list | findstr "fastapi streamlit tensorflow transformers"
```

### Step 2: Start the Application

**Option A: Using Two Terminals (Recommended)**

Terminal 1 - Backend:
```bash
cd "C:\Users\rattu\Downloads\Tweeter NER NLP Bussiness case\project\backend"
python -m uvicorn main:app --port 8000 --reload
```

Terminal 2 - Frontend:
```bash
cd "C:\Users\rattu\Downloads\Tweeter NER NLP Bussiness case\project\frontend"
streamlit run app.py --server.port 8501
```

**Option B: Using Batch Script**

Create `start.bat` in the project folder:
```batch
@echo off
start cmd /k "cd backend && python -m uvicorn main:app --port 8000 --reload"
timeout /t 5
start cmd /k "cd frontend && streamlit run app.py --server.port 8501"
```

Then double-click `start.bat`

### Step 3: Access the Application

Open your browser and go to:
- **Main UI**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 🎯 First Time Usage

### 1. Test Prediction (No Training Required)

The models are initialized and ready to use immediately!

1. Go to http://localhost:8501
2. Enter text: "Apple MacBook Pro is the best laptop"
3. Select model: BERT
4. Click "Analyze Text"
5. See the results!

### 2. View Dataset Statistics

1. Click "Data Statistics" tab
2. See training/test sample counts
3. View entity distribution

### 3. Train a Model (Optional)

**Quick Training (1 epoch - ~5 minutes):**

1. In sidebar, select "Model to train": BERT
2. Set Epochs: 1
3. Set Batch Size: 32
4. Click "Start Training"
5. Monitor progress in sidebar

**Full Training (3 epochs - ~15 minutes):**
- Same as above but set Epochs: 3

### 4. Compare Models

1. Try prediction with BERT
2. Switch to BiLSTM+CRF
3. Compare results!

## 📝 Sample Texts to Try

### Product Detection:
```
Apple MacBook Pro is the best laptop in the world
```

### Location Detection:
```
I visited New York City and saw the Empire State Building
```

### Person Detection:
```
Elon Musk announced Tesla's new electric car model
```

### Mixed Entities:
```
Google CEO Sundar Pichai spoke at the conference in San Francisco about Android updates
```

## 🔧 Troubleshooting

### Backend Not Starting?

```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill process if needed
taskkill /PID <PID> /F

# Restart backend
cd project/backend
python -m uvicorn main:app --port 8000
```

### Frontend Not Starting?

```bash
# Check if port 8501 is in use
netstat -ano | findstr :8501

# Kill process if needed
taskkill /PID <PID> /F

# Restart frontend
cd project/frontend
streamlit run app.py --server.port 8501
```

### Module Not Found Errors?

```bash
# Reinstall dependencies
pip install -r project/requirements.txt
```

### TensorFlow Errors?

```bash
# Ensure correct TensorFlow version
pip install tensorflow==2.15.0
pip install tensorflow-addons==0.22.0
```

## 📊 API Quick Reference

### Predict Entities (Python):
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "Your text here",
        "model_type": "bert"  # or "lstm_crf"
    }
)
print(response.json())
```

### Predict Entities (cURL):
```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"Apple MacBook Pro\", \"model_type\": \"bert\"}"
```

### Start Training (Python):
```python
import requests

response = requests.post(
    "http://localhost:8000/train",
    json={
        "model_type": "bert",
        "epochs": 3,
        "batch_size": 32
    }
)
print(response.json())
```

### Check Training Status:
```python
import requests

response = requests.get("http://localhost:8000/status")
print(response.json())
```

## 🎓 Learning Path

### Beginner:
1. ✅ Start the application
2. ✅ Try sample predictions
3. ✅ View data statistics
4. ✅ Read the About tab

### Intermediate:
1. ✅ Train BERT model (1 epoch)
2. ✅ Compare BERT vs LSTM+CRF
3. ✅ View API logs
4. ✅ Try custom text

### Advanced:
1. ✅ Train both models fully
2. ✅ Use API programmatically
3. ✅ Analyze training logs
4. ✅ Modify model parameters

## 📱 Keyboard Shortcuts

### Streamlit UI:
- `Ctrl + R` - Rerun app
- `Ctrl + Shift + R` - Clear cache and rerun
- `Ctrl + K` - Command palette

## 🎨 UI Navigation

```
┌─────────────────────────────────────┐
│  🐦 Twitter NER Analyzer            │
├─────────────────────────────────────┤
│ Sidebar:                            │
│  ├─ Settings (Model Selection)      │
│  ├─ Model Information               │
│  └─ Training Controls                │
├─────────────────────────────────────┤
│ Main Tabs:                          │
│  ├─ 🔍 Analyze (Main feature)       │
│  ├─ 📊 Data Statistics              │
│  ├─ 📝 Logs                         │
│  └─ ℹ️ About                        │
└─────────────────────────────────────┘
```

## ⏱️ Expected Times

| Task | Time |
|------|------|
| First startup | 30-60 seconds |
| Prediction | 1-2 seconds |
| Training (1 epoch, BERT) | 5-10 minutes |
| Training (1 epoch, LSTM+CRF) | 2-3 minutes |
| Training (3 epochs, BERT) | 15-30 minutes |
| Training (3 epochs, LSTM+CRF) | 6-9 minutes |

## 🎯 Success Checklist

- [ ] Backend running on port 8000
- [ ] Frontend running on port 8501
- [ ] Can access UI at http://localhost:8501
- [ ] Can see API docs at http://localhost:8000/docs
- [ ] Prediction works with sample text
- [ ] Data statistics load correctly
- [ ] Logs are visible

## 📞 Need Help?

1. Check logs in sidebar
2. View API logs: http://localhost:8000/logs
3. Check terminal output
4. Review PROJECT_SUMMARY.md
5. Check README.md

---

**Ready to start?** Open http://localhost:8501 and begin analyzing! 🚀
