from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model_utils import NERModel
import uvicorn
import os
import logging
from typing import List, Dict
from datetime import datetime

# Setup logging
# Configure the root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ner_api.log', mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
    force=True  # Force reconfiguration to override any previous setup
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Twitter NER API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model wrapper
DATA_DIR = os.path.dirname(__file__)
bert_model = NERModel(model_type='bert', data_dir=DATA_DIR)

# Training status
training_status = {
    'is_training': False,
    'progress': 0,
    'model_type': None,
    'message': 'Ready'
}

class TextRequest(BaseModel):
    text: str
    model_type: str = 'bert'

class TrainRequest(BaseModel):
    model_type: str = 'bert'
    epochs: int = 3
    batch_size: int = 32

class PredictionResponse(BaseModel):
    words: List[str]
    entities: List[str]
    annotated: List[Dict[str, str]]

@app.on_event("startup")
async def startup_event():
    logger.info("Starting NER API...")
    logger.info("Model will be initialized on first request to improve startup time")

def ensure_model_ready():
    """Lazy initialization of model - called when needed"""
    global bert_model
    
    if bert_model.model is None:
        logger.info("Initializing model on first use...")
        try:
            bert_model.prepare_data()
            
            # Try to load saved BERT model
            model_path = 'bert-base-uncased_ner_model'
            if os.path.exists(model_path):
                logger.info(f"Loading saved BERT model from {model_path}...")
                bert_model.load_saved_model(model_path)
            else:
                logger.info("No saved BERT model found. Building new model...")
                bert_model.build_bert_model()
                
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error during model initialization: {e}", exc_info=True)
            raise

@app.get("/")
async def root():
    return {
        "message": "Twitter NER API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "/predict": "POST - Predict entities in text",
            "/train": "POST - Train the BERT model",
            "/status": "GET - Get training status",
            "/models": "GET - Get model information",
            "/data-stats": "GET - Get dataset statistics",
            "/logs": "GET - Get API logs"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    try:
        logger.info(f"Prediction request received")
        
        # Ensure model is initialized
        ensure_model_ready()
        
        results = bert_model.predict(request.text)
        
        words = [word for word, _ in results]
        entities = [entity for _, entity in results]
        
        annotated = []
        for word, entity in results:
            annotated.append({
                "word": word,
                "entity": entity,
                "color": get_entity_color(entity)
            })
        
        logger.info(f"Prediction completed: {len(words)} words processed")
        
        return PredictionResponse(
            words=words,
            entities=entities,
            annotated=annotated
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def get_entity_color(entity: str) -> str:
    """Get color for entity type"""
    entity_lower = entity.lower()
    if 'person' in entity_lower:
        return '#C0392B'  # Dark Red
    elif 'geo-loc' in entity_lower or 'location' in entity_lower:
        return '#27AE60'  # Dark Green
    elif 'company' in entity_lower:
        return '#2980B9'  # Dark Blue
    elif 'product' in entity_lower:
        return '#D35400'  # Dark Orange
    elif 'facility' in entity_lower:
        return '#8E44AD'  # Purple
    elif 'musicartist' in entity_lower:
        return '#16A085'  # Teal
    elif 'tvshow' in entity_lower:
        return '#F39C12'  # Orange
    elif 'sportsteam' in entity_lower:
        return '#2C3E50'  # Midnight Blue
    else:
        return '#7F8C8D'  # Gray

async def train_model_background(epochs: int, batch_size: int, model_type: str):
    """Background task for model training"""
    global training_status, bert_model
    
    try:
        training_status['is_training'] = True
        training_status['model_type'] = model_type
        training_status['message'] = f'Initializing {model_type}...'
        
        # Check if we need to switch models
        if bert_model.model_name != model_type and model_type != 'lstm':
            logger.info(f"Switching model from {bert_model.model_name} to {model_type}")
            # Re-initialize the model wrapper
            bert_model = NERModel(model_type=model_type, data_dir=DATA_DIR)
            bert_model.prepare_data()
            bert_model.build_bert_model()
        
        training_status['message'] = f'Training {model_type}...'
        logger.info(f"Starting {model_type} training for {epochs} epochs")
        
        history = bert_model.train(epochs=epochs, batch_size=batch_size)
        
        # Save the model
        save_name = f"{model_type.replace('/', '-')}_ner_model"
        bert_model.save_model(save_name)
        
        training_status['is_training'] = False
        training_status['progress'] = 100
        training_status['message'] = f'{model_type} training completed successfully'
        
        logger.info(f"{model_type} training completed and model saved to {save_name}")
        
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)
        training_status['is_training'] = False
        training_status['message'] = f'Training failed: {str(e)}'

@app.post("/train")
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    global training_status
    
    if training_status['is_training']:
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    logger.info(f"Training request received: {request.epochs} epochs, Model: {request.model_type}")
    
    background_tasks.add_task(
        train_model_background,
        request.epochs,
        request.batch_size,
        request.model_type
    )
    
    return {
        "message": f"Training started for {request.model_type}",
        "epochs": request.epochs,
        "batch_size": request.batch_size,
        "model": request.model_type
    }

@app.get("/status")
async def get_status():
    return training_status

@app.get("/models")
async def get_models_info():
    return {
        "bert": {
            "name": "BERT (bert-base-uncased)",
            "loaded": bert_model.model is not None,
            "description": "Transformer-based model for token classification",
            "status": "active"
        }
    }

@app.post("/initialize")
async def initialize_model():
    """Manually initialize the BERT model"""
    try:
        logger.info("Manual model initialization requested")
        
        if bert_model.model is not None:
            return {"message": "Model already initialized", "status": "success"}
        
        # Ensure data is prepared
        if not bert_model.schema:
            bert_model.prepare_data()
        
        # Build the model
        bert_model.build_bert_model()
        
        logger.info("Model initialized successfully")
        return {"message": "Model initialized successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Error initializing model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-stats")
async def get_data_stats():
    try:
        # Ensure model and data are initialized
        ensure_model_ready()
        
        train_samples, test_samples = bert_model.prepare_data()
        
        # Calculate entity distribution
        entity_counts = {}
        for sample in train_samples + test_samples:
            for _, tag in sample:
                entity_counts[tag] = entity_counts.get(tag, 0) + 1
        
        return {
            "train_samples": len(train_samples),
            "test_samples": len(test_samples),
            "total_samples": len(train_samples) + len(test_samples),
            "num_entities": len(bert_model.schema) if bert_model.schema else 0,
            "entities": bert_model.schema if bert_model.schema else [],
            "entity_distribution": entity_counts,
            "max_sequence_length": bert_model.max_len
        }
    except Exception as e:
        logger.error(f"Error getting data stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs")
async def get_logs(lines: int = 50):
    """Get recent log entries"""
    try:
        if os.path.exists('ner_api.log'):
            with open('ner_api.log', 'r') as f:
                log_lines = f.readlines()
                return {"logs": log_lines[-lines:]}
        else:
            return {"logs": [], "message": "No log file found"}
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        return {"logs": [], "error": str(e)}

class LogRequest(BaseModel):
    message: str
    level: str = "INFO"

@app.post("/log-client-event")
async def log_client_event(request: LogRequest):
    """Log an event from the client (frontend)"""
    msg = f"[CLIENT] {request.message}"
    if request.level.upper() == "ERROR":
        logger.error(msg)
    elif request.level.upper() == "WARNING":
        logger.warning(msg)
    else:
        logger.info(msg)
    return {"status": "logged"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
