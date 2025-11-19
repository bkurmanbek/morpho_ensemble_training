"""
Production Deployment Script
=============================

FastAPI server for serving the ensemble model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uvicorn
import logging
from ensemble_model import create_ensemble, MorphologyOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Kazakh Morphology Ensemble API",
    description="REST API for Kazakh morphological analysis using ensemble of specialized models",
    version="1.0.0"
)

# Global ensemble instance
ensemble = None


class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    word: str
    pos_tag: str
    use_validation: bool = True


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction"""
    words: List[Dict[str, str]]  # [{"word": "...", "pos_tag": "..."}, ...]
    batch_size: int = 8
    use_validation: bool = True


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    word: str
    pos_tag: str
    lemma: str
    morphology: Dict[str, str]
    semantics: Dict[str, str]
    lexics: Dict[str, str]
    sozjasam: Dict[str, str]
    confidence: float
    source: str


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    results: List[PredictionResponse]
    count: int
    average_confidence: float


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    models_loaded: bool
    message: str


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    global ensemble
    
    logger.info("Loading ensemble models...")
    
    try:
        ensemble = create_ensemble(
            grammar_data_path="all_kazakh_grammar_data.json",
            pattern_db_path="pattern_database.json"
        )
        
        ensemble.load_models()
        
        logger.info("Ensemble models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ensemble models: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    models_loaded = ensemble is not None and ensemble.models_loaded
    
    return HealthResponse(
        status="healthy" if models_loaded else "unhealthy",
        models_loaded=models_loaded,
        message="Ensemble is ready" if models_loaded else "Ensemble not loaded"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict morphology for a single word.
    
    Args:
        request: PredictionRequest with word and POS tag
        
    Returns:
        PredictionResponse with complete morphological analysis
    """
    
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Ensemble not loaded")
    
    try:
        result = ensemble.predict(
            word=request.word,
            pos_tag=request.pos_tag,
            use_validation=request.use_validation
        )
        
        return PredictionResponse(
            word=result.word,
            pos_tag=result.pos_tag,
            lemma=result.lemma,
            morphology=result.morphology,
            semantics=result.semantics,
            lexics=result.lexics,
            sozjasam=result.sozjasam,
            confidence=result.confidence,
            source=result.source
        )
    
    except Exception as e:
        logger.error(f"Prediction error for {request.word}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict morphology for multiple words.
    
    Args:
        request: BatchPredictionRequest with list of words
        
    Returns:
        BatchPredictionResponse with list of predictions
    """
    
    if ensemble is None:
        raise HTTPException(status_code=503, detail="Ensemble not loaded")
    
    try:
        # Convert to list of tuples
        words = [(item["word"], item["pos_tag"]) for item in request.words]
        
        results = ensemble.predict_batch(
            words=words,
            batch_size=request.batch_size
        )
        
        # Convert to response format
        response_results = [
            PredictionResponse(
                word=result.word,
                pos_tag=result.pos_tag,
                lemma=result.lemma,
                morphology=result.morphology,
                semantics=result.semantics,
                lexics=result.lexics,
                sozjasam=result.sozjasam,
                confidence=result.confidence,
                source=result.source
            )
            for result in results
        ]
        
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        return BatchPredictionResponse(
            results=response_results,
            count=len(response_results),
            average_confidence=avg_confidence
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Kazakh Morphology Ensemble API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/predict": "Single prediction",
            "/predict/batch": "Batch prediction",
            "/docs": "API documentation"
        }
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port)