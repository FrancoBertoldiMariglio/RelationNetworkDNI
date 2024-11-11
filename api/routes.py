import os
from contextlib import asynccontextmanager
from fastapi import APIRouter, HTTPException, Depends, FastAPI
import logging
from api.schemas import PredictionRequest, PredictionResponse
from api.service import RelationNetPredictor

logger = logging.getLogger(__name__)

# Load environment variables
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "api/best_model.pth")
SUPPORT_FOLDER = os.getenv("SUPPORT_FOLDER", "api/support_images")
DEVICE = os.getenv("DEVICE", "cuda")

# Initialize predictor instance
predictor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global predictor
    try:
        predictor = RelationNetPredictor(
            checkpoint_path=CHECKPOINT_PATH,
            device=DEVICE
        )
        logger.info("RelationNet predictor initialized successfully")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {str(e)}")
        raise
    finally:
        # Cleanup
        if predictor:
            # Add any cleanup if needed
            pass


def get_predictor() -> RelationNetPredictor:
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Predictor service not initialized"
        )
    return predictor


def create_router() -> APIRouter:
    router = APIRouter()

    @router.post("/predict", response_model=PredictionResponse)
    async def predict(
            request: PredictionRequest,
            predictor: RelationNetPredictor = Depends(get_predictor)
    ) -> PredictionResponse:
        try:
            confidence = predictor.predict(
                query_base64=request.base64_image,
                support_folder="support_images"
            )
            return PredictionResponse(confidence=confidence)

        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail="Internal server error")

    return router