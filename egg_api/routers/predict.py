"""
Prediction Router Endpoint
POST /predict-image: Accepts multipart cropped image file and returns prediction JSON.
"""

import logging
import PIL
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from schemas import PredictionResponse, RGBResponse, CIELABResponse
from services.color_service import extract_color_features
from services.predict_service import get_predict_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])

# Maximum allowed upload size: 10 MB
MAX_FILE_SIZE = 10 * 1024 * 1024

# Allowed MIME types for image upload
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


@router.post(
    "/predict-image",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict Egg Yolk Color Fan Score from cropped image"
)
async def predict_image(file: UploadFile = File(...)):
    """
    Receives a cropped egg yolk image file via multipart/form-data,
    extracts RGB and CIELAB color features, and predicts the Yolk Color Fan score (1-15).
    """
    # Validate content-type before reading the file
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: jpeg, png, webp, bmp."
        )

    try:
        # Read uploaded image bytes
        image_bytes = await file.read()

        # Guard against excessively large uploads (DoS prevention)
        if len(image_bytes) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE // (1024 * 1024)} MB."
            )

        # Extract color features (RGB + CIELAB) using canonical Python color service
        color_feats = extract_color_features(image_bytes)

        # Get prediction service and calculate score
        predictor = get_predict_service()
        pred_score, raw_score = predictor.predict(
            r=color_feats['r'],
            g=color_feats['g'],
            b=color_feats['b'],
            l=color_feats['l'],
            a=color_feats['a'],
            b_lab=color_feats['b_lab']
        )

        return PredictionResponse(
            predicted_score=pred_score,
            raw_score=raw_score,
            rgb=RGBResponse(
                r=color_feats['r'],
                g=color_feats['g'],
                b=color_feats['b']
            ),
            cielab=CIELABResponse(
                l=color_feats['l'],
                a=color_feats['a'],
                b=color_feats['b_lab'],
                chroma=color_feats['chroma'],
                hue_angle=color_feats['hue_angle']
            )
        )

    except HTTPException:
        raise
    except PIL.UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a valid image."
        )
    except Exception as e:
        logger.exception("Unexpected error during image prediction")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction. Please try again."
        )
    finally:
        await file.close()
