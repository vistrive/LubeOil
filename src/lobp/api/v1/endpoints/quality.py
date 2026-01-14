"""Quality API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.db import get_db
from lobp.schemas.quality import (
    QualityMeasurementCreate,
    QualityMeasurementResponse,
    QualityMeasurementUpdate,
    QualityPredictionCreate,
    QualityPredictionResponse,
)
from lobp.services.quality_service import QualityService

router = APIRouter()


def get_quality_service(db: AsyncSession = Depends(get_db)) -> QualityService:
    """Dependency for quality service."""
    return QualityService(db)


# Quality Measurements


@router.get("/measurements", response_model=list[QualityMeasurementResponse])
async def list_measurements(
    blend_id: str | None = None,
    tank_tag: str | None = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    service: QualityService = Depends(get_quality_service),
) -> list[QualityMeasurementResponse]:
    """
    Get quality measurements.

    Supports filtering by blend ID or tank tag.
    """
    measurements = await service.get_measurements(
        blend_id=blend_id, tank_tag=tank_tag, skip=skip, limit=limit
    )
    return [QualityMeasurementResponse.model_validate(m) for m in measurements]


@router.get("/measurements/{measurement_id}", response_model=QualityMeasurementResponse)
async def get_measurement(
    measurement_id: str,
    service: QualityService = Depends(get_quality_service),
) -> QualityMeasurementResponse:
    """Get a quality measurement by ID."""
    measurement = await service.get_measurement_by_id(measurement_id)
    if not measurement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Measurement {measurement_id} not found",
        )
    return QualityMeasurementResponse.model_validate(measurement)


@router.post(
    "/measurements",
    response_model=QualityMeasurementResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_measurement(
    measurement_data: QualityMeasurementCreate,
    service: QualityService = Depends(get_quality_service),
) -> QualityMeasurementResponse:
    """
    Create a new quality measurement.

    Can be from inline analyzers, lab analysis, or portable instruments.
    """
    measurement = await service.create_measurement(measurement_data)
    return QualityMeasurementResponse.model_validate(measurement)


@router.patch("/measurements/{measurement_id}", response_model=QualityMeasurementResponse)
async def update_measurement(
    measurement_id: str,
    update_data: QualityMeasurementUpdate,
    service: QualityService = Depends(get_quality_service),
) -> QualityMeasurementResponse:
    """Update a quality measurement."""
    measurement = await service.update_measurement(measurement_id, update_data)
    if not measurement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Measurement {measurement_id} not found",
        )
    return QualityMeasurementResponse.model_validate(measurement)


@router.get(
    "/measurements/blend/{blend_id}/latest",
    response_model=QualityMeasurementResponse,
)
async def get_latest_blend_measurement(
    blend_id: str,
    service: QualityService = Depends(get_quality_service),
) -> QualityMeasurementResponse:
    """Get the latest quality measurement for a blend."""
    measurement = await service.get_latest_blend_measurement(blend_id)
    if not measurement:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No measurements found for blend {blend_id}",
        )
    return QualityMeasurementResponse.model_validate(measurement)


# Quality Predictions


@router.get("/predictions/blend/{blend_id}", response_model=list[QualityPredictionResponse])
async def list_predictions(
    blend_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    service: QualityService = Depends(get_quality_service),
) -> list[QualityPredictionResponse]:
    """Get quality predictions for a blend."""
    predictions = await service.get_predictions(blend_id, skip=skip, limit=limit)
    return [QualityPredictionResponse.model_validate(p) for p in predictions]


@router.get("/predictions/{prediction_id}", response_model=QualityPredictionResponse)
async def get_prediction(
    prediction_id: str,
    service: QualityService = Depends(get_quality_service),
) -> QualityPredictionResponse:
    """Get a quality prediction by ID."""
    prediction = await service.get_prediction_by_id(prediction_id)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found",
        )
    return QualityPredictionResponse.model_validate(prediction)


@router.post(
    "/predictions",
    response_model=QualityPredictionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_prediction(
    prediction_data: QualityPredictionCreate,
    service: QualityService = Depends(get_quality_service),
) -> QualityPredictionResponse:
    """
    Create a new quality prediction.

    Called by the AI system to store predictions.
    """
    prediction = await service.create_prediction(prediction_data)
    return QualityPredictionResponse.model_validate(prediction)


@router.post("/predictions/{prediction_id}/verify", response_model=QualityPredictionResponse)
async def verify_prediction(
    prediction_id: str,
    measurement_id: str = Query(..., description="Measurement ID to verify against"),
    accuracy: float = Query(..., ge=0, le=100, description="Prediction accuracy percentage"),
    service: QualityService = Depends(get_quality_service),
) -> QualityPredictionResponse:
    """Verify a prediction against an actual measurement."""
    prediction = await service.verify_prediction(prediction_id, measurement_id, accuracy)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prediction {prediction_id} not found",
        )
    return QualityPredictionResponse.model_validate(prediction)


@router.get(
    "/predictions/blend/{blend_id}/latest",
    response_model=QualityPredictionResponse,
)
async def get_latest_prediction(
    blend_id: str,
    service: QualityService = Depends(get_quality_service),
) -> QualityPredictionResponse:
    """Get the latest quality prediction for a blend."""
    prediction = await service.get_latest_prediction(blend_id)
    if not prediction:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No predictions found for blend {blend_id}",
        )
    return QualityPredictionResponse.model_validate(prediction)
