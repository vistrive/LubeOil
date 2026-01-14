"""Blend API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.db import get_db
from lobp.models.blend import BlendPriority, BlendStatus
from lobp.schemas.blend import (
    BlendCreate,
    BlendProgressUpdate,
    BlendQueueItem,
    BlendResponse,
    BlendStatusUpdate,
    BlendUpdate,
)
from lobp.schemas.common import Message
from lobp.services.blend_service import BlendService

router = APIRouter()


def get_blend_service(db: AsyncSession = Depends(get_db)) -> BlendService:
    """Dependency for blend service."""
    return BlendService(db)


@router.get("", response_model=list[BlendResponse])
async def list_blends(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    blend_status: BlendStatus | None = None,
    priority: BlendPriority | None = None,
    service: BlendService = Depends(get_blend_service),
) -> list[BlendResponse]:
    """
    Get all blends.

    Supports filtering by status and priority.
    """
    blends = await service.get_all(
        skip=skip, limit=limit, status=blend_status, priority=priority
    )
    return [BlendResponse.model_validate(b) for b in blends]


@router.get("/queue", response_model=list[BlendQueueItem])
async def get_blend_queue(
    service: BlendService = Depends(get_blend_service),
) -> list[BlendQueueItem]:
    """Get active blends in the queue."""
    blends = await service.get_queue()
    return [
        BlendQueueItem(
            id=b.id,
            batch_number=b.batch_number,
            recipe_code=b.recipe.code if b.recipe else "N/A",
            recipe_name=b.recipe.name if b.recipe else "N/A",
            target_volume_liters=b.target_volume_liters,
            status=b.status,
            priority=b.priority,
            scheduled_start=b.scheduled_start,
            progress_percent=b.progress_percent,
            blend_tank_tag=b.blend_tank_tag,
        )
        for b in blends
    ]


@router.get("/{blend_id}", response_model=BlendResponse)
async def get_blend(
    blend_id: str,
    service: BlendService = Depends(get_blend_service),
) -> BlendResponse:
    """Get a blend by ID."""
    blend = await service.get_by_id(blend_id)
    if not blend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blend {blend_id} not found",
        )
    return BlendResponse.model_validate(blend)


@router.get("/batch/{batch_number}", response_model=BlendResponse)
async def get_blend_by_batch(
    batch_number: str,
    service: BlendService = Depends(get_blend_service),
) -> BlendResponse:
    """Get a blend by batch number."""
    blend = await service.get_by_batch_number(batch_number)
    if not blend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blend with batch number {batch_number} not found",
        )
    return BlendResponse.model_validate(blend)


@router.post("", response_model=BlendResponse, status_code=status.HTTP_201_CREATED)
async def create_blend(
    blend_data: BlendCreate,
    created_by: str = Query(None, description="Username of creator"),
    service: BlendService = Depends(get_blend_service),
) -> BlendResponse:
    """
    Create a new blend.

    Can include custom ingredients or use recipe defaults.
    """
    blend = await service.create(blend_data, created_by)
    return BlendResponse.model_validate(blend)


@router.post("/from-recipe", response_model=BlendResponse, status_code=status.HTTP_201_CREATED)
async def create_blend_from_recipe(
    recipe_id: str = Query(..., description="Recipe ID to use"),
    target_volume: float = Query(..., gt=0, description="Target volume in liters"),
    created_by: str = Query(None, description="Username of creator"),
    service: BlendService = Depends(get_blend_service),
) -> BlendResponse:
    """Create a new blend using recipe defaults."""
    blend = await service.create_from_recipe(recipe_id, target_volume, created_by)
    if not blend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recipe {recipe_id} not found",
        )
    return BlendResponse.model_validate(blend)


@router.patch("/{blend_id}", response_model=BlendResponse)
async def update_blend(
    blend_id: str,
    blend_data: BlendUpdate,
    service: BlendService = Depends(get_blend_service),
) -> BlendResponse:
    """Update a blend."""
    blend = await service.update(blend_id, blend_data)
    if not blend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blend {blend_id} not found",
        )
    return BlendResponse.model_validate(blend)


@router.put("/{blend_id}/status", response_model=BlendResponse)
async def update_blend_status(
    blend_id: str,
    status_data: BlendStatusUpdate,
    service: BlendService = Depends(get_blend_service),
) -> BlendResponse:
    """Update blend status."""
    blend = await service.update_status(blend_id, status_data.status, status_data.notes)
    if not blend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blend {blend_id} not found",
        )
    return BlendResponse.model_validate(blend)


@router.put("/{blend_id}/progress", response_model=BlendResponse)
async def update_blend_progress(
    blend_id: str,
    progress_data: BlendProgressUpdate,
    service: BlendService = Depends(get_blend_service),
) -> BlendResponse:
    """
    Update blend progress from DCS.

    This endpoint is called by the DCS integration to update real-time progress.
    """
    blend = await service.update_progress(blend_id, progress_data)
    if not blend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blend {blend_id} not found",
        )
    return BlendResponse.model_validate(blend)


@router.post("/{blend_id}/hold", response_model=BlendResponse)
async def hold_blend(
    blend_id: str,
    reason: str = Query(..., min_length=1, description="Reason for hold"),
    service: BlendService = Depends(get_blend_service),
) -> BlendResponse:
    """Put a blend on hold."""
    blend = await service.hold_blend(blend_id, reason)
    if not blend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blend {blend_id} not found",
        )
    return BlendResponse.model_validate(blend)


@router.post("/{blend_id}/start", response_model=BlendResponse)
async def start_blend(
    blend_id: str,
    service: BlendService = Depends(get_blend_service),
) -> BlendResponse:
    """Start a blend operation."""
    blend = await service.update_status(blend_id, BlendStatus.IN_PROGRESS)
    if not blend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blend {blend_id} not found",
        )
    return BlendResponse.model_validate(blend)


@router.post("/{blend_id}/complete", response_model=BlendResponse)
async def complete_blend(
    blend_id: str,
    service: BlendService = Depends(get_blend_service),
) -> BlendResponse:
    """Mark a blend as completed."""
    blend = await service.update_status(blend_id, BlendStatus.COMPLETED)
    if not blend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blend {blend_id} not found",
        )
    return BlendResponse.model_validate(blend)


@router.get("/{blend_id}/off-spec-check", response_model=dict)
async def check_off_spec_risk(
    blend_id: str,
    service: BlendService = Depends(get_blend_service),
) -> dict:
    """Check if blend has high off-spec risk."""
    blend = await service.get_by_id(blend_id)
    if not blend:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blend {blend_id} not found",
        )

    is_high_risk = await service.check_off_spec_risk(blend_id)
    return {
        "blend_id": blend_id,
        "off_spec_risk_percent": blend.off_spec_risk_percent,
        "is_high_risk": is_high_risk,
        "threshold_percent": 5.0,
    }
