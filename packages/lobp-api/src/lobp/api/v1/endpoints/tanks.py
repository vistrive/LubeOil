"""Tank API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.db import get_db
from lobp.models.tank import TankStatus, TankType
from lobp.schemas.common import Message
from lobp.schemas.tank import (
    TankContentsUpdate,
    TankCreate,
    TankLevelUpdate,
    TankResponse,
    TankUpdate,
)
from lobp.services.tank_service import TankService

router = APIRouter()


def get_tank_service(db: AsyncSession = Depends(get_db)) -> TankService:
    """Dependency for tank service."""
    return TankService(db)


@router.get("", response_model=list[TankResponse])
async def list_tanks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    tank_type: TankType | None = None,
    tank_status: TankStatus | None = None,
    service: TankService = Depends(get_tank_service),
) -> list[TankResponse]:
    """
    Get all tanks.

    Supports filtering by type and status.
    """
    tanks = await service.get_all(
        skip=skip, limit=limit, tank_type=tank_type, status=tank_status
    )
    return [TankResponse.model_validate(t) for t in tanks]


@router.get("/inventory", response_model=dict)
async def get_inventory_summary(
    service: TankService = Depends(get_tank_service),
) -> dict:
    """Get inventory summary grouped by material code."""
    return await service.get_inventory_summary()


@router.get("/low-stock", response_model=list[TankResponse])
async def get_low_stock_tanks(
    service: TankService = Depends(get_tank_service),
) -> list[TankResponse]:
    """Get tanks with low stock levels for alerts."""
    tanks = await service.check_low_stock_alerts()
    return [TankResponse.model_validate(t) for t in tanks]


@router.get("/blend-tanks", response_model=list[TankResponse])
async def get_available_blend_tanks(
    required_capacity: float = Query(..., gt=0, description="Required capacity in liters"),
    service: TankService = Depends(get_tank_service),
) -> list[TankResponse]:
    """Get available blending tanks with sufficient capacity."""
    tanks = await service.get_available_blend_tanks(required_capacity)
    return [TankResponse.model_validate(t) for t in tanks]


@router.get("/{tank_id}", response_model=TankResponse)
async def get_tank(
    tank_id: str,
    service: TankService = Depends(get_tank_service),
) -> TankResponse:
    """Get a tank by ID."""
    tank = await service.get_by_id(tank_id)
    if not tank:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tank {tank_id} not found",
        )
    return TankResponse.model_validate(tank)


@router.get("/tag/{tag}", response_model=TankResponse)
async def get_tank_by_tag(
    tag: str,
    service: TankService = Depends(get_tank_service),
) -> TankResponse:
    """Get a tank by tag."""
    tank = await service.get_by_tag(tag)
    if not tank:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tank with tag {tag} not found",
        )
    return TankResponse.model_validate(tank)


@router.post("", response_model=TankResponse, status_code=status.HTTP_201_CREATED)
async def create_tank(
    tank_data: TankCreate,
    service: TankService = Depends(get_tank_service),
) -> TankResponse:
    """Create a new tank."""
    # Check if tag already exists
    existing = await service.get_by_tag(tank_data.tag)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tank with tag {tank_data.tag} already exists",
        )

    tank = await service.create(tank_data)
    return TankResponse.model_validate(tank)


@router.patch("/{tank_id}", response_model=TankResponse)
async def update_tank(
    tank_id: str,
    tank_data: TankUpdate,
    service: TankService = Depends(get_tank_service),
) -> TankResponse:
    """Update tank configuration."""
    tank = await service.update(tank_id, tank_data)
    if not tank:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tank {tank_id} not found",
        )
    return TankResponse.model_validate(tank)


@router.put("/{tank_id}/level", response_model=TankResponse)
async def update_tank_level(
    tank_id: str,
    level_data: TankLevelUpdate,
    service: TankService = Depends(get_tank_service),
) -> TankResponse:
    """
    Update tank level from DCS reading.

    This endpoint is called by the DCS integration to update real-time levels.
    """
    tank = await service.update_level(tank_id, level_data)
    if not tank:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tank {tank_id} not found",
        )
    return TankResponse.model_validate(tank)


@router.put("/{tank_id}/contents", response_model=TankResponse)
async def update_tank_contents(
    tank_id: str,
    contents_data: TankContentsUpdate,
    service: TankService = Depends(get_tank_service),
) -> TankResponse:
    """Update tank contents information."""
    tank = await service.update_contents(tank_id, contents_data)
    if not tank:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tank {tank_id} not found",
        )
    return TankResponse.model_validate(tank)


@router.post("/{tank_id}/reserve", response_model=TankResponse)
async def reserve_tank(
    tank_id: str,
    blend_id: str = Query(..., description="Blend ID to reserve for"),
    service: TankService = Depends(get_tank_service),
) -> TankResponse:
    """Reserve a tank for a blend operation."""
    tank = await service.reserve_tank(tank_id, blend_id)
    if not tank:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Tank {tank_id} not found or not available",
        )
    return TankResponse.model_validate(tank)


@router.post("/{tank_id}/release", response_model=TankResponse)
async def release_tank(
    tank_id: str,
    service: TankService = Depends(get_tank_service),
) -> TankResponse:
    """Release a tank from reservation."""
    tank = await service.release_tank(tank_id)
    if not tank:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tank {tank_id} not found",
        )
    return TankResponse.model_validate(tank)


@router.get("/material/{material_code}/available", response_model=list[TankResponse])
async def get_tanks_with_material(
    material_code: str,
    required_volume: float = Query(..., gt=0, description="Required volume in liters"),
    service: TankService = Depends(get_tank_service),
) -> list[TankResponse]:
    """Find tanks with sufficient material for blending."""
    tanks = await service.get_available_for_material(material_code, required_volume)
    return [TankResponse.model_validate(t) for t in tanks]
