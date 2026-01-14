"""Tank service for managing storage vessels and inventory."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.core.config import settings
from lobp.models.tank import Tank, TankStatus, TankType
from lobp.schemas.tank import (
    TankContentsUpdate,
    TankCreate,
    TankLevelUpdate,
    TankUpdate,
)


class TankService:
    """Service for tank management operations."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        tank_type: TankType | None = None,
        status: TankStatus | None = None,
    ) -> list[Tank]:
        """Get all tanks with optional filtering."""
        query = select(Tank)

        if tank_type:
            query = query.where(Tank.tank_type == tank_type)
        if status:
            query = query.where(Tank.status == status)

        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_by_id(self, tank_id: str) -> Tank | None:
        """Get a tank by ID."""
        query = select(Tank).where(Tank.id == tank_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_by_tag(self, tag: str) -> Tank | None:
        """Get a tank by tag."""
        query = select(Tank).where(Tank.tag == tag)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def create(self, tank_data: TankCreate) -> Tank:
        """Create a new tank."""
        tank = Tank(
            tag=tank_data.tag,
            name=tank_data.name,
            description=tank_data.description,
            location=tank_data.location,
            tank_type=tank_data.tank_type,
            capacity_liters=tank_data.capacity_liters,
            diameter_meters=tank_data.diameter_meters,
            height_meters=tank_data.height_meters,
            high_high_level=tank_data.high_high_level,
            high_level=tank_data.high_level,
            low_level=tank_data.low_level,
            low_low_level=tank_data.low_low_level,
            max_temperature=tank_data.max_temperature,
            min_temperature=tank_data.min_temperature,
            has_heating=tank_data.has_heating,
            has_agitator=tank_data.has_agitator,
            has_level_indicator=tank_data.has_level_indicator,
            has_temperature_indicator=tank_data.has_temperature_indicator,
            segregation_group=tank_data.segregation_group,
        )

        self.db.add(tank)
        await self.db.commit()
        await self.db.refresh(tank)
        return tank

    async def update(self, tank_id: str, tank_data: TankUpdate) -> Tank | None:
        """Update a tank configuration."""
        tank = await self.get_by_id(tank_id)
        if not tank:
            return None

        update_data = tank_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(tank, field, value)

        tank.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(tank)
        return tank

    async def update_level(self, tank_id: str, level_data: TankLevelUpdate) -> Tank | None:
        """Update tank level from DCS reading."""
        tank = await self.get_by_id(tank_id)
        if not tank:
            return None

        tank.current_level_liters = level_data.current_level_liters
        tank.current_level_percent = level_data.current_level_percent
        if level_data.current_temperature_celsius is not None:
            tank.current_temperature_celsius = level_data.current_temperature_celsius
        if level_data.current_pressure_bar is not None:
            tank.current_pressure_bar = level_data.current_pressure_bar

        tank.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(tank)
        return tank

    async def update_contents(
        self, tank_id: str, contents_data: TankContentsUpdate
    ) -> Tank | None:
        """Update tank contents information."""
        tank = await self.get_by_id(tank_id)
        if not tank:
            return None

        update_data = contents_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(tank, field, value)

        tank.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(tank)
        return tank

    async def get_available_for_material(
        self, material_code: str, required_volume: float
    ) -> list[Tank]:
        """Find tanks with sufficient material for blending."""
        query = (
            select(Tank)
            .where(Tank.material_code == material_code)
            .where(Tank.status == TankStatus.AVAILABLE)
            .where(Tank.current_level_liters >= required_volume)
            .order_by(Tank.fifo_priority.desc())
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_available_blend_tanks(self, required_capacity: float) -> list[Tank]:
        """Find available blending tanks with sufficient capacity."""
        query = (
            select(Tank)
            .where(Tank.tank_type == TankType.BLEND)
            .where(Tank.status == TankStatus.AVAILABLE)
            .where(Tank.capacity_liters >= required_capacity)
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def reserve_tank(self, tank_id: str, blend_id: str) -> Tank | None:
        """Reserve a tank for a blend operation."""
        tank = await self.get_by_id(tank_id)
        if not tank or tank.status != TankStatus.AVAILABLE:
            return None

        tank.status = TankStatus.RESERVED
        tank.reserved_for_blend_id = blend_id
        tank.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(tank)
        return tank

    async def release_tank(self, tank_id: str) -> Tank | None:
        """Release a tank from reservation."""
        tank = await self.get_by_id(tank_id)
        if not tank:
            return None

        tank.status = TankStatus.AVAILABLE
        tank.reserved_for_blend_id = None
        tank.updated_at = datetime.now(timezone.utc)
        await self.db.commit()
        await self.db.refresh(tank)
        return tank

    async def check_low_stock_alerts(self) -> list[Tank]:
        """Find tanks with low stock levels."""
        query = (
            select(Tank)
            .where(Tank.tank_type.in_([TankType.BASE_OIL, TankType.ADDITIVE]))
            .where(
                Tank.current_level_percent
                <= settings.tank_low_level_threshold * 100
            )
        )
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_inventory_summary(self) -> dict[str, list[dict]]:
        """Get inventory summary by material code."""
        tanks = await self.get_all(
            tank_type=None, status=TankStatus.AVAILABLE
        )

        summary: dict[str, list[dict]] = {}
        for tank in tanks:
            if tank.material_code:
                if tank.material_code not in summary:
                    summary[tank.material_code] = []
                summary[tank.material_code].append({
                    "tank_tag": tank.tag,
                    "volume_liters": tank.current_level_liters,
                    "usable_volume": tank.usable_volume,
                    "fifo_priority": tank.fifo_priority,
                })

        return summary
