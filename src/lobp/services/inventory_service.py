"""
Inventory service with FIFO enforcement and shelf-life tracking.

Implements:
- FIFO (First-In-First-Out) material selection
- Shelf-life monitoring and expiry alerts
- Material lot tracking and traceability
- Inventory reservations for blend planning
"""

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, or_, select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.inventory import (
    Material,
    MaterialCategory,
    MaterialLot,
    MaterialStatus,
    InventoryTransaction,
)


class InventoryService:
    """Service for inventory management with FIFO enforcement."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ==================== Material Management ====================

    async def get_material(self, material_id: str) -> Material | None:
        """Get material by ID."""
        query = (
            select(Material)
            .options(selectinload(Material.lots))
            .where(Material.id == material_id)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_material_by_code(self, code: str) -> Material | None:
        """Get material by code."""
        query = (
            select(Material)
            .options(selectinload(Material.lots))
            .where(Material.code == code)
        )
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_all_materials(
        self,
        category: MaterialCategory | None = None,
        active_only: bool = True,
    ) -> list[Material]:
        """Get all materials with optional filtering."""
        query = select(Material)
        if category:
            query = query.where(Material.category == category)
        if active_only:
            query = query.where(Material.is_active == True)
        query = query.order_by(Material.code)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    # ==================== FIFO Lot Selection ====================

    async def get_available_lots_fifo(
        self,
        material_code: str,
        required_quantity: float | None = None,
        exclude_expired: bool = True,
        exclude_quarantine: bool = True,
    ) -> list[MaterialLot]:
        """
        Get available lots for a material in FIFO order.

        Lots are ordered by:
        1. FIFO sequence (oldest first)
        2. Expiry date (nearest expiry first)
        3. Received date (oldest first)

        Args:
            material_code: Material code to find lots for
            required_quantity: Optional minimum quantity needed
            exclude_expired: Exclude expired lots
            exclude_quarantine: Exclude quarantined lots

        Returns:
            List of lots in FIFO order
        """
        # Build base query
        query = (
            select(MaterialLot)
            .join(Material)
            .where(Material.code == material_code)
            .where(MaterialLot.status == MaterialStatus.AVAILABLE)
            .where(MaterialLot.current_quantity_liters > MaterialLot.reserved_quantity_liters)
        )

        if exclude_expired:
            query = query.where(MaterialLot.expiry_date > datetime.now(timezone.utc))

        if exclude_quarantine:
            query = query.where(MaterialLot.status != MaterialStatus.QUARANTINE)

        # FIFO ordering: sequence, then expiry, then received date
        query = query.order_by(
            MaterialLot.fifo_sequence.asc(),
            MaterialLot.expiry_date.asc(),
            MaterialLot.received_date.asc(),
        )

        result = await self.db.execute(query)
        lots = list(result.scalars().all())

        # If quantity specified, ensure we have enough
        if required_quantity:
            total_available = sum(lot.available_quantity for lot in lots)
            if total_available < required_quantity:
                return []  # Not enough inventory

        return lots

    async def allocate_material_fifo(
        self,
        material_code: str,
        required_quantity: float,
        blend_id: str | None = None,
        reserve: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Allocate material from lots using FIFO.

        Returns allocation plan showing which lots to use.

        Args:
            material_code: Material to allocate
            required_quantity: Quantity needed in liters
            blend_id: Optional blend ID for reservation
            reserve: Whether to reserve the quantity

        Returns:
            List of allocations: [{"lot_id", "lot_number", "quantity", "tank_tag"}]
        """
        lots = await self.get_available_lots_fifo(
            material_code, required_quantity
        )

        if not lots:
            raise ValueError(
                f"Insufficient inventory for {material_code}: "
                f"need {required_quantity}L"
            )

        allocations = []
        remaining = required_quantity

        for lot in lots:
            if remaining <= 0:
                break

            available = lot.available_quantity
            allocate_qty = min(available, remaining)

            if allocate_qty > 0:
                allocation = {
                    "lot_id": lot.id,
                    "lot_number": lot.lot_number,
                    "quantity": allocate_qty,
                    "tank_tag": lot.tank_tag,
                    "tank_id": lot.tank_id,
                    "expiry_date": lot.expiry_date.isoformat() if lot.expiry_date else None,
                    "days_until_expiry": lot.days_until_expiry,
                    "unit_cost": lot.unit_cost,
                }
                allocations.append(allocation)

                if reserve:
                    lot.reserved_quantity_liters += allocate_qty

                remaining -= allocate_qty

        if remaining > 0:
            raise ValueError(
                f"Could only allocate {required_quantity - remaining}L "
                f"of {required_quantity}L required for {material_code}"
            )

        if reserve:
            await self.db.commit()

        return allocations

    async def release_reservation(
        self,
        lot_id: str,
        quantity: float,
    ) -> MaterialLot | None:
        """Release a reservation on a lot."""
        query = select(MaterialLot).where(MaterialLot.id == lot_id)
        result = await self.db.execute(query)
        lot = result.scalar_one_or_none()

        if lot:
            lot.reserved_quantity_liters = max(
                0, lot.reserved_quantity_liters - quantity
            )
            await self.db.commit()
            await self.db.refresh(lot)

        return lot

    async def consume_material(
        self,
        lot_id: str,
        quantity: float,
        blend_id: str,
        blend_batch_number: str,
        performed_by: str | None = None,
    ) -> InventoryTransaction:
        """
        Consume material from a lot (actual usage).

        Creates transaction record and updates lot quantity.
        """
        query = select(MaterialLot).where(MaterialLot.id == lot_id)
        result = await self.db.execute(query)
        lot = result.scalar_one_or_none()

        if not lot:
            raise ValueError(f"Lot {lot_id} not found")

        if lot.current_quantity_liters < quantity:
            raise ValueError(
                f"Insufficient quantity in lot {lot.lot_number}: "
                f"have {lot.current_quantity_liters}L, need {quantity}L"
            )

        # Create transaction
        transaction = InventoryTransaction(
            lot_id=lot_id,
            transaction_type="issue",
            quantity_liters=-quantity,
            quantity_before=lot.current_quantity_liters,
            quantity_after=lot.current_quantity_liters - quantity,
            blend_id=blend_id,
            blend_batch_number=blend_batch_number,
            unit_cost=lot.unit_cost,
            total_cost=(lot.unit_cost or 0) * quantity,
            performed_by=performed_by,
        )
        self.db.add(transaction)

        # Update lot
        lot.current_quantity_liters -= quantity
        lot.reserved_quantity_liters = max(
            0, lot.reserved_quantity_liters - quantity
        )

        if lot.current_quantity_liters <= 0:
            lot.status = MaterialStatus.DEPLETED

        await self.db.commit()
        await self.db.refresh(transaction)

        return transaction

    # ==================== Shelf Life Management ====================

    async def get_expiring_lots(
        self,
        days_threshold: int = 30,
    ) -> list[MaterialLot]:
        """Get lots expiring within threshold days."""
        expiry_cutoff = datetime.now(timezone.utc) + timedelta(days=days_threshold)

        query = (
            select(MaterialLot)
            .join(Material)
            .where(MaterialLot.status == MaterialStatus.AVAILABLE)
            .where(MaterialLot.current_quantity_liters > 0)
            .where(MaterialLot.expiry_date <= expiry_cutoff)
            .where(MaterialLot.expiry_date > datetime.now(timezone.utc))
            .order_by(MaterialLot.expiry_date.asc())
        )

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_expired_lots(self) -> list[MaterialLot]:
        """Get all expired lots with remaining quantity."""
        query = (
            select(MaterialLot)
            .join(Material)
            .where(MaterialLot.current_quantity_liters > 0)
            .where(MaterialLot.expiry_date <= datetime.now(timezone.utc))
            .where(MaterialLot.status != MaterialStatus.EXPIRED)
        )

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def mark_expired_lots(self) -> int:
        """Mark all expired lots and return count."""
        expired_lots = await self.get_expired_lots()

        for lot in expired_lots:
            lot.status = MaterialStatus.EXPIRED

        await self.db.commit()
        return len(expired_lots)

    async def get_shelf_life_report(self) -> dict[str, Any]:
        """Generate shelf life status report."""
        now = datetime.now(timezone.utc)

        # Get counts by status
        expired = await self.get_expired_lots()
        expiring_30 = await self.get_expiring_lots(30)
        expiring_60 = await self.get_expiring_lots(60)
        expiring_90 = await self.get_expiring_lots(90)

        # Calculate values at risk
        expired_value = sum(
            (lot.current_quantity_liters * (lot.unit_cost or 0))
            for lot in expired
        )
        expiring_30_value = sum(
            (lot.current_quantity_liters * (lot.unit_cost or 0))
            for lot in expiring_30
        )

        return {
            "report_date": now.isoformat(),
            "expired": {
                "count": len(expired),
                "total_liters": sum(lot.current_quantity_liters for lot in expired),
                "value_at_risk": expired_value,
                "lots": [
                    {
                        "lot_number": lot.lot_number,
                        "material_code": lot.material.code if lot.material else None,
                        "quantity": lot.current_quantity_liters,
                        "expired_date": lot.expiry_date.isoformat(),
                    }
                    for lot in expired[:10]  # Top 10
                ],
            },
            "expiring_30_days": {
                "count": len(expiring_30),
                "total_liters": sum(lot.current_quantity_liters for lot in expiring_30),
                "value_at_risk": expiring_30_value,
            },
            "expiring_60_days": {
                "count": len(expiring_60),
                "total_liters": sum(lot.current_quantity_liters for lot in expiring_60),
            },
            "expiring_90_days": {
                "count": len(expiring_90),
                "total_liters": sum(lot.current_quantity_liters for lot in expiring_90),
            },
        }

    # ==================== Inventory Levels ====================

    async def get_inventory_summary(
        self,
        material_code: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get inventory summary by material."""
        query = (
            select(
                Material.code,
                Material.name,
                Material.category,
                Material.reorder_point_liters,
                func.sum(MaterialLot.current_quantity_liters).label("total_quantity"),
                func.sum(MaterialLot.reserved_quantity_liters).label("reserved_quantity"),
                func.count(MaterialLot.id).label("lot_count"),
            )
            .join(MaterialLot, Material.id == MaterialLot.material_id)
            .where(MaterialLot.status == MaterialStatus.AVAILABLE)
            .where(MaterialLot.current_quantity_liters > 0)
            .group_by(
                Material.code,
                Material.name,
                Material.category,
                Material.reorder_point_liters,
            )
        )

        if material_code:
            query = query.where(Material.code == material_code)

        query = query.order_by(Material.code)
        result = await self.db.execute(query)

        summaries = []
        for row in result:
            available = (row.total_quantity or 0) - (row.reserved_quantity or 0)
            reorder_point = row.reorder_point_liters or 0

            summaries.append({
                "material_code": row.code,
                "material_name": row.name,
                "category": row.category,
                "total_quantity": row.total_quantity or 0,
                "reserved_quantity": row.reserved_quantity or 0,
                "available_quantity": available,
                "lot_count": row.lot_count,
                "reorder_point": reorder_point,
                "below_reorder": available < reorder_point if reorder_point else False,
            })

        return summaries

    async def get_low_stock_materials(self) -> list[dict[str, Any]]:
        """Get materials below reorder point."""
        summaries = await self.get_inventory_summary()
        return [s for s in summaries if s.get("below_reorder")]

    # ==================== Lot Management ====================

    async def receive_material(
        self,
        material_code: str,
        lot_number: str,
        quantity: float,
        received_date: datetime,
        expiry_date: datetime,
        tank_tag: str | None = None,
        supplier_lot: str | None = None,
        unit_cost: float | None = None,
        coa_number: str | None = None,
        performed_by: str | None = None,
    ) -> MaterialLot:
        """Receive new material lot into inventory."""
        material = await self.get_material_by_code(material_code)
        if not material:
            raise ValueError(f"Material {material_code} not found")

        # Get next FIFO sequence
        query = select(func.max(MaterialLot.fifo_sequence)).where(
            MaterialLot.material_id == material.id
        )
        result = await self.db.execute(query)
        max_seq = result.scalar() or 0

        lot = MaterialLot(
            material_id=material.id,
            lot_number=lot_number,
            supplier_lot_number=supplier_lot,
            tank_tag=tank_tag,
            received_quantity_liters=quantity,
            current_quantity_liters=quantity,
            received_date=received_date,
            expiry_date=expiry_date,
            fifo_sequence=max_seq + 1,
            unit_cost=unit_cost or material.standard_cost_per_liter,
            total_cost=(unit_cost or material.standard_cost_per_liter or 0) * quantity,
            coa_number=coa_number,
        )
        self.db.add(lot)

        # Create receipt transaction
        transaction = InventoryTransaction(
            lot_id=lot.id,
            transaction_type="receipt",
            quantity_liters=quantity,
            quantity_before=0,
            quantity_after=quantity,
            unit_cost=lot.unit_cost,
            total_cost=lot.total_cost,
            performed_by=performed_by,
        )
        self.db.add(transaction)

        await self.db.commit()
        await self.db.refresh(lot)

        return lot
