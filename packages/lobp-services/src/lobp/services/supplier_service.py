"""
Supplier integration service for procurement and inventory management.

Implements:
- Reorder point monitoring and alerts
- Supplier API integration (mock)
- Purchase order generation
- Delivery tracking
- Lead time optimization
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lobp.models.inventory import Material, MaterialLot, MaterialStatus


class OrderStatus(str, Enum):
    """Purchase order status."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass
class Supplier:
    """Supplier information."""

    id: str
    name: str
    code: str
    contact_email: str
    lead_time_days: int
    minimum_order_value: float
    payment_terms: str
    api_enabled: bool = False


@dataclass
class PurchaseOrder:
    """A purchase order."""

    order_id: str
    supplier_id: str
    order_date: datetime
    expected_delivery: datetime
    status: OrderStatus
    items: list[dict[str, Any]]
    total_value: float
    notes: str | None = None


@dataclass
class ReorderAlert:
    """Material reorder alert."""

    material_code: str
    material_name: str
    current_stock: float
    reorder_point: float
    days_of_stock: float
    recommended_order_qty: float
    priority: str  # low, medium, high, critical
    suppliers: list[str]


class SupplierService:
    """Service for supplier integration and procurement."""

    def __init__(self, db: AsyncSession):
        self.db = db

        # Mock supplier data (would be from database in production)
        self.suppliers = {
            "SUP001": Supplier(
                id="SUP001",
                name="Base Oil Corp",
                code="BOC",
                contact_email="orders@baseoilcorp.com",
                lead_time_days=7,
                minimum_order_value=5000,
                payment_terms="Net 30",
                api_enabled=True,
            ),
            "SUP002": Supplier(
                id="SUP002",
                name="Additive Solutions",
                code="ADS",
                contact_email="supply@additivesolutions.com",
                lead_time_days=14,
                minimum_order_value=2000,
                payment_terms="Net 45",
                api_enabled=True,
            ),
            "SUP003": Supplier(
                id="SUP003",
                name="Chemical Distributors Inc",
                code="CDI",
                contact_email="orders@chemdist.com",
                lead_time_days=5,
                minimum_order_value=1000,
                payment_terms="Net 30",
                api_enabled=False,
            ),
        }

        # Average daily consumption (would be calculated from history)
        self.avg_daily_consumption: dict[str, float] = {}

    async def check_reorder_points(self) -> list[ReorderAlert]:
        """
        Check all materials against reorder points.

        Returns list of materials needing reorder.
        """
        alerts = []

        # Get inventory summary
        query = (
            select(
                Material.code,
                Material.name,
                Material.reorder_point_liters,
                Material.minimum_order_quantity,
                Material.lead_time_days,
                func.sum(MaterialLot.current_quantity_liters).label("current_stock"),
            )
            .join(MaterialLot, Material.id == MaterialLot.material_id)
            .where(MaterialLot.status == MaterialStatus.AVAILABLE)
            .where(Material.reorder_point_liters.isnot(None))
            .group_by(
                Material.code,
                Material.name,
                Material.reorder_point_liters,
                Material.minimum_order_quantity,
                Material.lead_time_days,
            )
        )

        result = await self.db.execute(query)

        for row in result:
            current_stock = row.current_stock or 0
            reorder_point = row.reorder_point_liters or 0

            if current_stock <= reorder_point:
                # Calculate days of stock
                daily_usage = self.avg_daily_consumption.get(row.code, 100)
                days_of_stock = current_stock / daily_usage if daily_usage > 0 else 999

                # Determine priority
                if days_of_stock <= 3:
                    priority = "critical"
                elif days_of_stock <= 7:
                    priority = "high"
                elif days_of_stock <= 14:
                    priority = "medium"
                else:
                    priority = "low"

                # Calculate recommended order
                lead_time = row.lead_time_days or 7
                safety_days = 7
                recommended_qty = daily_usage * (lead_time + safety_days)
                min_order = row.minimum_order_quantity or 0
                recommended_qty = max(recommended_qty, min_order)

                # Get suppliers for this material
                suppliers = await self._get_material_suppliers(row.code)

                alerts.append(ReorderAlert(
                    material_code=row.code,
                    material_name=row.name,
                    current_stock=current_stock,
                    reorder_point=reorder_point,
                    days_of_stock=days_of_stock,
                    recommended_order_qty=recommended_qty,
                    priority=priority,
                    suppliers=suppliers,
                ))

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        alerts.sort(key=lambda a: priority_order.get(a.priority, 99))

        return alerts

    async def _get_material_suppliers(self, material_code: str) -> list[str]:
        """Get list of suppliers for a material."""
        # In production, this would query supplier-material mappings
        # For now, return mock data based on material type
        if "SN-" in material_code.upper() or "BASE" in material_code.upper():
            return ["SUP001", "SUP003"]
        elif "ADD" in material_code.upper() or "VI" in material_code.upper():
            return ["SUP002"]
        else:
            return ["SUP003"]

    async def create_purchase_order(
        self,
        supplier_id: str,
        items: list[dict[str, Any]],
        notes: str | None = None,
    ) -> PurchaseOrder:
        """
        Create a new purchase order.

        Args:
            supplier_id: Supplier to order from
            items: List of items [{material_code, quantity, unit_price}]
            notes: Optional notes
        """
        supplier = self.suppliers.get(supplier_id)
        if not supplier:
            raise ValueError(f"Supplier {supplier_id} not found")

        order_date = datetime.now(timezone.utc)
        expected_delivery = order_date + timedelta(days=supplier.lead_time_days)

        # Calculate total
        total_value = sum(
            item.get("quantity", 0) * item.get("unit_price", 0)
            for item in items
        )

        if total_value < supplier.minimum_order_value:
            raise ValueError(
                f"Order value ${total_value:.2f} below minimum "
                f"${supplier.minimum_order_value:.2f}"
            )

        order = PurchaseOrder(
            order_id=f"PO-{datetime.now().strftime('%Y%m%d')}-{str(uuid4())[:8].upper()}",
            supplier_id=supplier_id,
            order_date=order_date,
            expected_delivery=expected_delivery,
            status=OrderStatus.DRAFT,
            items=items,
            total_value=total_value,
            notes=notes,
        )

        return order

    async def submit_order(
        self,
        order: PurchaseOrder,
    ) -> dict[str, Any]:
        """
        Submit order to supplier.

        If supplier has API, sends electronically.
        Otherwise, marks for manual processing.
        """
        supplier = self.suppliers.get(order.supplier_id)
        if not supplier:
            raise ValueError(f"Supplier {order.supplier_id} not found")

        if supplier.api_enabled:
            # Mock API call
            response = await self._call_supplier_api(supplier, order)
            order.status = OrderStatus.SUBMITTED

            return {
                "order_id": order.order_id,
                "status": "submitted",
                "supplier_confirmation": response.get("confirmation_number"),
                "expected_delivery": order.expected_delivery.isoformat(),
                "method": "api",
            }
        else:
            order.status = OrderStatus.DRAFT
            return {
                "order_id": order.order_id,
                "status": "pending_manual",
                "message": f"Please email order to {supplier.contact_email}",
                "method": "manual",
            }

    async def _call_supplier_api(
        self,
        supplier: Supplier,
        order: PurchaseOrder,
    ) -> dict[str, Any]:
        """Mock supplier API call."""
        # In production, this would make actual HTTP requests
        return {
            "confirmation_number": f"CONF-{str(uuid4())[:8].upper()}",
            "status": "accepted",
            "estimated_ship_date": (
                datetime.now(timezone.utc) + timedelta(days=2)
            ).isoformat(),
        }

    async def check_supplier_inventory(
        self,
        supplier_id: str,
        material_code: str,
    ) -> dict[str, Any]:
        """
        Check material availability with supplier.

        Mock implementation - would call supplier API.
        """
        supplier = self.suppliers.get(supplier_id)
        if not supplier:
            raise ValueError(f"Supplier {supplier_id} not found")

        # Mock response
        return {
            "material_code": material_code,
            "supplier": supplier.name,
            "available": True,
            "stock_quantity": 50000,
            "unit_price": 2.50,
            "lead_time_days": supplier.lead_time_days,
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }

    async def get_price_comparison(
        self,
        material_code: str,
        quantity: float,
    ) -> list[dict[str, Any]]:
        """
        Get price comparison across suppliers.

        Returns quotes from multiple suppliers.
        """
        suppliers = await self._get_material_suppliers(material_code)
        quotes = []

        for supplier_id in suppliers:
            supplier = self.suppliers.get(supplier_id)
            if not supplier:
                continue

            # Mock pricing (would call APIs)
            base_price = 2.0 + (hash(material_code) % 100) / 100
            volume_discount = 0.95 if quantity > 10000 else 1.0

            quotes.append({
                "supplier_id": supplier_id,
                "supplier_name": supplier.name,
                "unit_price": base_price * volume_discount,
                "total_price": base_price * volume_discount * quantity,
                "lead_time_days": supplier.lead_time_days,
                "payment_terms": supplier.payment_terms,
                "minimum_order": supplier.minimum_order_value,
            })

        # Sort by total price
        quotes.sort(key=lambda q: q["total_price"])

        return quotes

    async def forecast_requirements(
        self,
        days_ahead: int = 30,
    ) -> list[dict[str, Any]]:
        """
        Forecast material requirements based on blend schedule.

        Returns list of materials with projected needs.
        """
        # In production, would analyze blend schedule and recipes
        # For now, use simple projection based on current consumption

        requirements = []
        alerts = await self.check_reorder_points()

        for alert in alerts:
            daily_usage = self.avg_daily_consumption.get(alert.material_code, 100)
            projected_usage = daily_usage * days_ahead

            requirements.append({
                "material_code": alert.material_code,
                "material_name": alert.material_name,
                "current_stock": alert.current_stock,
                "projected_usage": projected_usage,
                "projected_balance": alert.current_stock - projected_usage,
                "stockout_date": (
                    (datetime.now(timezone.utc) + timedelta(days=alert.days_of_stock)).isoformat()
                    if alert.days_of_stock < days_ahead else None
                ),
                "recommended_order": max(0, projected_usage - alert.current_stock + alert.reorder_point),
            })

        return requirements

    async def get_delivery_schedule(
        self,
        days_ahead: int = 14,
    ) -> list[dict[str, Any]]:
        """Get expected deliveries in the next N days."""
        # In production, would query purchase orders
        # Mock data for demonstration
        now = datetime.now(timezone.utc)

        return [
            {
                "order_id": "PO-20240115-ABC123",
                "supplier": "Base Oil Corp",
                "expected_date": (now + timedelta(days=2)).isoformat(),
                "materials": [
                    {"code": "SN-150", "quantity": 20000, "unit": "liters"},
                ],
                "status": "shipped",
            },
            {
                "order_id": "PO-20240112-DEF456",
                "supplier": "Additive Solutions",
                "expected_date": (now + timedelta(days=5)).isoformat(),
                "materials": [
                    {"code": "VI-IMPROVER-100", "quantity": 500, "unit": "liters"},
                    {"code": "DETERGENT-HD", "quantity": 300, "unit": "liters"},
                ],
                "status": "confirmed",
            },
        ]

    async def update_consumption_rates(self) -> None:
        """
        Update average daily consumption rates from history.

        Should be run periodically (daily).
        """
        # Calculate from last 30 days of blend consumption
        # In production, would query actual consumption data

        # Mock update
        self.avg_daily_consumption = {
            "SN-150": 2000,
            "SN-500": 1500,
            "VI-IMPROVER-100": 50,
            "DETERGENT-HD": 30,
            "PPD-100": 20,
        }
