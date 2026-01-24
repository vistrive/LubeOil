"""Supplier and pricing database models."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from lobp.models.base import BaseModel


class Supplier(BaseModel):
    """
    Supplier master data for raw material vendors.

    Contains contact information, performance metrics,
    and qualification status for material suppliers.
    """

    __tablename__ = "suppliers"

    # Identification
    code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    legal_name: Mapped[str | None] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)

    # Contact information
    address: Mapped[str | None] = mapped_column(Text)
    city: Mapped[str | None] = mapped_column(String(100))
    country: Mapped[str | None] = mapped_column(String(100))
    postal_code: Mapped[str | None] = mapped_column(String(20))
    phone: Mapped[str | None] = mapped_column(String(50))
    email: Mapped[str | None] = mapped_column(String(200))
    website: Mapped[str | None] = mapped_column(String(200))

    # Primary contact
    contact_name: Mapped[str | None] = mapped_column(String(200))
    contact_phone: Mapped[str | None] = mapped_column(String(50))
    contact_email: Mapped[str | None] = mapped_column(String(200))

    # Qualification
    is_qualified: Mapped[bool] = mapped_column(Boolean, default=False)
    qualification_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    qualification_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    quality_rating: Mapped[float | None] = mapped_column(Float)  # 0-100
    delivery_rating: Mapped[float | None] = mapped_column(Float)  # 0-100

    # Performance metrics
    total_orders: Mapped[int] = mapped_column(Integer, default=0)
    on_time_delivery_percent: Mapped[float | None] = mapped_column(Float)
    quality_rejection_percent: Mapped[float | None] = mapped_column(Float)
    average_lead_time_days: Mapped[float | None] = mapped_column(Float)

    # Payment terms
    payment_terms: Mapped[str | None] = mapped_column(String(100))
    credit_limit: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str] = mapped_column(String(3), default="USD")

    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_preferred: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text)

    # Relationships
    prices: Mapped[list["SupplierPrice"]] = relationship(
        "SupplierPrice", back_populates="supplier"
    )
    price_history: Mapped[list["PriceHistory"]] = relationship(
        "PriceHistory", back_populates="supplier"
    )


class SupplierPrice(BaseModel):
    """
    Current pricing from suppliers for materials.

    Supports multi-supplier pricing with validity periods,
    quantity breaks, and quality grade distinctions.
    Format matches AI Recipe Optimization document requirements.
    """

    __tablename__ = "supplier_prices"
    __table_args__ = (
        UniqueConstraint(
            "material_code", "supplier_id", "quality_grade", "valid_from",
            name="uq_supplier_price_material_supplier_grade_date"
        ),
    )

    # Material identification
    material_code: Mapped[str] = mapped_column(String(50), index=True)
    material_name: Mapped[str] = mapped_column(String(200))

    # Supplier link
    supplier_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("suppliers.id"), index=True
    )

    # Pricing
    cost_per_unit: Mapped[float] = mapped_column(Float)
    unit_type: Mapped[str] = mapped_column(String(20))  # per_liter, per_kg, per_drum
    currency: Mapped[str] = mapped_column(String(3), default="USD")

    # Validity period (matches document format)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)

    # Quality grade
    quality_grade: Mapped[str | None] = mapped_column(String(50))  # Grade_A, Premium, etc.

    # Quantity breaks
    min_order_quantity: Mapped[float | None] = mapped_column(Float)
    max_order_quantity: Mapped[float | None] = mapped_column(Float)
    quantity_break_1: Mapped[float | None] = mapped_column(Float)
    price_break_1: Mapped[float | None] = mapped_column(Float)
    quantity_break_2: Mapped[float | None] = mapped_column(Float)
    price_break_2: Mapped[float | None] = mapped_column(Float)

    # Lead time
    lead_time_days: Mapped[int | None] = mapped_column(Integer)

    # Contract reference
    contract_number: Mapped[str | None] = mapped_column(String(100))
    contract_expiry: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # Notes
    notes: Mapped[str | None] = mapped_column(Text)

    # Relationships
    supplier: Mapped["Supplier"] = relationship("Supplier", back_populates="prices")


class PriceHistory(BaseModel):
    """
    Historical pricing records for trend analysis.

    Tracks all price changes over time for materials
    from each supplier for cost analysis and forecasting.
    """

    __tablename__ = "price_history"

    # Material and supplier
    material_code: Mapped[str] = mapped_column(String(50), index=True)
    supplier_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("suppliers.id"), index=True
    )

    # Price record
    effective_date: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    cost_per_unit: Mapped[float] = mapped_column(Float)
    unit_type: Mapped[str] = mapped_column(String(20))
    currency: Mapped[str] = mapped_column(String(3), default="USD")

    # Change tracking
    previous_price: Mapped[float | None] = mapped_column(Float)
    price_change_percent: Mapped[float | None] = mapped_column(Float)
    change_reason: Mapped[str | None] = mapped_column(String(200))

    # Quality grade
    quality_grade: Mapped[str | None] = mapped_column(String(50))

    # Source of price
    source: Mapped[str | None] = mapped_column(String(100))  # quote, invoice, contract

    # Relationships
    supplier: Mapped["Supplier"] = relationship("Supplier", back_populates="price_history")
