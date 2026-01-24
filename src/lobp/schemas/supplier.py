"""Pydantic schemas for Supplier and Pricing models."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class SupplierBase(BaseModel):
    """Base schema for suppliers."""

    code: str = Field(..., min_length=1, max_length=50)
    name: str = Field(..., min_length=1, max_length=200)
    legal_name: str | None = Field(None, max_length=200)
    description: str | None = None

    # Contact
    address: str | None = None
    city: str | None = Field(None, max_length=100)
    country: str | None = Field(None, max_length=100)
    postal_code: str | None = Field(None, max_length=20)
    phone: str | None = Field(None, max_length=50)
    email: str | None = Field(None, max_length=200)
    website: str | None = Field(None, max_length=200)

    # Primary contact
    contact_name: str | None = Field(None, max_length=200)
    contact_phone: str | None = Field(None, max_length=50)
    contact_email: str | None = Field(None, max_length=200)

    # Status
    is_qualified: bool = False
    is_active: bool = True
    is_preferred: bool = False

    # Payment
    payment_terms: str | None = Field(None, max_length=100)
    currency: str = Field(default="USD", max_length=3)


class SupplierCreate(SupplierBase):
    """Schema for creating a supplier."""

    pass


class SupplierUpdate(BaseModel):
    """Schema for updating a supplier."""

    name: str | None = Field(None, max_length=200)
    legal_name: str | None = Field(None, max_length=200)
    description: str | None = None
    address: str | None = None
    city: str | None = Field(None, max_length=100)
    country: str | None = Field(None, max_length=100)
    phone: str | None = Field(None, max_length=50)
    email: str | None = Field(None, max_length=200)
    contact_name: str | None = Field(None, max_length=200)
    contact_phone: str | None = Field(None, max_length=50)
    contact_email: str | None = Field(None, max_length=200)
    is_qualified: bool | None = None
    is_active: bool | None = None
    is_preferred: bool | None = None
    quality_rating: float | None = Field(None, ge=0, le=100)
    delivery_rating: float | None = Field(None, ge=0, le=100)


class SupplierResponse(SupplierBase):
    """Response schema for suppliers."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    qualification_date: datetime | None
    qualification_expiry: datetime | None
    quality_rating: float | None
    delivery_rating: float | None
    total_orders: int
    on_time_delivery_percent: float | None
    quality_rejection_percent: float | None
    average_lead_time_days: float | None
    credit_limit: float | None
    notes: str | None
    created_at: datetime
    updated_at: datetime


class SupplierPriceBase(BaseModel):
    """
    Base schema for supplier pricing.

    Matches document format from Section 3.1C:
    Material_Code,Material_Name,Supplier,Cost_per_Unit,Unit_Type,
    Price_Valid_From,Price_Valid_To,Quality_Grade
    """

    material_code: str = Field(..., max_length=50)
    material_name: str = Field(..., max_length=200)
    cost_per_unit: float = Field(..., gt=0, description="Cost per unit (e.g., $5.42)")
    unit_type: str = Field(..., max_length=20, description="per_liter, per_kg, per_drum")
    currency: str = Field(default="USD", max_length=3)

    # Validity period
    valid_from: datetime = Field(..., description="Price valid from date")
    valid_to: datetime | None = Field(None, description="Price valid to date")
    is_current: bool = True

    # Quality grade (matches document)
    quality_grade: str | None = Field(None, max_length=50, description="e.g., Grade_A, Premium")

    # Quantity breaks
    min_order_quantity: float | None = Field(None, ge=0)
    max_order_quantity: float | None = Field(None, ge=0)
    quantity_break_1: float | None = Field(None, ge=0)
    price_break_1: float | None = Field(None, gt=0)
    quantity_break_2: float | None = Field(None, ge=0)
    price_break_2: float | None = Field(None, gt=0)

    # Lead time
    lead_time_days: int | None = Field(None, ge=0)

    # Contract
    contract_number: str | None = Field(None, max_length=100)
    contract_expiry: datetime | None = None


class SupplierPriceCreate(SupplierPriceBase):
    """Schema for creating supplier price."""

    supplier_id: str


class SupplierPriceUpdate(BaseModel):
    """Schema for updating supplier price."""

    cost_per_unit: float | None = Field(None, gt=0)
    valid_to: datetime | None = None
    is_current: bool | None = None
    quality_grade: str | None = Field(None, max_length=50)
    min_order_quantity: float | None = None
    lead_time_days: int | None = None
    notes: str | None = None


class SupplierPriceResponse(SupplierPriceBase):
    """Response schema for supplier price."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    supplier_id: str
    notes: str | None
    created_at: datetime
    updated_at: datetime


class PriceHistoryBase(BaseModel):
    """Base schema for price history tracking."""

    material_code: str = Field(..., max_length=50)
    effective_date: datetime
    cost_per_unit: float = Field(..., gt=0)
    unit_type: str = Field(..., max_length=20)
    currency: str = Field(default="USD", max_length=3)
    previous_price: float | None = None
    price_change_percent: float | None = None
    change_reason: str | None = Field(None, max_length=200)
    quality_grade: str | None = Field(None, max_length=50)
    source: str | None = Field(None, max_length=100)


class PriceHistoryCreate(PriceHistoryBase):
    """Schema for creating price history record."""

    supplier_id: str


class PriceHistoryResponse(PriceHistoryBase):
    """Response schema for price history."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    supplier_id: str
    created_at: datetime


class MaterialPriceComparison(BaseModel):
    """Schema for comparing prices across suppliers."""

    material_code: str
    material_name: str
    suppliers: list["SupplierPriceComparison"]
    lowest_price: float
    highest_price: float
    price_spread_percent: float
    recommended_supplier_id: str | None
    recommendation_reason: str | None


class SupplierPriceComparison(BaseModel):
    """Individual supplier in price comparison."""

    supplier_id: str
    supplier_code: str
    supplier_name: str
    cost_per_unit: float
    unit_type: str
    quality_grade: str | None
    lead_time_days: int | None
    is_qualified: bool
    is_preferred: bool
    total_score: float  # Weighted score considering price, quality, delivery


class CostDataCSVRow(BaseModel):
    """
    Schema for cost data CSV import matching document format.

    Matches Section 3.1C of the document:
    Material_Code,Material_Name,Supplier,Cost_per_Unit,Unit_Type,
    Price_Valid_From,Price_Valid_To,Quality_Grade
    """

    Material_Code: str
    Material_Name: str
    Supplier: str  # Supplier code
    Cost_per_Unit: float
    Unit_Type: str
    Price_Valid_From: str  # Date string
    Price_Valid_To: str | None = None  # Date string
    Quality_Grade: str | None = None
