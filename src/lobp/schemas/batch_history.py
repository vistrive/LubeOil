"""Pydantic schemas for BatchHistory models - AI Recipe Optimization data format."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class RawMaterialDataBase(BaseModel):
    """Raw material data for a batch ingredient."""

    material_code: str = Field(..., max_length=50)
    material_name: str = Field(..., max_length=200)
    material_type: str = Field(..., max_length=100)
    lot_number: str | None = Field(None, max_length=100)
    supplier_code: str | None = Field(None, max_length=50)
    supplier_name: str | None = Field(None, max_length=200)

    # Quantity
    quantity_wt_percent: float = Field(..., ge=0, le=100)
    quantity_liters: float | None = Field(None, ge=0)
    quantity_kg: float | None = Field(None, ge=0)

    # Lot properties at time of use
    viscosity_40c_cst: float | None = Field(None, gt=0)
    viscosity_100c_cst: float | None = Field(None, gt=0)
    viscosity_index: float | None = None
    density_15c: float | None = Field(None, gt=0)
    tbn: float | None = Field(None, ge=0)
    tan: float | None = Field(None, ge=0)
    pour_point_c: float | None = None
    flash_point_c: float | None = None
    water_content_ppm: float | None = Field(None, ge=0)

    # Cost
    unit_cost: float | None = Field(None, ge=0)
    total_cost: float | None = Field(None, ge=0)

    # Processing
    addition_order: int = Field(default=1, ge=1)
    addition_temperature_c: float | None = None


class RawMaterialDataCreate(RawMaterialDataBase):
    """Schema for creating raw material batch data."""

    pass


class RawMaterialDataResponse(RawMaterialDataBase):
    """Response schema for raw material batch data."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    batch_history_id: str
    created_at: datetime


class BlendQualityResultBase(BaseModel):
    """Quality result at a point during blending."""

    measurement_type: str = Field(..., max_length=50)  # inline, lab_interim, lab_final
    measurement_time: datetime
    sample_id: str | None = Field(None, max_length=100)
    blend_progress_percent: float | None = Field(None, ge=0, le=100)
    current_volume_liters: float | None = Field(None, ge=0)

    # Quality parameters
    viscosity_40c_cst: float | None = Field(None, gt=0)
    viscosity_100c_cst: float | None = Field(None, gt=0)
    viscosity_index: float | None = None
    tbn_mgkoh: float | None = Field(None, ge=0)
    pour_point_c: float | None = None
    flash_point_c: float | None = None
    water_content_ppm: float | None = Field(None, ge=0)
    foam_test_ml: float | None = Field(None, ge=0)
    density_15c: float | None = Field(None, gt=0)

    on_spec: bool | None = None
    notes: str | None = None


class BlendQualityResultCreate(BlendQualityResultBase):
    """Schema for creating quality result."""

    pass


class BlendQualityResultResponse(BlendQualityResultBase):
    """Response schema for quality result."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    batch_history_id: str
    created_at: datetime


class BatchHistoryBase(BaseModel):
    """
    Base schema for batch history - matches AI Recipe Optimization document format.

    This schema is designed to store historical batch data in the exact format
    required for AI model training as specified in the document.
    """

    # Batch identification
    batch_id: str = Field(..., max_length=50, description="Unique batch ID (e.g., BLD20240115001)")
    blend_date: datetime = Field(..., description="Date of blend operation")
    recipe_code: str = Field(..., max_length=50)
    product_code: str | None = Field(None, max_length=50)

    # Volume
    blend_volume_liters: float = Field(..., gt=0, description="Total blend volume in liters")

    # Primary base oil properties (matches document CSV format)
    base_oil_type: str | None = Field(None, max_length=100, description="e.g., Paraffinic_32")
    base_oil_supplier: str | None = Field(None, max_length=100)
    base_oil_lot_number: str | None = Field(None, max_length=100)
    base_oil_viscosity_cst: float | None = Field(None, gt=0, description="Base oil viscosity @ 40°C")
    base_oil_tbn: float | None = Field(None, ge=0)
    base_oil_density: float | None = Field(None, gt=0)
    base_oil_percentage: float | None = Field(None, ge=0, le=100)

    # Additive 1 (matches document format: Additive1_Type, Additive1_Qty_wt%)
    additive1_type: str | None = Field(None, max_length=100, description="e.g., Detergent_PB")
    additive1_code: str | None = Field(None, max_length=50)
    additive1_qty_wt_percent: float | None = Field(None, ge=0, le=100)
    additive1_lot_number: str | None = Field(None, max_length=100)

    # Additive 2
    additive2_type: str | None = Field(None, max_length=100, description="e.g., Oxidant_Inhibitor")
    additive2_code: str | None = Field(None, max_length=50)
    additive2_qty_wt_percent: float | None = Field(None, ge=0, le=100)
    additive2_lot_number: str | None = Field(None, max_length=100)

    # Additive 3
    additive3_type: str | None = Field(None, max_length=100, description="e.g., AntiWear_ZDDP")
    additive3_code: str | None = Field(None, max_length=50)
    additive3_qty_wt_percent: float | None = Field(None, ge=0, le=100)
    additive3_lot_number: str | None = Field(None, max_length=100)

    # Additive 4
    additive4_type: str | None = Field(None, max_length=100, description="e.g., Dispersant")
    additive4_code: str | None = Field(None, max_length=50)
    additive4_qty_wt_percent: float | None = Field(None, ge=0, le=100)
    additive4_lot_number: str | None = Field(None, max_length=100)

    # Process parameters
    temperature_blending_c: float | None = Field(None, description="Blending temperature in °C")
    mixing_speed_rpm: float | None = Field(None, ge=0)
    mixing_duration_minutes: int | None = Field(None, ge=0)
    ambient_temperature_c: float | None = None
    humidity_percent: float | None = Field(None, ge=0, le=100)

    # Final quality results (matches document output format)
    viscosity_40c_cst: float | None = Field(None, gt=0, description="Final viscosity @ 40°C")
    viscosity_100c_cst: float | None = Field(None, gt=0, description="Final viscosity @ 100°C")
    viscosity_index: float | None = None
    tbn_mgkoh: float | None = Field(None, ge=0, description="Total Base Number")
    pour_point_c: float | None = None
    flash_point_c: float | None = None
    water_content_ppm: float | None = Field(None, ge=0)
    foam_test_ml: float | None = Field(None, ge=0)
    oxidation_stability_hours: float | None = Field(None, ge=0)
    density_15c: float | None = Field(None, gt=0)

    # Quality assessment (matches document: Off_Spec_Flag, Specification_Met)
    off_spec_flag: bool = Field(default=False, description="1 if off-spec, 0 if on-spec")
    specification_met: str = Field(default="PASS", description="PASS or FAIL")
    quality_status: str | None = Field(None, max_length=20)
    quality_notes: str | None = None

    # Target specifications for comparison
    target_viscosity_40c: float | None = None
    target_viscosity_100c: float | None = None
    target_tbn: float | None = None
    target_pour_point: float | None = None
    target_flash_point: float | None = None

    # Cost data
    total_material_cost: float | None = Field(None, ge=0)
    cost_per_liter: float | None = Field(None, ge=0)
    energy_consumed_kwh: float | None = Field(None, ge=0)
    energy_cost: float | None = Field(None, ge=0)
    labor_cost: float | None = Field(None, ge=0)
    total_batch_cost: float | None = Field(None, ge=0)

    # AI optimization tracking
    ai_optimized: bool = False
    ai_model_version: str | None = Field(None, max_length=50)
    ai_predicted_viscosity_40c: float | None = None
    ai_predicted_tbn: float | None = None
    ai_confidence: float | None = Field(None, ge=0, le=100)
    ai_cost_savings_percent: float | None = None


class BatchHistoryCreate(BatchHistoryBase):
    """Schema for creating batch history record."""

    blend_id: str = Field(..., description="Reference to source blend")


class BatchHistoryUpdate(BaseModel):
    """Schema for updating batch history record."""

    # Quality results (can be updated after lab analysis)
    viscosity_40c_cst: float | None = None
    viscosity_100c_cst: float | None = None
    viscosity_index: float | None = None
    tbn_mgkoh: float | None = None
    pour_point_c: float | None = None
    flash_point_c: float | None = None
    water_content_ppm: float | None = None
    foam_test_ml: float | None = None

    # Assessment
    off_spec_flag: bool | None = None
    specification_met: str | None = None
    quality_status: str | None = None
    quality_notes: str | None = None

    # Validation
    data_validated: bool | None = None
    validated_by: str | None = None
    validation_notes: str | None = None

    # ML training
    used_for_training: bool | None = None
    training_run_id: str | None = None
    excluded_reason: str | None = None


class BatchHistoryResponse(BatchHistoryBase):
    """Response schema for batch history."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    blend_id: str

    # Data quality
    data_complete: bool
    data_validated: bool
    validation_date: datetime | None
    validated_by: str | None

    # ML tracking
    used_for_training: bool
    training_run_id: str | None

    # Relationships
    raw_material_data: list[RawMaterialDataResponse] = []
    quality_results: list[BlendQualityResultResponse] = []

    created_at: datetime
    updated_at: datetime


class BatchHistoryCSVRow(BaseModel):
    """
    Schema for CSV import matching document format exactly.

    Matches the CSV format specified in Section 3 of the document:
    Batch_ID,Blend_Date,Base_Oil_Type,Base_Oil_Viscosity_cSt,Base_Oil_TBN,
    Additive1_Type,Additive1_Qty_wt%,...
    """

    # Required fields from document
    Batch_ID: str
    Blend_Date: str  # Will be parsed to datetime
    Base_Oil_Type: str
    Base_Oil_Viscosity_cSt: float
    Base_Oil_TBN: float | None = None
    Additive1_Type: str | None = None
    Additive1_Qty_wt_percent: float | None = Field(None, alias="Additive1_Qty_wt%")
    Additive2_Type: str | None = None
    Additive2_Qty_wt_percent: float | None = Field(None, alias="Additive2_Qty_wt%")
    Additive3_Type: str | None = None
    Additive3_Qty_wt_percent: float | None = Field(None, alias="Additive3_Qty_wt%")
    Additive4_Type: str | None = None
    Additive4_Qty_wt_percent: float | None = Field(None, alias="Additive4_Qty_wt%")
    Temperature_Blending_C: float | None = None
    Blend_Volume_L: float


class QualityResultCSVRow(BaseModel):
    """
    Schema for quality result CSV import matching document format.

    Matches Section 3.1B of the document:
    Batch_ID,Viscosity_40C_cSt,Viscosity_100C_cSt,Viscosity_Index,TBN_mgKOH,...
    """

    Batch_ID: str
    Viscosity_40C_cSt: float | None = None
    Viscosity_100C_cSt: float | None = None
    Viscosity_Index: float | None = None
    TBN_mgKOH: float | None = None
    Pour_Point_C: float | None = None
    Flash_Point_C: float | None = None
    Water_Content_ppm: float | None = None
    Foam_Test_ML: float | None = None
    Off_Spec_Flag: int | None = None  # 0 or 1
    Specification_Met: str | None = None  # PASS or FAIL


class BatchHistorySummary(BaseModel):
    """Summary statistics for batch history analysis."""

    total_batches: int
    date_range_start: datetime | None
    date_range_end: datetime | None
    on_spec_count: int
    off_spec_count: int
    on_spec_rate_percent: float
    avg_viscosity_40c: float | None
    std_viscosity_40c: float | None
    avg_cost_per_liter: float | None
    total_volume_liters: float
    recipes_count: int
    ai_optimized_count: int
