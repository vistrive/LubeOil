"""Pydantic schemas for API request/response validation."""

from lobp.schemas.recipe import (
    RecipeCreate,
    RecipeIngredientCreate,
    RecipeIngredientResponse,
    RecipeResponse,
    RecipeUpdate,
)
from lobp.schemas.tank import TankCreate, TankResponse, TankUpdate
from lobp.schemas.blend import BlendCreate, BlendResponse, BlendUpdate
from lobp.schemas.quality import (
    QualityMeasurementCreate,
    QualityMeasurementResponse,
    QualityPredictionCreate,
    QualityPredictionResponse,
)
from lobp.schemas.common import Message, PaginatedResponse
from lobp.schemas.batch_history import (
    BatchHistoryCreate,
    BatchHistoryResponse,
    BatchHistoryUpdate,
    BatchHistorySummary,
    BatchHistoryCSVRow,
    QualityResultCSVRow,
    RawMaterialDataCreate,
    RawMaterialDataResponse,
    BlendQualityResultCreate,
    BlendQualityResultResponse,
)
from lobp.schemas.supplier import (
    SupplierCreate,
    SupplierResponse,
    SupplierUpdate,
    SupplierPriceCreate,
    SupplierPriceResponse,
    SupplierPriceUpdate,
    PriceHistoryCreate,
    PriceHistoryResponse,
    MaterialPriceComparison,
    CostDataCSVRow,
)
from lobp.schemas.inventory import (
    MaterialCreate,
    MaterialResponse,
    MaterialUpdate,
    MaterialLotCreate,
    MaterialLotResponse,
    MaterialLotUpdate,
    MaterialWithLots,
    LotSelectionRequest,
    LotSelectionResponse,
)
from lobp.schemas.ai_models import (
    AIModelVersionCreate,
    AIModelVersionResponse,
    AIModelVersionUpdate,
    ModelTrainingRunCreate,
    ModelTrainingRunResponse,
    ModelPredictionLogCreate,
    ModelPredictionLogResponse,
    ModelPredictionLogVerify,
    RetrainingScheduleCreate,
    RetrainingScheduleResponse,
    RetrainingScheduleUpdate,
    ModelPerformanceSummary,
)

__all__ = [
    # Recipe
    "RecipeCreate",
    "RecipeUpdate",
    "RecipeResponse",
    "RecipeIngredientCreate",
    "RecipeIngredientResponse",
    # Tank
    "TankCreate",
    "TankUpdate",
    "TankResponse",
    # Blend
    "BlendCreate",
    "BlendUpdate",
    "BlendResponse",
    # Quality
    "QualityMeasurementCreate",
    "QualityMeasurementResponse",
    "QualityPredictionCreate",
    "QualityPredictionResponse",
    # Common
    "Message",
    "PaginatedResponse",
    # Batch History
    "BatchHistoryCreate",
    "BatchHistoryResponse",
    "BatchHistoryUpdate",
    "BatchHistorySummary",
    "BatchHistoryCSVRow",
    "QualityResultCSVRow",
    "RawMaterialDataCreate",
    "RawMaterialDataResponse",
    "BlendQualityResultCreate",
    "BlendQualityResultResponse",
    # Supplier
    "SupplierCreate",
    "SupplierResponse",
    "SupplierUpdate",
    "SupplierPriceCreate",
    "SupplierPriceResponse",
    "SupplierPriceUpdate",
    "PriceHistoryCreate",
    "PriceHistoryResponse",
    "MaterialPriceComparison",
    "CostDataCSVRow",
    # Inventory
    "MaterialCreate",
    "MaterialResponse",
    "MaterialUpdate",
    "MaterialLotCreate",
    "MaterialLotResponse",
    "MaterialLotUpdate",
    "MaterialWithLots",
    "LotSelectionRequest",
    "LotSelectionResponse",
    # AI Models
    "AIModelVersionCreate",
    "AIModelVersionResponse",
    "AIModelVersionUpdate",
    "ModelTrainingRunCreate",
    "ModelTrainingRunResponse",
    "ModelPredictionLogCreate",
    "ModelPredictionLogResponse",
    "ModelPredictionLogVerify",
    "RetrainingScheduleCreate",
    "RetrainingScheduleResponse",
    "RetrainingScheduleUpdate",
    "ModelPerformanceSummary",
]
