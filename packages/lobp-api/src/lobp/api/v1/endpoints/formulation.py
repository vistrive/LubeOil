"""Recipe formulation and blending calculator API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.db import get_db
from lobp.services.formulation_service import FormulationService
from lobp.services.excel_import_service import parse_recipe_excel
from lobp.services.seed_data import seed_all

router = APIRouter()


# --- Request/Response schemas ---

class FormulationRequest(BaseModel):
    """Request to generate a new formulation."""
    target_viscosity_40c: float | None = Field(None, gt=0, description="Target KV@40C in cSt")
    target_viscosity_100c: float | None = Field(None, gt=0, description="Target KV@100C in cSt")
    max_components: int = Field(4, ge=2, le=8)
    max_additive_percent: float = Field(25.0, ge=1, le=50)
    iterations: int = Field(500, ge=100, le=5000)


class IngredientInput(BaseModel):
    """Single ingredient for property calculation."""
    material_code: str
    weight_percent: float = Field(..., gt=0, le=100)


class PropertyCalcRequest(BaseModel):
    """Request to calculate blend properties."""
    ingredients: list[IngredientInput] = Field(..., min_length=1)


# --- Endpoints ---

def get_service(db: AsyncSession = Depends(get_db)) -> FormulationService:
    return FormulationService(db)


@router.post("/formulate")
async def formulate_recipe(
    request: FormulationRequest,
    service: FormulationService = Depends(get_service),
) -> dict:
    """
    Generate recipe candidates for given target viscosity specifications.

    Uses the Walther/ASTM D341 blending equation and iterative optimization
    to find blend combinations that meet the targets.
    """
    if not request.target_viscosity_40c and not request.target_viscosity_100c:
        raise HTTPException(400, "At least one target viscosity must be specified")

    return await service.formulate(
        target_viscosity_40c=request.target_viscosity_40c,
        target_viscosity_100c=request.target_viscosity_100c,
        max_components=request.max_components,
        max_additive_pct=request.max_additive_percent,
        iterations=request.iterations,
    )


@router.post("/calculate-properties")
async def calculate_properties(
    request: PropertyCalcRequest,
    service: FormulationService = Depends(get_service),
) -> dict:
    """
    Calculate predicted blend properties for a set of ingredients.

    Uses the Walther equation for viscosity and industry correlations
    for flash point, pour point, and density.
    """
    ingredients = [ing.model_dump() for ing in request.ingredients]
    return await service.calculate_recipe_properties(ingredients)


@router.get("/materials")
async def list_available_materials(
    service: FormulationService = Depends(get_service),
) -> list[dict]:
    """List all active materials available for formulation."""
    return await service.get_available_materials()


@router.post("/import-excel")
async def import_recipes_from_excel(
    file: UploadFile = File(..., description="Excel file with recipe data"),
) -> dict:
    """
    Parse recipes from an uploaded Excel file.

    Expected format: Product | BO/AD | Component | Description | Wt % | KV 40 | KV 100
    """
    if not file.filename or not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(400, "File must be an Excel file (.xlsx)")

    content = await file.read()
    try:
        recipes = parse_recipe_excel(content)
    except Exception as e:
        raise HTTPException(400, f"Failed to parse Excel: {e}")

    return {"recipes_parsed": len(recipes), "recipes": recipes}


@router.post("/seed-data")
async def seed_database(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Seed database with material and recipe data from recipe.xlsx."""
    result = await seed_all(db)
    return {"status": "success", **result}
