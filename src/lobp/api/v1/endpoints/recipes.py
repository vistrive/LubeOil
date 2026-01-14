"""Recipe API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.db import get_db
from lobp.models.recipe import RecipeStatus
from lobp.schemas.common import Message
from lobp.schemas.recipe import RecipeCreate, RecipeResponse, RecipeUpdate
from lobp.services.recipe_service import RecipeService

router = APIRouter()


def get_recipe_service(db: AsyncSession = Depends(get_db)) -> RecipeService:
    """Dependency for recipe service."""
    return RecipeService(db)


@router.get("", response_model=list[RecipeResponse])
async def list_recipes(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: RecipeStatus | None = None,
    service: RecipeService = Depends(get_recipe_service),
) -> list[RecipeResponse]:
    """
    Get all recipes.

    Supports filtering by status and pagination.
    """
    recipes = await service.get_all(skip=skip, limit=limit, status=status)
    return [RecipeResponse.model_validate(r) for r in recipes]


@router.get("/approved", response_model=list[RecipeResponse])
async def list_approved_recipes(
    service: RecipeService = Depends(get_recipe_service),
) -> list[RecipeResponse]:
    """Get all approved recipes available for blending."""
    recipes = await service.get_approved_recipes()
    return [RecipeResponse.model_validate(r) for r in recipes]


@router.get("/{recipe_id}", response_model=RecipeResponse)
async def get_recipe(
    recipe_id: str,
    service: RecipeService = Depends(get_recipe_service),
) -> RecipeResponse:
    """Get a recipe by ID."""
    recipe = await service.get_by_id(recipe_id)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recipe {recipe_id} not found",
        )
    return RecipeResponse.model_validate(recipe)


@router.get("/code/{code}", response_model=RecipeResponse)
async def get_recipe_by_code(
    code: str,
    service: RecipeService = Depends(get_recipe_service),
) -> RecipeResponse:
    """Get a recipe by code."""
    recipe = await service.get_by_code(code)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recipe with code {code} not found",
        )
    return RecipeResponse.model_validate(recipe)


@router.post("", response_model=RecipeResponse, status_code=status.HTTP_201_CREATED)
async def create_recipe(
    recipe_data: RecipeCreate,
    service: RecipeService = Depends(get_recipe_service),
) -> RecipeResponse:
    """
    Create a new recipe.

    Includes all ingredients with their target percentages.
    """
    # Check if code already exists
    existing = await service.get_by_code(recipe_data.code)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Recipe with code {recipe_data.code} already exists",
        )

    recipe = await service.create(recipe_data)
    return RecipeResponse.model_validate(recipe)


@router.patch("/{recipe_id}", response_model=RecipeResponse)
async def update_recipe(
    recipe_id: str,
    recipe_data: RecipeUpdate,
    service: RecipeService = Depends(get_recipe_service),
) -> RecipeResponse:
    """Update a recipe."""
    recipe = await service.update(recipe_id, recipe_data)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recipe {recipe_id} not found",
        )
    return RecipeResponse.model_validate(recipe)


@router.delete("/{recipe_id}", response_model=Message)
async def delete_recipe(
    recipe_id: str,
    service: RecipeService = Depends(get_recipe_service),
) -> Message:
    """Delete a recipe."""
    deleted = await service.delete(recipe_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recipe {recipe_id} not found",
        )
    return Message(message=f"Recipe {recipe_id} deleted successfully")


@router.post("/{recipe_id}/approve", response_model=RecipeResponse)
async def approve_recipe(
    recipe_id: str,
    approved_by: str = Query(..., description="Username of approver"),
    service: RecipeService = Depends(get_recipe_service),
) -> RecipeResponse:
    """Approve a recipe for production use."""
    # Validate ingredients total
    valid, total = await service.validate_ingredients_total(recipe_id)
    if not valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Recipe ingredients must sum to 100% (currently {total:.2f}%)",
        )

    recipe = await service.approve(recipe_id, approved_by)
    if not recipe:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Recipe {recipe_id} not found",
        )
    return RecipeResponse.model_validate(recipe)


@router.get("/{recipe_id}/validate", response_model=dict)
async def validate_recipe(
    recipe_id: str,
    service: RecipeService = Depends(get_recipe_service),
) -> dict:
    """Validate recipe ingredient percentages."""
    valid, total = await service.validate_ingredients_total(recipe_id)
    return {
        "valid": valid,
        "total_percentage": total,
        "message": "Valid" if valid else f"Ingredients sum to {total:.2f}%, expected 100%",
    }
