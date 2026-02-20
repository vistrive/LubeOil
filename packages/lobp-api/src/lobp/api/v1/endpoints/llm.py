"""LLM-powered recipe analysis and chat API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from lobp.db import get_db
from lobp.llm import get_llm_provider, get_all_providers, LLMResponse
from lobp.llm.prompts import (
    SYSTEM_PROMPT,
    recipe_analysis_prompt,
    formulation_suggestion_prompt,
    quality_prediction_prompt,
)
from lobp.services.formulation_service import FormulationService

router = APIRouter()


# --- Schemas ---

class ChatRequest(BaseModel):
    """Free-form chat request to LLM."""
    message: str = Field(..., min_length=1, max_length=5000)
    provider: str | None = Field(None, description="LLM provider: openai, anthropic, ollama, sarvam")


class RecipeAnalysisRequest(BaseModel):
    """Request to analyze a recipe using LLM."""
    recipe_name: str
    application: str = ""
    target_viscosity_40c: float | None = None
    target_viscosity_100c: float | None = None
    ingredients: list[dict] = Field(default_factory=list)
    provider: str | None = None


class FormulationSuggestionRequest(BaseModel):
    """Request LLM to suggest a formulation."""
    application: str = Field(..., description="e.g. engine_oil, gear_oil, hydraulic_oil")
    target_viscosity_40c: float | None = None
    target_viscosity_100c: float | None = None
    provider: str | None = None


class QualityPredictionRequest(BaseModel):
    """Request LLM to predict blend quality."""
    recipe: dict
    blend_conditions: dict = Field(default_factory=dict)
    provider: str | None = None


class LLMChatResponse(BaseModel):
    """Standardized LLM response."""
    content: str
    provider: str
    model: str
    error: str | None = None


# --- Endpoints ---

@router.get("/providers")
async def list_providers() -> dict:
    """List all configured LLM providers and their availability."""
    return {"providers": get_all_providers()}


@router.post("/chat", response_model=LLMChatResponse)
async def chat(request: ChatRequest) -> LLMChatResponse:
    """
    Free-form chat with an LLM about lubricant formulation.

    Uses the lubricant expert system prompt for domain-specific responses.
    """
    try:
        provider = get_llm_provider(request.provider)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not provider.is_available():
        raise HTTPException(
            503, f"Provider '{request.provider or 'default'}' is not available. "
            "Check API keys in configuration."
        )

    result = await provider.generate(
        prompt=request.message,
        system_prompt=SYSTEM_PROMPT,
    )

    return LLMChatResponse(
        content=result.content,
        provider=result.provider,
        model=result.model,
        error=result.error,
    )


@router.post("/analyze-recipe", response_model=LLMChatResponse)
async def analyze_recipe(request: RecipeAnalysisRequest) -> LLMChatResponse:
    """Analyze a recipe formulation using LLM expertise."""
    try:
        provider = get_llm_provider(request.provider)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not provider.is_available():
        raise HTTPException(503, f"Provider not available")

    prompt = recipe_analysis_prompt({
        "name": request.recipe_name,
        "application": request.application,
        "target_viscosity_40c": request.target_viscosity_40c,
        "target_viscosity_100c": request.target_viscosity_100c,
        "ingredients": request.ingredients,
    })

    result = await provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
    return LLMChatResponse(
        content=result.content, provider=result.provider,
        model=result.model, error=result.error,
    )


@router.post("/suggest-formulation", response_model=LLMChatResponse)
async def suggest_formulation(
    request: FormulationSuggestionRequest,
    db: AsyncSession = Depends(get_db),
) -> LLMChatResponse:
    """Ask LLM to suggest a formulation using available materials."""
    try:
        provider = get_llm_provider(request.provider)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not provider.is_available():
        raise HTTPException(503, f"Provider not available")

    # Fetch available materials for context
    service = FormulationService(db)
    materials = await service.get_available_materials()

    prompt = formulation_suggestion_prompt(
        target_visc_40c=request.target_viscosity_40c,
        target_visc_100c=request.target_viscosity_100c,
        application=request.application,
        available_materials=materials,
    )

    result = await provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
    return LLMChatResponse(
        content=result.content, provider=result.provider,
        model=result.model, error=result.error,
    )


@router.post("/predict-quality", response_model=LLMChatResponse)
async def predict_quality(request: QualityPredictionRequest) -> LLMChatResponse:
    """Ask LLM to predict blend quality outcome."""
    try:
        provider = get_llm_provider(request.provider)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not provider.is_available():
        raise HTTPException(503, f"Provider not available")

    prompt = quality_prediction_prompt(request.recipe, request.blend_conditions)
    result = await provider.generate(prompt=prompt, system_prompt=SYSTEM_PROMPT)
    return LLMChatResponse(
        content=result.content, provider=result.provider,
        model=result.model, error=result.error,
    )
