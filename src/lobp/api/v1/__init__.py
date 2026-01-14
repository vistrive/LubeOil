"""API v1 module."""

from fastapi import APIRouter

from lobp.api.v1.endpoints import blends, health, quality, recipes, tanks

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(recipes.router, prefix="/recipes", tags=["recipes"])
api_router.include_router(tanks.router, prefix="/tanks", tags=["tanks"])
api_router.include_router(blends.router, prefix="/blends", tags=["blends"])
api_router.include_router(quality.router, prefix="/quality", tags=["quality"])
