"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from lobp import __version__
from lobp.db import get_db
from lobp.schemas.common import HealthCheck

router = APIRouter()


@router.get("", response_model=HealthCheck)
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthCheck:
    """
    Health check endpoint for monitoring.

    Checks database connectivity and returns system status.
    """
    # Check database connection
    try:
        await db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"

    # TODO: Add Redis health check
    redis_status = "not_configured"

    return HealthCheck(
        status="healthy" if db_status == "healthy" else "degraded",
        version=__version__,
        database=db_status,
        redis=redis_status,
    )


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Kubernetes readiness probe."""
    return {"status": "ready"}


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}
