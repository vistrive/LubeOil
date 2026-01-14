"""
LOBP Control System - Main FastAPI Application

AI-powered Lube Oil Blending Plant Control System
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from lobp import __version__
from lobp.api.v1 import api_router
from lobp.core.config import settings

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for startup and shutdown events."""
    # Startup
    logger.info(
        "Starting LOBP Control System",
        version=__version__,
        environment=settings.environment,
    )
    yield
    # Shutdown
    logger.info("Shutting down LOBP Control System")


def create_application() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="""
## LOBP Control System API

AI-powered Lube Oil Blending Plant Control System for:

- **Recipe Management**: Create and manage blend formulations
- **Tank Management**: Monitor tank levels and inventory
- **Blend Operations**: Execute and track blending batches
- **Quality Control**: Measurements and AI-powered predictions
- **AI Optimization**: Dynamic recipe adjustments and predictions

### Core Features

- Recipe optimization using neural networks
- Predictive quality control with auto-correction
- Pump sequencing and occupancy management
- Dynamic rerouting and rescheduling
- Multi-blend parallelism support

### Documentation

- [OpenAPI Spec](/openapi.json)
- [ReDoc](/redoc)
        """,
        openapi_url=f"{settings.api_v1_prefix}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API router
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    return app


# Create application instance
app = create_application()


@app.get("/")
async def root() -> dict:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": __version__,
        "docs": "/docs",
        "redoc": "/redoc",
        "api": settings.api_v1_prefix,
    }
