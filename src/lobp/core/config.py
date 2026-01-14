"""
Application configuration module.

Handles all configuration settings for the LOBP Control System including:
- Database connections
- Redis cache configuration
- AI model settings
- Safety thresholds
- Industry-specific parameters
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "LOBP Control System"
    app_version: str = "0.1.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # API
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8080"]

    # Security
    secret_key: str = Field(default="change-me-in-production-use-strong-secret")
    access_token_expire_minutes: int = 60 * 24  # 24 hours
    algorithm: str = "HS256"

    # Database
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://lobp:lobp@localhost:5432/lobp"
    )
    database_pool_size: int = 10
    database_max_overflow: int = 20

    # Redis
    redis_url: RedisDsn = Field(default="redis://localhost:6379/0")
    redis_cache_ttl: int = 300  # 5 minutes

    # Celery
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    # AI/ML Configuration
    ai_model_path: str = "./models"
    ai_prediction_confidence_threshold: float = 0.85
    ai_quality_deviation_threshold: float = 0.05  # 5% off-spec risk threshold
    ai_adaptive_learning_enabled: bool = True
    ai_model_retrain_interval_hours: int = 24

    # Process Control Thresholds
    tank_high_level_threshold: float = 0.80  # 80%
    tank_low_level_threshold: float = 0.20  # 20%
    tank_critical_high_threshold: float = 0.90  # 90%
    tank_critical_low_threshold: float = 0.10  # 10%

    # Blend Quality Specifications (default ranges)
    viscosity_tolerance_percent: float = 2.0
    flash_point_tolerance_celsius: float = 5.0
    pour_point_tolerance_celsius: float = 3.0

    # Energy Management
    energy_optimization_enabled: bool = True
    energy_savings_target_percent: float = 15.0

    # Safety Configuration
    safety_hazop_enabled: bool = True
    safety_auto_shutdown_enabled: bool = True
    safety_compliance_logging: bool = True

    # Monitoring
    prometheus_enabled: bool = True
    prometheus_port: int = 9090

    # Supply Chain
    supplier_api_timeout_seconds: int = 30
    inventory_forecast_days: int = 7


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
