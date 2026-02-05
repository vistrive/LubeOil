# LOBP - Lube Oil Blending Plant Control System

## Product Documentation

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Introduction](#introduction)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [System Architecture](#system-architecture)
6. [Core Modules](#core-modules)
7. [AI/ML Capabilities](#aiml-capabilities)
8. [API Reference](#api-reference)
9. [User Interface](#user-interface)
10. [Database Models](#database-models)
11. [Configuration](#configuration)
12. [Getting Started](#getting-started)

---

## Executive Summary

LOBP (Lube Oil Blending Plant) is an **AI-powered industrial control system** designed to automate and optimize the production of lubricant oils. The system combines traditional plant control functionality with advanced machine learning capabilities to deliver:

- **Automated recipe management** for lubricant formulations
- **Real-time blending operations** monitoring and control
- **AI-driven quality prediction** and optimization
- **Multi-supplier ingredient sourcing** with cost optimization
- **Digital twin simulation** for scenario analysis
- **Predictive maintenance** and equipment health monitoring

The platform targets manufacturing operations in the lubricant oil industry, providing end-to-end automation for production scheduling, cost optimization, and quality assurance.

---

## Introduction

### What is LOBP?

LOBP is a comprehensive solution for managing lube oil blending plant operations. It integrates with Distributed Control Systems (DCS) to provide:

- Centralized recipe and formulation management
- Real-time monitoring of blending operations
- Automated quality control with AI predictions
- Intelligent scheduling and resource optimization
- Complete traceability from raw materials to finished products

### Target Users

- **Plant Operators**: Execute and monitor blending operations
- **Quality Engineers**: Review quality predictions and lab results
- **Production Planners**: Schedule blends and manage inventory
- **Process Engineers**: Optimize recipes and production parameters
- **Management**: Access production KPIs and reports

---

## Key Features

### 1. Recipe Management

- Create, edit, and version lubricant formulations
- Define precise ingredient specifications with tolerance ranges
- Multi-level approval workflow (draft, pending, approved, retired)
- Quality target specifications for 15+ parameters
- AI-powered recipe optimization for cost and quality balance

### 2. Blend Operations

- Real-time batch execution and progress tracking
- Priority-based scheduling (low, normal, high, urgent)
- Multi-stage status tracking (queued → mixing → sampling → completed)
- Automated ingredient sequencing and pump control
- Off-spec risk assessment with AI predictions
- Cost tracking (material, energy, labor)

### 3. Tank Management

- Real-time inventory monitoring across storage and blending tanks
- Low stock alerts and automatic reorder suggestions
- Tank reservation for blend operations
- Material compatibility tracking
- DCS integration for level updates

### 4. Quality Control

- Real-time quality prediction using AI models
- Lab sample management and result tracking
- Inline analyzer integration
- Off-spec early warning system
- Automatic blend corrections based on predictions
- Quality parameter trend analysis

### 5. Inventory & Supply Chain

- Multi-supplier material sourcing
- Price optimization with quantity breaks
- Quality grade tracking per supplier
- Lead time and availability management
- Material lot traceability

### 6. AI-Powered Optimization

- Neural network-based recipe optimization
- Predictive quality control with confidence intervals
- Digital twin simulation for what-if analysis
- Cross-recipe learning for new product development
- Natural language interface for operator commands
- Dynamic rescheduling based on plant events

### 7. Reporting & Analytics

- Production KPIs and dashboards
- Quality trend analysis
- Cost analysis and optimization reports
- Equipment utilization metrics
- Energy consumption tracking

---

## Technology Stack

### Backend

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | FastAPI | 0.109.0 |
| Server | Uvicorn (ASGI) | Latest |
| Language | Python | 3.11+ |
| Database | PostgreSQL | Latest |
| Async Driver | asyncpg | Latest |
| ORM | SQLAlchemy | 2.0.25 |
| Migrations | Alembic | 1.13.1 |
| Validation | Pydantic | 2.5.3 |
| Task Queue | Celery | 5.3.6 |
| Cache | Redis | 5.0.1 |

### AI/ML Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| ML Framework | TensorFlow | 2.15.0 |
| Data Processing | NumPy | 1.26.3 |
| Data Analysis | Pandas | 2.1.4 |
| ML Algorithms | scikit-learn | 1.4.0 |

### Frontend

| Component | Technology | Version |
|-----------|-----------|---------|
| Framework | React | 18.2.0 |
| Language | TypeScript | Latest |
| Build Tool | Vite | 5.0.11 |
| Routing | React Router | 6.21.1 |
| Data Fetching | TanStack Query | Latest |
| Charts | Recharts | 2.10.3 |
| State Management | Zustand | 4.4.7 |
| Styling | Tailwind CSS | 3.4.1 |
| Icons | Lucide React | 0.303.0 |

### Security & Monitoring

| Component | Technology |
|-----------|-----------|
| Authentication | python-jose + JWT |
| Password Hashing | passlib + bcrypt |
| Logging | structlog (JSON) |
| Metrics | Prometheus |
| WebSockets | websockets 12.0 |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LOBP Control System                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   React HMI  │────▶│  FastAPI     │────▶│  PostgreSQL  │        │
│  │  Dashboard   │◀────│  Backend     │◀────│   Database   │        │
│  └──────────────┘     └──────────────┘     └──────────────┘        │
│         │                    │                                      │
│         │              ┌─────┴─────┐                                │
│         │              │           │                                │
│         ▼              ▼           ▼                                │
│  ┌──────────────┐ ┌──────────┐ ┌──────────┐                        │
│  │  WebSocket   │ │  Redis   │ │  Celery  │                        │
│  │  Real-time   │ │  Cache   │ │  Tasks   │                        │
│  └──────────────┘ └──────────┘ └──────────┘                        │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    AI/ML Engine                              │   │
│  ├─────────────┬─────────────┬─────────────┬──────────────────┤   │
│  │  Quality    │   Recipe    │  Digital    │  Cross-Recipe    │   │
│  │  Predictor  │  Optimizer  │   Twin      │   Learning       │   │
│  ├─────────────┼─────────────┼─────────────┼──────────────────┤   │
│  │   Soft      │  Dynamic    │   NLP       │  Multi-Supplier  │   │
│  │  Sensors    │  Scheduler  │ Interface   │   Optimizer      │   │
│  └─────────────┴─────────────┴─────────────┴──────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 External Integrations                        │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │   DCS Integration  │  Lab Systems  │  ERP Connectors        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Project Structure

```
LubeOil/
├── frontend/                    # React HMI Dashboard
│   ├── src/
│   │   ├── pages/              # Page components
│   │   │   ├── Dashboard.tsx   # Main operations view
│   │   │   ├── Recipes.tsx     # Recipe management
│   │   │   ├── Tanks.tsx       # Tank inventory
│   │   │   ├── Blends.tsx      # Blend operations
│   │   │   └── Quality.tsx     # Quality metrics
│   │   ├── components/         # Reusable components
│   │   └── App.tsx             # Router configuration
│   ├── package.json
│   └── vite.config.ts
│
├── src/lobp/                    # Python Backend
│   ├── api/v1/endpoints/       # REST API endpoints
│   ├── models/                 # SQLAlchemy ORM models
│   ├── schemas/                # Pydantic validation schemas
│   ├── services/               # Business logic layer
│   ├── ai/                     # AI/ML modules
│   ├── core/                   # Configuration & security
│   ├── db/                     # Database connections
│   └── main.py                 # FastAPI application
│
├── alembic/                    # Database migrations
├── pyproject.toml              # Python dependencies
└── alembic.ini                 # Migration configuration
```

---

## Core Modules

### 1. Recipe Service

Manages lubricant formulation recipes with full lifecycle support.

**Capabilities:**
- CRUD operations for recipes and ingredients
- Version control and approval workflow
- Ingredient validation (percentages must sum to 100%)
- Quality specification management
- Production parameter configuration

**Key Functions:**
- `create_recipe()` - Create new formulation
- `approve_recipe()` - Move to production status
- `validate_recipe()` - Check ingredient constraints
- `get_approved_recipes()` - List production-ready recipes

### 2. Blend Service

Handles batch blending operations from creation to completion.

**Capabilities:**
- Blend creation from recipes or manual specification
- Status lifecycle management
- Progress tracking with DCS integration
- Ingredient addition sequencing
- Cost calculation and tracking

**Blend Status Flow:**
```
draft → queued → scheduled → in_progress → mixing → cooling
→ sampling → lab_analysis → quality_hold → completed → transferred
```

### 3. Tank Service

Manages tank inventory and material tracking.

**Capabilities:**
- Real-time level monitoring
- Material content tracking
- Tank reservation for blend operations
- Low stock alerts
- Blend tank availability

**Tank Types:**
- Storage tanks (raw materials)
- Blend tanks (production)
- Finished goods tanks

### 4. Quality Service

Handles quality measurements and AI predictions.

**Capabilities:**
- Lab result entry and tracking
- Inline analyzer integration
- AI prediction management
- Off-spec risk assessment
- Prediction verification

**Quality Parameters:**
- Viscosity (40°C, 100°C, Index)
- Flash Point
- Pour Point
- Density
- TBN (Total Base Number)
- TAN (Total Acid Number)
- Water Content
- Sulfur Content
- Foam Test
- Oxidation Stability
- Color

### 5. Supplier Service

Manages vendor relationships and pricing.

**Capabilities:**
- Supplier qualification tracking
- Multi-tier pricing management
- Performance metrics (on-time %, rejection %)
- Quality grade assignment
- Lead time tracking

### 6. Inventory Service

Tracks raw materials and lots.

**Capabilities:**
- Material master data management
- Lot tracking with expiration
- Supplier assignment
- Property tracking per lot
- Usage history

---

## AI/ML Capabilities

### 1. Quality Predictor

**Purpose:** Predicts blend quality parameters before production completion.

**Technology:**
- Multi-layer neural network
- ReLU activation functions
- Trained on historical batch data

**Outputs:**
- Predicted values for 6 key quality parameters
- 95% confidence intervals
- Off-spec risk percentage
- Risk factors and recommendations

**Parameters Predicted:**
| Parameter | Unit | Typical Range |
|-----------|------|---------------|
| Viscosity @ 40°C | cSt | 10-500 |
| Viscosity @ 100°C | cSt | 2-50 |
| Viscosity Index | - | 80-200 |
| Flash Point | °C | 150-300 |
| Pour Point | °C | -50 to +10 |
| TBN | mgKOH/g | 0-15 |

### 2. Recipe Optimizer

**Purpose:** Optimizes ingredient percentages for cost and quality balance.

**Capabilities:**
- Gradient-based optimization
- Multi-objective optimization (cost vs. quality)
- Constraint satisfaction (min/max percentages)
- Waste reduction targeting

**Optimization Weights:**
- Cost Weight: 0.0 to 1.0
- Quality Weight: 0.0 to 1.0
- Waste Reduction: 0.0 to 1.0

### 3. Multi-Supplier Optimizer

**Purpose:** Selects optimal suppliers for material sourcing.

**Considerations:**
- Price per unit across suppliers
- Quality grades offered
- Minimum order quantities
- Lead times
- Quantity discounts
- Supplier reliability scores

### 4. Digital Twin

**Purpose:** Virtual plant model for simulation and analysis.

**Capabilities:**
- Real-time state synchronization with actual plant
- What-if scenario simulation
- Bottleneck identification
- Equipment utilization analysis
- Training environment for operators

### 5. Soft Sensors

**Purpose:** Real-time quality estimation from process parameters.

**Technology:**
- Walther equation for viscosity estimation
- Temperature-density correlations
- Continuous calibration from lab results

**Benefits:**
- Reduced sampling frequency
- Earlier off-spec detection
- Process optimization feedback

### 6. Dynamic Scheduler

**Purpose:** Real-time schedule adaptation based on events.

**Event Types Handled:**
- Equipment failures
- Priority changes
- Material delays
- Quality holds
- Rush orders

**Outputs:**
- Updated schedule
- Impact analysis
- Alternative recommendations

### 7. Cross-Recipe Learning

**Purpose:** Transfer knowledge between similar recipes.

**Capabilities:**
- Pattern recognition across product families
- Ingredient interaction modeling
- Success rate analysis
- New recipe suggestions

### 8. NLP Interface

**Purpose:** Natural language commands for operators.

**Supported Intents:**
- Status queries ("What's the status of blend B001?")
- Control commands ("Start blend B002")
- Quality queries ("Show quality for tank T-101")
- Scheduling ("Schedule blend for tomorrow")

---

## API Reference

### Base URL

```
http://localhost:8000/api/v1
```

### Authentication

All endpoints require JWT authentication via Bearer token.

```http
Authorization: Bearer <token>
```

### Endpoints

#### Health Check

```http
GET /health
```

Returns system status and component health.

---

#### Recipes

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/recipes` | List all recipes |
| GET | `/recipes/approved` | List approved recipes |
| GET | `/recipes/{id}` | Get recipe by ID |
| GET | `/recipes/code/{code}` | Get recipe by code |
| POST | `/recipes` | Create new recipe |
| PATCH | `/recipes/{id}` | Update recipe |
| DELETE | `/recipes/{id}` | Delete recipe |
| POST | `/recipes/{id}/approve` | Approve recipe |
| GET | `/recipes/{id}/validate` | Validate recipe |

**Example - Create Recipe:**

```json
POST /recipes
{
  "code": "ENG-5W30-001",
  "name": "Engine Oil 5W-30",
  "product_type": "engine_oil",
  "batch_size_liters_standard": 10000,
  "ingredients": [
    {
      "material_code": "BASE-SN150",
      "material_name": "SN150 Base Oil",
      "ingredient_type": "base_oil",
      "target_percentage": 75.0,
      "min_percentage": 70.0,
      "max_percentage": 80.0
    }
  ],
  "quality_targets": {
    "viscosity_40c": 65.0,
    "viscosity_100c": 11.0,
    "flash_point": 220.0
  }
}
```

---

#### Tanks

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tanks` | List all tanks |
| GET | `/tanks/inventory` | Get inventory summary |
| GET | `/tanks/low-stock` | Get low stock alerts |
| GET | `/tanks/blend-tanks` | Get available blend tanks |
| GET | `/tanks/{id}` | Get tank by ID |
| GET | `/tanks/tag/{tag}` | Get tank by tag |
| POST | `/tanks` | Create tank |
| PATCH | `/tanks/{id}` | Update tank |
| PUT | `/tanks/{id}/level` | Update tank level |
| PUT | `/tanks/{id}/contents` | Update tank contents |
| POST | `/tanks/{id}/reserve` | Reserve tank |
| POST | `/tanks/{id}/release` | Release tank |
| GET | `/tanks/material/{code}/available` | Find tanks with material |

---

#### Blends

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/blends` | List all blends |
| GET | `/blends/queue` | Get active blend queue |
| GET | `/blends/{id}` | Get blend by ID |
| GET | `/blends/batch/{number}` | Get blend by batch number |
| POST | `/blends` | Create blend |
| POST | `/blends/from-recipe` | Create from recipe |
| PATCH | `/blends/{id}` | Update blend |
| PUT | `/blends/{id}/status` | Update status |
| PUT | `/blends/{id}/progress` | Update progress |
| POST | `/blends/{id}/hold` | Put on hold |
| POST | `/blends/{id}/start` | Start blend |
| POST | `/blends/{id}/complete` | Complete blend |
| GET | `/blends/{id}/off-spec-check` | Check off-spec risk |

**Example - Create Blend from Recipe:**

```json
POST /blends/from-recipe
{
  "recipe_id": "uuid-here",
  "target_volume_liters": 10000,
  "priority": "high",
  "blend_tank_id": "uuid-here",
  "destination_tank_id": "uuid-here",
  "planned_start_time": "2024-01-15T08:00:00Z"
}
```

---

#### Quality

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/quality/measurements` | List measurements |
| GET | `/quality/measurements/{id}` | Get measurement |
| POST | `/quality/measurements` | Create measurement |
| PATCH | `/quality/measurements/{id}` | Update measurement |
| GET | `/quality/measurements/blend/{id}/latest` | Get latest for blend |
| GET | `/quality/predictions/blend/{id}` | List predictions |
| GET | `/quality/predictions/{id}` | Get prediction |
| POST | `/quality/predictions` | Create prediction |
| POST | `/quality/predictions/{id}/verify` | Verify prediction |
| GET | `/quality/predictions/blend/{id}/latest` | Get latest prediction |

**Example - Create Quality Measurement:**

```json
POST /quality/measurements
{
  "blend_id": "uuid-here",
  "sample_id": "LAB-2024-0123",
  "source": "lab_analysis",
  "viscosity_40c": 65.2,
  "viscosity_100c": 11.1,
  "viscosity_index": 155,
  "flash_point": 222.0,
  "pour_point": -36.0,
  "density": 0.875
}
```

---

## User Interface

### Dashboard

The main operations view displaying:

- **KPI Cards**: Production volume, active blends, alerts count
- **Production Progress**: Chart showing blend completion rates
- **Blend Queue**: List of active and pending blends
- **Active Alarms**: System alerts requiring attention
- **Real-time Metrics**: Live plant statistics

### Recipe Management

- Recipe list with search and filtering
- Status indicators (draft, pending, approved)
- Quality specifications display
- Ingredient table with percentages
- Create/edit forms
- Approval workflow buttons

### Tank Inventory

- Tank grid with visual level indicators
- Material contents display
- Capacity utilization
- Low stock highlighting
- Tank status (available, in-use, reserved)
- Quick actions (reserve, release)

### Blend Operations

- Active blend queue with status
- Progress bars and timers
- AI optimization indicators
- Off-spec risk display
- Control buttons (start, pause, resume, complete)
- Ingredient addition tracking
- Cost summary

### Quality Control

- Measurement history table
- Quality parameter trend charts
- Off-spec notifications
- Lab sample tracking
- Prediction vs actual comparison
- AI confidence indicators

---

## Database Models

### Core Entities

#### Recipe

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| code | String | Unique recipe code |
| name | String | Recipe name |
| version | Integer | Version number |
| status | Enum | draft/pending/approved/retired |
| product_type | String | Product category |
| batch_size_liters_* | Float | Min/max/standard batch sizes |
| quality targets | Float | Target values for each parameter |
| quality tolerances | Float | Acceptable deviation ranges |
| mixing_* | Float | Production parameters |
| cost_optimization_weight | Float | AI optimization preference |

#### Blend

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| batch_number | String | Unique batch identifier |
| recipe_id | UUID | Reference to recipe |
| status | Enum | Current operation status |
| priority | Enum | low/normal/high/urgent |
| target_volume_liters | Float | Planned volume |
| actual_volume_liters | Float | Produced volume |
| blend_tank_id | UUID | Production tank |
| destination_tank_id | UUID | Storage tank |
| progress_* | Various | Progress tracking fields |
| cost_* | Float | Cost breakdown fields |
| ai_* | Various | AI optimization fields |

#### Tank

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| tag | String | DCS tag |
| name | String | Tank name |
| tank_type | Enum | storage/blend/finished |
| capacity_liters | Float | Maximum capacity |
| current_level_liters | Float | Current contents |
| current_material_code | String | Material in tank |
| status | Enum | available/in_use/reserved |
| low_level_threshold | Float | Alert threshold |

#### QualityMeasurement

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| blend_id | UUID | Reference to blend |
| sample_id | String | Lab sample identifier |
| source | Enum | inline/lab/soft_sensor |
| viscosity_* | Float | Viscosity measurements |
| flash_point | Float | Flash point value |
| pour_point | Float | Pour point value |
| density | Float | Density value |
| tbn/tan | Float | Acid/base numbers |
| certified | Boolean | Certification status |

#### QualityPrediction

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Primary key |
| blend_id | UUID | Reference to blend |
| model_version | String | AI model used |
| predicted_* | Float | Predicted values |
| confidence_* | Float | Confidence intervals |
| off_spec_risk | Float | Risk percentage |
| risk_factors | JSON | Detailed risk analysis |
| recommendations | JSON | Suggested actions |

---

## Configuration

### Environment Variables

```bash
# Application
APP_NAME=LOBP
APP_VERSION=1.0.0
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/lobp
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_TTL=3600

# Celery
CELERY_BROKER_URL=redis://localhost:6379/1

# AI Configuration
AI_MODEL_PATH=/models
AI_CONFIDENCE_THRESHOLD=0.85
AI_QUALITY_DEVIATION_THRESHOLD=0.05
AI_ADAPTIVE_LEARNING_ENABLED=true
AI_RETRAINING_INTERVAL_HOURS=24

# Safety
SAFETY_TANK_HIGH_LEVEL=0.95
SAFETY_TANK_LOW_LEVEL=0.10
SAFETY_ENERGY_OPTIMIZATION_TARGET=0.15
SAFETY_HAZOP_ENABLED=true

# Monitoring
PROMETHEUS_ENABLED=true
```

### AI Model Configuration

```python
# Quality Predictor
confidence_threshold = 0.85      # Minimum prediction confidence
deviation_threshold = 0.05       # Max acceptable quality deviation

# Recipe Optimizer
cost_weight = 0.5               # Cost optimization priority
quality_weight = 0.5            # Quality optimization priority
max_iterations = 100            # Optimization iterations

# Soft Sensors
calibration_interval = 3600     # Seconds between recalibrations
lab_feedback_weight = 0.3       # Weight given to lab corrections
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+

### Backend Setup

```bash
# Clone repository
git clone <repository-url>
cd LubeOil

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Setup database
alembic upgrade head

# Run server
uvicorn src.lobp.main:app --reload
```

### Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Running with Docker

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## Support

For issues and feature requests, please refer to the project's issue tracker.

---

*LOBP - Lube Oil Blending Plant Control System*
*Version 1.0.0*
