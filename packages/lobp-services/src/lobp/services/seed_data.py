"""Seed data for materials and recipes from recipe.xlsx."""

from datetime import datetime, timezone
from typing import Any

import structlog

from lobp.models.inventory import Material, MaterialCategory
from lobp.models.recipe import (
    IngredientType,
    ProductApplication,
    Recipe,
    RecipeIngredient,
    RecipeStatus,
)

logger = structlog.get_logger()

# Base oil material properties (typical values from industry data)
MATERIAL_SEED_DATA: list[dict[str, Any]] = [
    {
        "code": "SN-500",
        "name": "SN 500",
        "description": "Solvent Neutral 500 - Group I base oil, high viscosity paraffinic base stock",
        "category": MaterialCategory.BASE_OIL_PARAFFINIC,
        "standard_viscosity_40c": 95.0,
        "standard_viscosity_100c": 11.0,
        "standard_viscosity_index": 95,
        "standard_density_15c": 0.885,
        "standard_flash_point": 250.0,
        "standard_pour_point": -6.0,
        "standard_cost_per_liter": 1.10,
        "primary_uom": "liter",
    },
    {
        "code": "BS-460",
        "name": "BS 460",
        "description": "Bright Stock 460 - Heavy paraffinic base oil, very high viscosity",
        "category": MaterialCategory.BASE_OIL_PARAFFINIC,
        "standard_viscosity_40c": 460.0,
        "standard_viscosity_100c": 32.0,
        "standard_viscosity_index": 95,
        "standard_density_15c": 0.900,
        "standard_flash_point": 300.0,
        "standard_pour_point": -6.0,
        "standard_cost_per_liter": 1.30,
        "primary_uom": "liter",
    },
    {
        "code": "HVI-60",
        "name": "HVI 60",
        "description": "High Viscosity Index 60 - Light base oil for low-viscosity blends",
        "category": MaterialCategory.BASE_OIL_PARAFFINIC,
        "standard_viscosity_40c": 12.0,
        "standard_viscosity_100c": 3.0,
        "standard_viscosity_index": 100,
        "standard_density_15c": 0.855,
        "standard_flash_point": 180.0,
        "standard_pour_point": -15.0,
        "standard_cost_per_liter": 0.95,
        "primary_uom": "liter",
    },
    {
        "code": "HVI-120",
        "name": "HVI 120",
        "description": "High Viscosity Index 120 - Medium base oil, versatile blending component",
        "category": MaterialCategory.BASE_OIL_PARAFFINIC,
        "standard_viscosity_40c": 24.0,
        "standard_viscosity_100c": 4.8,
        "standard_viscosity_index": 105,
        "standard_density_15c": 0.865,
        "standard_flash_point": 210.0,
        "standard_pour_point": -12.0,
        "standard_cost_per_liter": 1.00,
        "primary_uom": "liter",
    },
    {
        "code": "HVI-650",
        "name": "HVI 650",
        "description": "High Viscosity Index 650 - Heavy base oil from Pulau Bukom refinery",
        "category": MaterialCategory.BASE_OIL_PARAFFINIC,
        "standard_viscosity_40c": 130.0,
        "standard_viscosity_100c": 14.5,
        "standard_viscosity_index": 97,
        "standard_density_15c": 0.892,
        "standard_flash_point": 270.0,
        "standard_pour_point": -9.0,
        "standard_cost_per_liter": 1.25,
        "primary_uom": "liter",
    },
    {
        "code": "GTL-BO8",
        "name": "GTL BO8",
        "description": "Gas-to-Liquid Base Oil 8 - Synthetic-quality Group III base oil",
        "category": MaterialCategory.BASE_OIL_SYNTHETIC,
        "standard_viscosity_40c": 47.0,
        "standard_viscosity_100c": 8.0,
        "standard_viscosity_index": 130,
        "standard_density_15c": 0.835,
        "standard_flash_point": 240.0,
        "standard_pour_point": -18.0,
        "standard_cost_per_liter": 1.80,
        "primary_uom": "liter",
    },
    {
        "code": "EHC-110",
        "name": "EHC 110",
        "description": "EHC 110 (J500) - ExxonMobil extra-high quality base oil",
        "category": MaterialCategory.BASE_OIL_PARAFFINIC,
        "standard_viscosity_40c": 22.0,
        "standard_viscosity_100c": 4.5,
        "standard_viscosity_index": 108,
        "standard_density_15c": 0.860,
        "standard_flash_point": 220.0,
        "standard_pour_point": -15.0,
        "standard_cost_per_liter": 1.40,
        "primary_uom": "liter",
    },
    {
        "code": "ADD-PKG-45",
        "name": "Additives 4-5",
        "description": "Multi-functional additive package for industrial gear and circulating oils",
        "category": MaterialCategory.ADDITIVE_PACKAGE,
        "standard_viscosity_40c": 150.0,
        "standard_viscosity_100c": 15.0,
        "standard_density_15c": 0.950,
        "standard_flash_point": 180.0,
        "standard_tbn": 40.0,
        "standard_cost_per_liter": 4.50,
        "primary_uom": "liter",
    },
    {
        "code": "ADD-3",
        "name": "Additive 3",
        "description": "Performance additive package for automotive engine and 2-stroke oils",
        "category": MaterialCategory.ADDITIVE_PACKAGE,
        "standard_viscosity_40c": 200.0,
        "standard_viscosity_100c": 18.0,
        "standard_density_15c": 0.960,
        "standard_flash_point": 170.0,
        "standard_tbn": 70.0,
        "standard_cost_per_liter": 5.20,
        "primary_uom": "liter",
    },
    {
        "code": "ADD-GEN",
        "name": "Additive General",
        "description": "General-purpose additive package for industrial lubricants",
        "category": MaterialCategory.ADDITIVE_PACKAGE,
        "standard_viscosity_40c": 120.0,
        "standard_viscosity_100c": 12.0,
        "standard_density_15c": 0.940,
        "standard_flash_point": 190.0,
        "standard_tbn": 30.0,
        "standard_cost_per_liter": 3.80,
        "primary_uom": "liter",
    },
]

# Recipe definitions from recipe.xlsx
RECIPE_SEED_DATA: list[dict[str, Any]] = [
    {
        "code": "G-40",
        "name": "G 40",
        "description": "Industrial gear oil ISO VG 40 - for enclosed gear systems and circulating oil applications",
        "product_application": ProductApplication.GEAR_OIL,
        "usage_description": "Enclosed gear systems, circulating oil, industrial gearboxes",
        "sae_grade": None,
        "iso_grade": "ISO VG 140",
        "target_viscosity_40c": 140.0,
        "target_viscosity_100c": 14.4,
        "formulation_source": "recipe.xlsx",
        "ingredients": [
            {"material_code": "SN-500", "name": "SN 500", "type": IngredientType.BASE_OIL, "pct": 75.42},
            {"material_code": "BS-460", "name": "BS 460", "type": IngredientType.BASE_OIL, "pct": 16.92},
            {"material_code": "ADD-PKG-45", "name": "Additives 4-5", "type": IngredientType.ADDITIVE, "pct": 7.66},
        ],
    },
    {
        "code": "A-40",
        "name": "A 40",
        "description": "Automotive/industrial lubricant ISO VG 40 - general purpose oil with higher additive treat",
        "product_application": ProductApplication.INDUSTRIAL_LUBRICANT,
        "usage_description": "General purpose industrial lubrication, hydraulic systems, light-duty gears",
        "sae_grade": None,
        "iso_grade": "ISO VG 127",
        "target_viscosity_40c": 127.0,
        "target_viscosity_100c": 14.0,
        "formulation_source": "recipe.xlsx",
        "ingredients": [
            {"material_code": "SN-500", "name": "SN 500", "type": IngredientType.BASE_OIL, "pct": 79.0542},
            {"material_code": "BS-460", "name": "BS 460", "type": IngredientType.BASE_OIL, "pct": 5.89},
            {"material_code": "ADD-PKG-45", "name": "Additives 4-5", "type": IngredientType.ADDITIVE, "pct": 15.056},
        ],
    },
    {
        "code": "H-20W50-SJCF",
        "name": "H 20W50 SJ/CF",
        "description": "Multi-grade engine oil SAE 20W-50 API SJ/CF - for passenger cars and light trucks",
        "product_application": ProductApplication.ENGINE_OIL,
        "usage_description": "Passenger car motor oil, light truck engines, API SJ/CF performance level",
        "sae_grade": "20W-50",
        "iso_grade": None,
        "target_viscosity_40c": 178.0,
        "target_viscosity_100c": 19.0,
        "formulation_source": "recipe.xlsx",
        "ingredients": [
            {"material_code": "HVI-120", "name": "HVI 120", "type": IngredientType.BASE_OIL, "pct": 45.3},
            {"material_code": "HVI-650", "name": "HVI 650", "type": IngredientType.BASE_OIL, "pct": 15.0},
            {"material_code": "GTL-BO8", "name": "GTL BO8", "type": IngredientType.BASE_OIL, "pct": 20.0},
            {"material_code": "ADD-3", "name": "Additive 3", "type": IngredientType.ADDITIVE, "pct": 19.7},
        ],
    },
    {
        "code": "R-50",
        "name": "R 50",
        "description": "Heavy-duty industrial oil ISO VG 50 - for robust industrial applications",
        "product_application": ProductApplication.INDUSTRIAL_LUBRICANT,
        "usage_description": "Heavy industrial applications, bearing lubrication, high-load machinery",
        "sae_grade": None,
        "iso_grade": "ISO VG 179",
        "target_viscosity_40c": 179.0,
        "target_viscosity_100c": 17.0,
        "formulation_source": "recipe.xlsx",
        "ingredients": [
            {"material_code": "BS-460", "name": "BS 460", "type": IngredientType.BASE_OIL, "pct": 50.0},
            {"material_code": "EHC-110", "name": "EHC 110", "type": IngredientType.BASE_OIL, "pct": 36.973},
            {"material_code": "GTL-BO8", "name": "GTL BO8", "type": IngredientType.BASE_OIL, "pct": 6.877},
            {"material_code": "ADD-GEN", "name": "Additive General", "type": IngredientType.ADDITIVE, "pct": 6.15},
        ],
    },
    {
        "code": "2T",
        "name": "2T Two-Stroke Oil",
        "description": "Two-stroke engine oil - for 2-stroke motorcycle and small engine applications",
        "product_application": ProductApplication.TWO_STROKE_OIL,
        "usage_description": "2-stroke motorcycle engines, scooters, small engines, chain saws, outboard motors",
        "sae_grade": None,
        "iso_grade": None,
        "target_viscosity_40c": None,
        "target_viscosity_100c": 7.5,
        "formulation_source": "recipe.xlsx",
        "ingredients": [
            {"material_code": "HVI-60", "name": "HVI 60", "type": IngredientType.BASE_OIL, "pct": 65.0},
            {"material_code": "HVI-120", "name": "HVI 120", "type": IngredientType.BASE_OIL, "pct": 10.6},
            {"material_code": "HVI-650", "name": "HVI 650", "type": IngredientType.BASE_OIL, "pct": 13.0},
            {"material_code": "ADD-3", "name": "Additive 3", "type": IngredientType.ADDITIVE, "pct": 11.4},
        ],
    },
]


async def seed_materials(db) -> list[Material]:
    """Seed the materials table with base oil and additive data."""
    from sqlalchemy import select

    created = []
    for mat_data in MATERIAL_SEED_DATA:
        existing = await db.execute(
            select(Material).where(Material.code == mat_data["code"])
        )
        if existing.scalar_one_or_none():
            logger.info("Material already exists, skipping", code=mat_data["code"])
            continue

        material = Material(**mat_data)
        db.add(material)
        created.append(material)
        logger.info("Created material", code=mat_data["code"], name=mat_data["name"])

    if created:
        await db.flush()
    return created


async def seed_recipes(db) -> list[Recipe]:
    """Seed the recipes table with formulations from recipe.xlsx."""
    from sqlalchemy import select

    created = []
    for rec_data in RECIPE_SEED_DATA:
        existing = await db.execute(
            select(Recipe).where(Recipe.code == rec_data["code"])
        )
        if existing.scalar_one_or_none():
            logger.info("Recipe already exists, skipping", code=rec_data["code"])
            continue

        ingredients_data = rec_data.pop("ingredients")

        recipe = Recipe(
            code=rec_data["code"],
            name=rec_data["name"],
            description=rec_data["description"],
            product_application=rec_data["product_application"],
            usage_description=rec_data["usage_description"],
            sae_grade=rec_data.get("sae_grade"),
            iso_grade=rec_data.get("iso_grade"),
            target_viscosity_40c=rec_data.get("target_viscosity_40c"),
            target_viscosity_100c=rec_data.get("target_viscosity_100c"),
            formulation_source=rec_data.get("formulation_source"),
            status=RecipeStatus.APPROVED,
        )
        db.add(recipe)
        await db.flush()

        for order, ing in enumerate(ingredients_data, 1):
            ingredient = RecipeIngredient(
                recipe_id=recipe.id,
                material_code=ing["material_code"],
                material_name=ing["name"],
                ingredient_type=ing["type"],
                target_percentage=ing["pct"],
                addition_order=order,
                ai_adjustable=ing["type"] == IngredientType.BASE_OIL,
            )
            db.add(ingredient)

        created.append(recipe)
        logger.info("Created recipe", code=rec_data["code"], name=rec_data["name"])

    if created:
        await db.flush()
        await db.commit()
    return created


async def seed_all(db) -> dict[str, int]:
    """Seed all data."""
    materials = await seed_materials(db)
    recipes = await seed_recipes(db)
    return {
        "materials_created": len(materials),
        "recipes_created": len(recipes),
    }
