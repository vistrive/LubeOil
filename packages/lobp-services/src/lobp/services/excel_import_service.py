"""Service for importing recipe data from Excel files."""

from io import BytesIO
from typing import Any

import structlog

logger = structlog.get_logger()


def parse_recipe_excel(file_content: bytes) -> list[dict[str, Any]]:
    """
    Parse recipe data from an Excel file matching recipe.xlsx format.

    Expected columns: Product, BO/AD, Component, Description, Wt %, KV 40, KV 100

    Returns list of parsed recipe dicts grouped by product.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required for Excel import: pip install openpyxl")

    wb = openpyxl.load_workbook(BytesIO(file_content), read_only=True)
    ws = wb.active
    if ws is None:
        raise ValueError("Excel file has no active sheet")

    # Read header row
    rows = list(ws.iter_rows(min_row=1, values_only=True))
    if not rows:
        raise ValueError("Excel file is empty")

    headers = [str(h).strip().lower() if h else "" for h in rows[0]]

    # Map columns
    col_map = _map_columns(headers)

    # Parse rows grouped by product
    products: dict[str, dict[str, Any]] = {}
    for row in rows[1:]:
        if not row or not row[col_map["product"]]:
            continue

        product = str(row[col_map["product"]]).strip()
        bo_ad = str(row[col_map["bo_ad"]]).strip().upper() if row[col_map["bo_ad"]] else ""
        component = str(row[col_map["component"]]).strip() if row[col_map["component"]] else ""
        description = str(row[col_map["description"]]).strip() if row[col_map["description"]] else ""
        wt_pct = float(row[col_map["wt_pct"]]) if row[col_map["wt_pct"]] else 0.0

        if product not in products:
            kv40 = float(row[col_map["kv40"]]) if col_map.get("kv40") is not None and row[col_map["kv40"]] else None
            kv100 = float(row[col_map["kv100"]]) if col_map.get("kv100") is not None and row[col_map["kv100"]] else None
            products[product] = {
                "product_name": product,
                "target_viscosity_40c": kv40,
                "target_viscosity_100c": kv100,
                "ingredients": [],
            }

        # Convert weight fraction (0-1) to percentage (0-100) if needed
        pct = wt_pct * 100.0 if wt_pct <= 1.0 else wt_pct

        products[product]["ingredients"].append({
            "type": "base_oil" if bo_ad == "BO" else "additive",
            "component_code": component,
            "description": description,
            "weight_percent": round(pct, 4),
        })

    wb.close()

    result = list(products.values())
    logger.info("Parsed Excel recipes", count=len(result))
    return result


def _map_columns(headers: list[str]) -> dict[str, int]:
    """Map header names to column indices."""
    mapping = {}
    for i, h in enumerate(headers):
        if "product" in h:
            mapping["product"] = i
        elif "bo" in h and "ad" in h:
            mapping["bo_ad"] = i
        elif "component" in h:
            mapping["component"] = i
        elif "description" in h or "desc" in h:
            mapping["description"] = i
        elif "wt" in h or "weight" in h:
            mapping["wt_pct"] = i
        elif "40" in h:
            mapping["kv40"] = i
        elif "100" in h:
            mapping["kv100"] = i

    required = ["product", "bo_ad", "component", "wt_pct"]
    missing = [k for k in required if k not in mapping]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {headers}")

    # Set defaults for optional columns
    mapping.setdefault("description", mapping.get("component", 0))

    return mapping
