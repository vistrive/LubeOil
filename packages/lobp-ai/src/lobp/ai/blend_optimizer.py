"""
Blend formulation optimizer.

Given target viscosity specs and available materials, finds optimal
blend combinations using iterative random sampling with refinement.
"""

import numpy as np
import structlog

from lobp.ai.blending_calculator import BlendComponent, calculate_blend

logger = structlog.get_logger()


def optimize_for_target(
    available_materials: list[dict],
    target_viscosity_40c: float | None = None,
    target_viscosity_100c: float | None = None,
    max_components: int = 4,
    max_additive_pct: float = 0.25,
    iterations: int = 500,
) -> list[dict]:
    """
    Find blend combinations that achieve target viscosity specifications.

    Returns top 10 candidate formulations sorted by quality score.
    """
    base_oils = [m for m in available_materials if "base_oil" in m.get("category", "")]
    additives = [m for m in available_materials if "additive" in m.get("category", "")]

    if not base_oils or not additives:
        return []

    rng = np.random.default_rng(42)
    candidates = []

    for _ in range(iterations):
        n_bo = rng.integers(1, min(len(base_oils), max_components - 1) + 1)
        n_ad = rng.integers(1, min(len(additives), max_components - n_bo) + 1)

        sel_bo = rng.choice(len(base_oils), size=n_bo, replace=False)
        sel_ad = rng.choice(len(additives), size=n_ad, replace=False)

        ad_total = rng.uniform(0.03, max_additive_pct)
        bo_total = 1.0 - ad_total

        bo_fracs = _random_split(rng, n_bo, bo_total)
        ad_fracs = _random_split(rng, n_ad, ad_total)

        components = []
        for i, idx in enumerate(sel_bo):
            components.append(_mat_to_component(base_oils[idx], bo_fracs[i]))
        for i, idx in enumerate(sel_ad):
            components.append(_mat_to_component(additives[idx], ad_fracs[i]))

        result = calculate_blend(components)
        score = _score_blend(result, target_viscosity_40c, target_viscosity_100c)

        if score > 20:
            cost = _estimate_cost(components, available_materials)
            candidates.append({
                "components": [
                    {"material_code": c.material_code, "name": c.name,
                     "weight_percent": round(c.weight_fraction * 100, 4)}
                    for c in components
                ],
                "predicted_properties": {
                    "viscosity_40c": result.viscosity_40c,
                    "viscosity_100c": result.viscosity_100c,
                    "viscosity_index": result.viscosity_index,
                    "density_15c": result.density_15c,
                    "flash_point": result.flash_point_estimate,
                    "pour_point": result.pour_point_estimate,
                },
                "score": round(score, 2),
                "estimated_cost_per_liter": round(cost, 2),
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:10]


def _random_split(rng, n: int, total: float) -> list[float]:
    if n == 1:
        return [total]
    splits = np.sort(rng.uniform(0, 1, size=n - 1))
    fracs = np.diff(np.concatenate([[0], splits, [1]])) * total
    return fracs.tolist()


def _mat_to_component(mat: dict, fraction: float) -> BlendComponent:
    return BlendComponent(
        material_code=mat["code"],
        name=mat["name"],
        weight_fraction=fraction,
        viscosity_40c=mat.get("standard_viscosity_40c"),
        viscosity_100c=mat.get("standard_viscosity_100c"),
        viscosity_index=mat.get("standard_viscosity_index"),
        density_15c=mat.get("standard_density_15c"),
        flash_point=mat.get("standard_flash_point"),
        pour_point=mat.get("standard_pour_point"),
    )


def _score_blend(result, target_40c, target_100c) -> float:
    score = 0.0
    if target_40c and result.viscosity_40c:
        dev = abs(result.viscosity_40c - target_40c) / target_40c
        if dev < 0.05:
            score += 50 * (1 - dev / 0.05)
    if target_100c and result.viscosity_100c:
        dev = abs(result.viscosity_100c - target_100c) / target_100c
        if dev < 0.05:
            score += 50 * (1 - dev / 0.05)
    return score


def _estimate_cost(components: list[BlendComponent], materials: list[dict]) -> float:
    cost_map = {m["code"]: m.get("standard_cost_per_liter", 2.0) for m in materials}
    return sum(c.weight_fraction * cost_map.get(c.material_code, 2.0) for c in components)
