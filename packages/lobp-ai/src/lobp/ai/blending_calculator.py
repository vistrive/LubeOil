"""
ASTM D341 / Walther equation viscosity blending calculator.

Implements the industry-standard method for predicting kinematic viscosity
of lubricant blends from individual component viscosities and weight fractions.
"""

import math
from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class BlendComponent:
    """A single component in a blend with its properties."""

    material_code: str
    name: str
    weight_fraction: float  # 0.0 to 1.0
    viscosity_40c: float | None = None
    viscosity_100c: float | None = None
    viscosity_index: float | None = None
    density_15c: float | None = None
    flash_point: float | None = None
    pour_point: float | None = None


@dataclass
class BlendResult:
    """Calculated properties of a blend."""

    viscosity_40c: float | None
    viscosity_100c: float | None
    viscosity_index: float | None
    density_15c: float | None
    flash_point_estimate: float | None
    pour_point_estimate: float | None
    total_weight_fraction: float
    warnings: list[str]


def walther_transform(viscosity_cst: float) -> float:
    """Apply Walther (ASTM D341) double-log transform: W = log10(log10(v + 0.7))."""
    if viscosity_cst <= 0:
        raise ValueError(f"Viscosity must be positive, got {viscosity_cst}")
    return math.log10(math.log10(viscosity_cst + 0.7))


def inverse_walther_transform(w: float) -> float:
    """Inverse Walther transform: v = 10^(10^W) - 0.7."""
    return 10.0 ** (10.0 ** w) - 0.7


def blend_viscosity(
    components: list[BlendComponent],
    temperature: str = "40c",
) -> float | None:
    """
    Calculate blend viscosity using the Walther/Wright blending equation.

    W_blend = sum(x_i * W_i) where W_i = log10(log10(v_i + 0.7))
    """
    valid = []
    total_frac = 0.0
    for c in components:
        visc = c.viscosity_40c if temperature == "40c" else c.viscosity_100c
        if visc and visc > 0 and c.weight_fraction > 0:
            valid.append((c.weight_fraction, visc))
            total_frac += c.weight_fraction

    if not valid or total_frac < 0.01:
        return None

    w_blend = sum(
        (frac / total_frac) * walther_transform(visc)
        for frac, visc in valid
    )
    return inverse_walther_transform(w_blend)


def blend_density(components: list[BlendComponent]) -> float | None:
    """Calculate blend density using weight-fraction additive mixing."""
    total = sum(c.weight_fraction * c.density_15c for c in components
                if c.density_15c and c.weight_fraction > 0)
    total_frac = sum(c.weight_fraction for c in components
                     if c.density_15c and c.weight_fraction > 0)
    return (total / total_frac) if total_frac > 0.01 else None


def estimate_flash_point(components: list[BlendComponent]) -> float | None:
    """Estimate flash point (biased toward lightest component)."""
    valid = [(c.flash_point, c.weight_fraction) for c in components
             if c.flash_point and c.weight_fraction > 0]
    if not valid:
        return None
    min_fp = min(fp for fp, _ in valid)
    avg = sum(fp * wf for fp, wf in valid) / sum(wf for _, wf in valid)
    return 0.3 * min_fp + 0.7 * avg


def estimate_pour_point(components: list[BlendComponent]) -> float | None:
    """Estimate pour point (biased toward worst component)."""
    valid = [(c.pour_point, c.weight_fraction) for c in components
             if c.pour_point is not None and c.weight_fraction > 0]
    if not valid:
        return None
    max_pp = max(pp for pp, _ in valid)
    avg = sum(pp * wf for pp, wf in valid) / sum(wf for _, wf in valid)
    return 0.4 * max_pp + 0.6 * avg


def calculate_viscosity_index(visc_40c: float, visc_100c: float) -> float | None:
    """Calculate Viscosity Index per ASTM D2270 (simplified)."""
    if visc_40c <= 0 or visc_100c <= 0 or visc_100c >= visc_40c:
        return None
    y = visc_100c
    if y < 2.0:
        return None
    l_val = 0.8353 * y * y + 14.67 * y - 216.0
    h_val = 0.1684 * y * y + 11.85 * y - 97.0
    if l_val <= h_val:
        return None
    return round(((l_val - visc_40c) / (l_val - h_val)) * 100.0, 1)


def calculate_blend(components: list[BlendComponent]) -> BlendResult:
    """Main entry point: calculate all properties for a blend."""
    warnings = []
    total_frac = sum(c.weight_fraction for c in components)
    if abs(total_frac - 1.0) > 0.001:
        warnings.append(f"Weight fractions sum to {total_frac:.4f}, expected 1.0")

    v40 = blend_viscosity(components, "40c")
    v100 = blend_viscosity(components, "100c")
    vi = calculate_viscosity_index(v40, v100) if v40 and v100 else None

    return BlendResult(
        viscosity_40c=round(v40, 2) if v40 else None,
        viscosity_100c=round(v100, 2) if v100 else None,
        viscosity_index=vi,
        density_15c=round(blend_density(components), 4) if blend_density(components) else None,
        flash_point_estimate=round(estimate_flash_point(components), 1) if estimate_flash_point(components) else None,
        pour_point_estimate=round(estimate_pour_point(components), 1) if estimate_pour_point(components) else None,
        total_weight_fraction=round(total_frac, 6),
        warnings=warnings,
    )
