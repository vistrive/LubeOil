"""Prompt templates for LLM-powered recipe analysis."""

SYSTEM_PROMPT = """You are an expert lubricant formulation chemist and blending engineer.
You have deep knowledge of:
- Base oil groups (Group I-V), their viscosity characteristics, and blending behavior
- Additive chemistry (detergents, dispersants, anti-wear, VI improvers, pour point depressants)
- ASTM standards (D341 viscosity-temperature, D2270 viscosity index, D445 kinematic viscosity)
- SAE and ISO viscosity grading systems
- Walther equation for viscosity blending
- Product applications (engine oils, gear oils, hydraulic oils, 2-stroke oils, etc.)

Provide concise, technically accurate responses. Use metric units (cSt, mm2/s, degC).
When suggesting formulations, always ensure weight percentages sum to 100%."""


def recipe_analysis_prompt(recipe: dict) -> str:
    """Generate prompt for analyzing an existing recipe."""
    ingredients = "\n".join(
        f"  - {ing['name']} ({ing['type']}): {ing['weight_percent']:.2f}%"
        for ing in recipe.get("ingredients", [])
    )
    return f"""Analyze this lubricant recipe:

Product: {recipe.get('name', 'Unknown')}
Application: {recipe.get('application', 'Unknown')}
Target KV@40C: {recipe.get('target_viscosity_40c', 'N/A')} cSt
Target KV@100C: {recipe.get('target_viscosity_100c', 'N/A')} cSt

Ingredients:
{ingredients}

Provide:
1. Assessment of the formulation balance (base oil vs additive ratio)
2. Expected performance characteristics
3. Potential improvement suggestions
4. Compatibility considerations"""


def formulation_suggestion_prompt(
    target_visc_40c: float | None,
    target_visc_100c: float | None,
    application: str,
    available_materials: list[dict],
) -> str:
    """Generate prompt for suggesting a new formulation."""
    mat_list = "\n".join(
        f"  - {m['name']} (code: {m['code']}, KV40: {m.get('standard_viscosity_40c', 'N/A')}, "
        f"KV100: {m.get('standard_viscosity_100c', 'N/A')}, "
        f"cost: ${m.get('standard_cost_per_liter', 'N/A')}/L)"
        for m in available_materials
    )
    return f"""Suggest a lubricant formulation for:

Application: {application}
Target KV@40C: {target_visc_40c or 'Not specified'} cSt
Target KV@100C: {target_visc_100c or 'Not specified'} cSt

Available materials:
{mat_list}

Provide:
1. Recommended formulation with specific weight percentages (must sum to 100%)
2. Expected properties (viscosity, flash point, pour point)
3. Rationale for material selection
4. Cost estimation
5. Alternative formulation if possible"""


def quality_prediction_prompt(recipe: dict, blend_conditions: dict) -> str:
    """Generate prompt for predicting blend quality."""
    ingredients = "\n".join(
        f"  - {ing['name']}: {ing['weight_percent']:.2f}%"
        for ing in recipe.get("ingredients", [])
    )
    return f"""Predict quality outcome for this blend:

Recipe: {recipe.get('name', 'Unknown')}
Ingredients:
{ingredients}

Blending conditions:
  Temperature: {blend_conditions.get('temperature', 'N/A')} C
  Mixing speed: {blend_conditions.get('mixing_speed', 'N/A')} RPM
  Duration: {blend_conditions.get('duration', 'N/A')} minutes

Predict:
1. Expected viscosity at 40C and 100C
2. Viscosity index
3. Flash point and pour point estimates
4. Risk of off-spec result (high/medium/low)
5. Recommended quality checks"""
