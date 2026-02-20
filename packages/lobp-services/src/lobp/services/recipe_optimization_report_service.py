"""
Recipe Optimization Report Generation Service.

Generates the 4 report types specified in the AI Recipe Optimization document:
1. Optimal Recipe Recommendation
2. Predicted Final Quality Report
3. Cost Comparison Report
4. Production Variance Report

All reports match the exact format specified in Section 4 of the document.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class TargetSpecification:
    """Target specification for a quality parameter."""

    parameter: str
    target: float
    min_value: float | None = None
    max_value: float | None = None
    unit: str = ""


@dataclass
class RawMaterialInfo:
    """Raw material information for recipe report."""

    name: str
    property_value: str  # e.g., "32.6 cSt @ 40°C"
    cost: str  # e.g., "$5.42/L"
    in_stock: bool = True


@dataclass
class RecipeIngredientLine:
    """Single ingredient line in recipe recommendation."""

    material_name: str
    quantity_wt_percent: float
    cost_per_unit: str
    total_cost: float


@dataclass
class QualityPredictionLine:
    """Single quality parameter prediction."""

    parameter: str
    predicted_value: float
    target_range: str
    status: str  # ✓ PASS, ⚠ MARGINAL, ✗ FAIL
    unit: str = ""


@dataclass
class CostComparisonLine:
    """Cost comparison line item."""

    category: str
    legacy_cost: float
    optimized_cost: float
    savings: float
    savings_percent: float


@dataclass
class VarianceMetric:
    """Production variance metric."""

    parameter: str
    std_dev: float
    target_std: float
    status: str  # OK, WARNING, CRITICAL
    trend: str  # ↑ ↓ →


class RecipeOptimizationReportService:
    """
    Service for generating AI Recipe Optimization reports.

    Implements all 4 report types from the document:
    - Optimal Recipe Recommendation
    - Predicted Final Quality Report
    - Cost Comparison Report
    - Production Variance Report
    """

    def __init__(self, batches_per_year: int = 250):
        """
        Initialize report service.

        Args:
            batches_per_year: Annual batch count for savings projections
        """
        self.batches_per_year = batches_per_year

    def generate_recipe_recommendation_report(
        self,
        recipe_code: str,
        product_name: str,
        batch_volume: float,
        target_specs: list[TargetSpecification],
        raw_materials: list[RawMaterialInfo],
        ingredients: list[RecipeIngredientLine],
        predicted_quality: list[QualityPredictionLine],
        confidence_level: float,
        historical_batches: int,
        standard_cost_per_liter: float,
        optimized_cost_per_liter: float,
        notes: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate Optimal Recipe Recommendation report.

        Matches the format shown on pages 5-6 of the document.

        Args:
            recipe_code: Recipe identifier
            product_name: Product name (e.g., "ISO 32 Hydraulic Oil")
            batch_volume: Target batch volume in liters
            target_specs: List of target specifications
            raw_materials: Available raw materials with properties
            ingredients: Optimized ingredient lines
            predicted_quality: Quality predictions
            confidence_level: AI confidence (0-100%)
            historical_batches: Number of batches used for training
            standard_cost_per_liter: Legacy recipe cost
            optimized_cost_per_liter: AI-optimized cost
            notes: Optional notes

        Returns:
            Structured report dictionary
        """
        generated_at = datetime.now(timezone.utc)

        # Calculate totals
        total_batch_cost = sum(ing.total_cost for ing in ingredients)
        savings_per_batch = (standard_cost_per_liter - optimized_cost_per_liter) * batch_volume
        savings_percent = (
            (standard_cost_per_liter - optimized_cost_per_liter) / standard_cost_per_liter * 100
            if standard_cost_per_liter > 0 else 0
        )
        annual_savings = savings_per_batch * self.batches_per_year

        report = {
            "report_type": "OPTIMIZED_RECIPE_RECOMMENDATION",
            "generated_at": generated_at.isoformat(),

            # Header
            "header": {
                "product": product_name,
                "batch_volume_liters": batch_volume,
                "recipe_code": recipe_code,
            },

            # Target specifications
            "target_specifications": [
                {
                    "parameter": spec.parameter,
                    "target": spec.target,
                    "tolerance_min": spec.min_value,
                    "tolerance_max": spec.max_value,
                    "unit": spec.unit,
                    "display": f"{spec.parameter}: {spec.target} {spec.unit} (Tolerance: {spec.min_value}-{spec.max_value})"
                }
                for spec in target_specs
            ],

            # Raw materials available
            "raw_materials_available": [
                {
                    "name": mat.name,
                    "property": mat.property_value,
                    "cost": mat.cost,
                    "in_stock": mat.in_stock,
                }
                for mat in raw_materials
            ],

            # Optimized recipe
            "optimized_recipe": {
                "ingredients": [
                    {
                        "material": ing.material_name,
                        "qty_wt_percent": ing.quantity_wt_percent,
                        "cost_per_unit": ing.cost_per_unit,
                        "total_cost": ing.total_cost,
                    }
                    for ing in ingredients
                ],
                "total_batch_cost": total_batch_cost,
                "cost_per_liter": optimized_cost_per_liter,
            },

            # Predicted output quality
            "predicted_quality": [
                {
                    "parameter": pred.parameter,
                    "predicted": pred.predicted_value,
                    "target_range": pred.target_range,
                    "status": pred.status,
                    "unit": pred.unit,
                }
                for pred in predicted_quality
            ],

            # Confidence
            "confidence": {
                "level_percent": confidence_level,
                "historical_batches_used": historical_batches,
            },

            # Cost comparison
            "cost_comparison": {
                "standard_recipe_cost_per_liter": standard_cost_per_liter,
                "ai_optimized_cost_per_liter": optimized_cost_per_liter,
                "savings_per_batch": savings_per_batch,
                "savings_percent": savings_percent,
                "annual_savings_estimate": annual_savings,
                "batches_per_year": self.batches_per_year,
            },

            # Notes
            "notes": notes or [
                "AI considers current base oil properties & available additives",
                "Recipe updates automatically if raw materials change",
                "Recommendation valid for 2 hours or until new material lot received",
            ],
        }

        logger.info(
            "Generated recipe recommendation report",
            recipe=recipe_code,
            savings_percent=savings_percent,
        )

        return report

    def generate_quality_prediction_report(
        self,
        blend_id: str,
        product_name: str,
        batch_volume: float,
        predictions: list[QualityPredictionLine],
        confidence_percent: float,
        historical_batches: int,
        risk_assessment: str,
        risk_factors: list[str] | None = None,
        recommendations: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate Predicted Final Quality Report.

        Matches the format shown on pages 6-7 of the document.

        Args:
            blend_id: Blend identifier
            product_name: Product name
            batch_volume: Batch volume in liters
            predictions: List of quality predictions
            confidence_percent: Overall confidence
            historical_batches: Batches used for prediction
            risk_assessment: Risk level (LOW, MEDIUM, HIGH)
            risk_factors: List of identified risks
            recommendations: Optimization recommendations

        Returns:
            Structured report dictionary
        """
        generated_at = datetime.now(timezone.utc)

        # Determine overall status
        statuses = [p.status for p in predictions]
        if any("FAIL" in s for s in statuses):
            overall_status = "OFF_SPEC_RISK"
        elif any("MARGINAL" in s for s in statuses):
            overall_status = "MARGINAL"
        else:
            overall_status = "ON_SPEC"

        report = {
            "report_type": "PREDICTED_BLEND_QUALITY_ANALYSIS",
            "generated_at": generated_at.isoformat(),

            # Header
            "header": {
                "blend_id": blend_id,
                "product": product_name,
                "volume_liters": batch_volume,
            },

            # Predictions
            "predicted_quality_metrics": [
                {
                    "parameter": pred.parameter,
                    "predicted_value": pred.predicted_value,
                    "target": pred.target_range,
                    "status": pred.status,
                    "unit": pred.unit,
                }
                for pred in predictions
            ],

            # Confidence
            "quality_prediction_confidence": {
                "percent": confidence_percent,
                "historical_batches_used": historical_batches,
                "note": f"Based on similar blend conditions from {historical_batches} historical batches",
            },

            # Risk assessment
            "risk_assessment": {
                "level": risk_assessment,
                "factors": risk_factors or [],
                "overall_status": overall_status,
            },

            # Recommendations
            "recommendations": recommendations or [
                "All parameters within specification ranges",
                "No predicted off-spec risk",
                "Optimal temperature control recommended during blending",
            ],

            # Expected lab results
            "expected_lab_results": {
                "viscosity_tolerance": "±0.3 cSt (within specification)",
                "tbn_tolerance": "±0.15 mgKOH (within specification)",
                "verification_time": "~24 hours post-blend",
            },
        }

        logger.info(
            "Generated quality prediction report",
            blend_id=blend_id,
            confidence=confidence_percent,
            risk=risk_assessment,
        )

        return report

    def generate_cost_comparison_report(
        self,
        product_name: str,
        batch_volume: float,
        legacy_costs: dict[str, float],
        optimized_costs: dict[str, float],
        optimization_methods: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate Cost Comparison Report.

        Matches the format shown on pages 7-8 of the document.

        Args:
            product_name: Product name
            batch_volume: Batch volume in liters
            legacy_costs: Legacy recipe costs by category
            optimized_costs: AI-optimized costs by category
            optimization_methods: Methods used for optimization

        Returns:
            Structured report dictionary
        """
        generated_at = datetime.now(timezone.utc)

        # Calculate totals
        legacy_total = sum(legacy_costs.values())
        optimized_total = sum(optimized_costs.values())
        batch_savings = legacy_total - optimized_total
        savings_percent = (batch_savings / legacy_total * 100) if legacy_total > 0 else 0
        annual_savings = batch_savings * self.batches_per_year

        # Build comparison lines
        comparisons = []
        for category in legacy_costs.keys():
            legacy = legacy_costs.get(category, 0)
            optimized = optimized_costs.get(category, 0)
            savings = legacy - optimized
            pct = (savings / legacy * 100) if legacy > 0 else 0

            comparisons.append({
                "category": category,
                "legacy_cost": legacy,
                "optimized_cost": optimized,
                "savings": savings,
                "savings_percent": pct,
            })

        report = {
            "report_type": "COST_OPTIMIZATION_ANALYSIS",
            "generated_at": generated_at.isoformat(),

            # Header
            "header": {
                "product": product_name,
                "batch_volume_liters": batch_volume,
            },

            # Legacy recipe
            "legacy_recipe": {
                "costs": legacy_costs,
                "total_cost": legacy_total,
                "cost_per_liter": legacy_total / batch_volume if batch_volume > 0 else 0,
            },

            # AI-optimized recipe
            "ai_optimized_recipe": {
                "costs": optimized_costs,
                "total_cost": optimized_total,
                "cost_per_liter": optimized_total / batch_volume if batch_volume > 0 else 0,
            },

            # Comparison
            "comparison": {
                "by_category": comparisons,
                "batch_savings": batch_savings,
                "batch_savings_percent": savings_percent,
                "annual_savings": annual_savings,
                "batches_per_year": self.batches_per_year,
            },

            # Optimization achieved through
            "optimization_methods": optimization_methods or [
                "Supplier switching: Base oil from cheaper vendor",
                "Additive ratio optimization: Reduced expensive component usage",
                "Inventory utilization: Prioritized high-stock, low-cost materials",
            ],
        }

        logger.info(
            "Generated cost comparison report",
            product=product_name,
            savings_percent=savings_percent,
            annual_savings=annual_savings,
        )

        return report

    def generate_variance_report(
        self,
        period_days: int,
        batches_analyzed: int,
        variance_metrics: list[VarianceMetric],
        off_spec_rate_current: float,
        off_spec_rate_previous: float,
        improvement_notes: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate Production Variance Report.

        Matches the format shown on page 8 of the document.

        Args:
            period_days: Analysis period in days
            batches_analyzed: Number of batches in period
            variance_metrics: List of variance metrics
            off_spec_rate_current: Current off-spec rate (%)
            off_spec_rate_previous: Previous period off-spec rate (%)
            improvement_notes: Notes on improvements

        Returns:
            Structured report dictionary
        """
        generated_at = datetime.now(timezone.utc)
        period_start = generated_at - timedelta(days=period_days)

        # Calculate improvement
        off_spec_improvement = (
            (off_spec_rate_previous - off_spec_rate_current) / off_spec_rate_previous * 100
            if off_spec_rate_previous > 0 else 0
        )

        # Determine overall trend
        warning_count = sum(1 for m in variance_metrics if m.status == "WARNING")
        critical_count = sum(1 for m in variance_metrics if m.status == "CRITICAL")

        if critical_count > 0:
            consistency_trend = "NEEDS_ATTENTION"
        elif warning_count > 0:
            consistency_trend = "GOOD"
        else:
            consistency_trend = "EXCELLENT"

        report = {
            "report_type": "PRODUCTION_CONSISTENCY_DASHBOARD",
            "generated_at": generated_at.isoformat(),

            # Header
            "header": {
                "period_days": period_days,
                "period_start": period_start.isoformat(),
                "period_end": generated_at.isoformat(),
                "batches_analyzed": batches_analyzed,
            },

            # Quality variance analysis
            "quality_variance_analysis": [
                {
                    "parameter": metric.parameter,
                    "std_dev": metric.std_dev,
                    "target_std": metric.target_std,
                    "status": metric.status,
                    "trend": metric.trend,
                    "within_target": metric.std_dev <= metric.target_std,
                }
                for metric in variance_metrics
            ],

            # Off-spec performance
            "off_spec_performance": {
                "current_rate_percent": off_spec_rate_current,
                "previous_rate_percent": off_spec_rate_previous,
                "improvement_percent": off_spec_improvement,
                "target_rate_percent": 0.5,  # Document target: <0.5%
                "meets_target": off_spec_rate_current < 0.5,
            },

            # Performance improvement summary
            "performance_improvement": {
                "previous_period_off_spec": f"{off_spec_rate_previous}%",
                "current_period_off_spec": f"{off_spec_rate_current}%",
                "improvement": f"{off_spec_improvement:.0f}% reduction in off-spec batches",
            },

            # Consistency trend
            "consistency_trend": {
                "status": consistency_trend,
                "notes": improvement_notes or [
                    "All parameters stable",
                    "Viscosity variance decreased significantly",
                    "TBN consistency improved",
                    "No evidence of recipe drift",
                ],
            },
        }

        logger.info(
            "Generated variance report",
            period_days=period_days,
            batches=batches_analyzed,
            off_spec_rate=off_spec_rate_current,
        )

        return report

    def format_report_as_text(self, report: dict[str, Any]) -> str:
        """
        Format a report dictionary as formatted text for display.

        Matches the visual format shown in the document.
        """
        report_type = report.get("report_type", "REPORT")
        lines = []

        # Header separator
        lines.append("=" * 70)
        lines.append(report_type)
        lines.append("=" * 70)

        if report_type == "OPTIMIZED_RECIPE_RECOMMENDATION":
            return self._format_recipe_recommendation(report, lines)
        elif report_type == "PREDICTED_BLEND_QUALITY_ANALYSIS":
            return self._format_quality_prediction(report, lines)
        elif report_type == "COST_OPTIMIZATION_ANALYSIS":
            return self._format_cost_comparison(report, lines)
        elif report_type == "PRODUCTION_CONSISTENCY_DASHBOARD":
            return self._format_variance_report(report, lines)
        else:
            lines.append(str(report))

        return "\n".join(lines)

    def _format_recipe_recommendation(
        self, report: dict[str, Any], lines: list[str]
    ) -> str:
        """Format recipe recommendation as text."""
        header = report.get("header", {})
        lines.append(
            f"Product: {header.get('product')} | "
            f"Batch Volume: {header.get('batch_volume_liters')} L | "
            f"Generated: {report.get('generated_at', '')[:16]}"
        )
        lines.append("")

        # Target specs
        lines.append("TARGET SPECIFICATIONS:")
        for spec in report.get("target_specifications", []):
            lines.append(f"  {spec.get('display', '')}")
        lines.append("")

        # Raw materials
        lines.append("RAW MATERIALS AVAILABLE TODAY:")
        for mat in report.get("raw_materials_available", []):
            lines.append(f"  {mat['name']}: {mat['property']} | Cost: {mat['cost']}")
        lines.append("")

        # Optimized recipe table
        lines.append("OPTIMIZED BLEND RECIPE (Recommended by AI):")
        lines.append("-" * 60)
        lines.append(f"{'Material':<30} {'Qty (wt%)':<12} {'Cost/Unit':<12} {'Total Cost':<12}")
        lines.append("-" * 60)

        for ing in report.get("optimized_recipe", {}).get("ingredients", []):
            lines.append(
                f"{ing['material']:<30} "
                f"{ing['qty_wt_percent']:<12.1f} "
                f"{ing['cost_per_unit']:<12} "
                f"${ing['total_cost']:,.0f}"
            )

        lines.append("-" * 60)
        recipe = report.get("optimized_recipe", {})
        lines.append(f"{'TOTAL BATCH COST':<30} {'100.0%':<12} {'':<12} ${recipe.get('total_batch_cost', 0):,.0f}")
        lines.append(f"{'COST PER LITER':<30} {'':<12} {'':<12} ${recipe.get('cost_per_liter', 0):.2f}/L")
        lines.append("")

        # Predicted quality
        lines.append("PREDICTED OUTPUT QUALITY:")
        for pred in report.get("predicted_quality", []):
            status_symbol = "✓" if "PASS" in pred['status'] else ("⚠" if "MARGINAL" in pred['status'] else "✗")
            lines.append(
                f"  {pred['parameter']}: {pred['predicted_value']} {pred['unit']} "
                f"{status_symbol} ({pred['target_range']})"
            )
        lines.append("")

        # Confidence
        conf = report.get("confidence", {})
        lines.append(
            f"CONFIDENCE LEVEL: {conf.get('level_percent', 0):.1f}% "
            f"(Based on {conf.get('historical_batches_used', 0)} historical blends)"
        )
        lines.append("")

        # Cost comparison
        comp = report.get("cost_comparison", {})
        lines.append("COMPARISON TO STANDARD RECIPE:")
        lines.append(f"  Standard Recipe Cost: ${comp.get('standard_recipe_cost_per_liter', 0):.2f}/L")
        lines.append(f"  AI Optimized Cost: ${comp.get('ai_optimized_cost_per_liter', 0):.2f}/L")
        lines.append(f"  Savings per Batch: ${comp.get('savings_per_batch', 0):,.0f} ({comp.get('savings_percent', 0):.1f}% reduction)")
        lines.append(f"  Estimated Annual Savings: ${comp.get('annual_savings_estimate', 0):,.0f} ({comp.get('batches_per_year', 0)} batches/year)")
        lines.append("")

        # Notes
        lines.append("NOTES:")
        for note in report.get("notes", []):
            lines.append(f"  • {note}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _format_quality_prediction(
        self, report: dict[str, Any], lines: list[str]
    ) -> str:
        """Format quality prediction report as text."""
        header = report.get("header", {})
        lines.append(
            f"Blend ID: {header.get('blend_id')} | "
            f"Product: {header.get('product')} | "
            f"Volume: {header.get('volume_liters')} L"
        )
        lines.append("")

        lines.append("PREDICTED QUALITY METRICS (AI Forecast):")
        lines.append("-" * 60)
        lines.append(f"{'Parameter':<25} {'Predicted':<12} {'Target':<15} {'Status':<10}")
        lines.append("-" * 60)

        for metric in report.get("predicted_quality_metrics", []):
            status_symbol = "✓ PASS" if "PASS" in metric['status'] else (
                "⚠ MARGINAL" if "MARGINAL" in metric['status'] else "✗ FAIL"
            )
            lines.append(
                f"{metric['parameter']:<25} "
                f"{metric['predicted_value']:<12} "
                f"{metric['target']:<15} "
                f"{status_symbol}"
            )

        lines.append("-" * 60)
        lines.append("")

        conf = report.get("quality_prediction_confidence", {})
        lines.append(f"QUALITY PREDICTION CONFIDENCE: {conf.get('percent', 0):.1f}%")
        lines.append(f"({conf.get('note', '')})")
        lines.append("")

        risk = report.get("risk_assessment", {})
        lines.append(f"RISK ASSESSMENT: {risk.get('level', 'UNKNOWN')}")
        for factor in risk.get("factors", []):
            lines.append(f"  • {factor}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _format_cost_comparison(
        self, report: dict[str, Any], lines: list[str]
    ) -> str:
        """Format cost comparison report as text."""
        header = report.get("header", {})
        lines.append(
            f"Product: {header.get('product')} | "
            f"Batch Volume: {header.get('batch_volume_liters')} L"
        )
        lines.append("")

        lines.append("RECIPE COST COMPARISON:")
        lines.append("")

        legacy = report.get("legacy_recipe", {})
        lines.append("Legacy Recipe (Manual formulation):")
        for cat, cost in legacy.get("costs", {}).items():
            lines.append(f"  {cat}: ${cost:,.0f}")
        lines.append(f"  Total Cost: ${legacy.get('total_cost', 0):,.0f}")
        lines.append(f"  Cost/Liter: ${legacy.get('cost_per_liter', 0):.2f}")
        lines.append("")

        optimized = report.get("ai_optimized_recipe", {})
        lines.append("AI-Optimized Recipe (Recommended):")
        for cat, cost in optimized.get("costs", {}).items():
            lines.append(f"  {cat}: ${cost:,.0f}")
        lines.append(f"  Total Cost: ${optimized.get('total_cost', 0):,.0f}")
        lines.append(f"  Cost/Liter: ${optimized.get('cost_per_liter', 0):.2f}")
        lines.append("")

        comp = report.get("comparison", {})
        lines.append(f"BATCH SAVINGS: ${comp.get('batch_savings', 0):,.0f} per batch (-{comp.get('batch_savings_percent', 0):.1f}%)")
        lines.append(f"ANNUAL SAVINGS ({comp.get('batches_per_year', 0)} batches): ${comp.get('annual_savings', 0):,.0f}")
        lines.append("")

        lines.append("OPTIMIZATION ACHIEVED THROUGH:")
        for method in report.get("optimization_methods", []):
            lines.append(f"  ✓ {method}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def _format_variance_report(
        self, report: dict[str, Any], lines: list[str]
    ) -> str:
        """Format variance report as text."""
        header = report.get("header", {})
        lines.append(f"Period: Last {header.get('period_days')} Days")
        lines.append(f"Blend Batches Analyzed: {header.get('batches_analyzed')}")
        lines.append("")

        lines.append("QUALITY VARIANCE ANALYSIS:")
        lines.append("-" * 60)
        lines.append(f"{'Parameter':<25} {'Std Dev':<12} {'Target':<12} {'Trend':<8}")
        lines.append("-" * 60)

        for metric in report.get("quality_variance_analysis", []):
            status = "OK" if metric['within_target'] else "!"
            lines.append(
                f"{metric['parameter']:<25} "
                f"±{metric['std_dev']:<11} "
                f"±{metric['target_std']} {status:<6} "
                f"{metric['trend']}"
            )

        lines.append("-" * 60)
        lines.append("")

        perf = report.get("performance_improvement", {})
        lines.append("PERFORMANCE IMPROVEMENT:")
        lines.append(f"  Previous Period Off-Spec Rate: {perf.get('previous_period_off_spec', 'N/A')}")
        lines.append(f"  Current Period Off-Spec Rate: {perf.get('current_period_off_spec', 'N/A')}")
        lines.append(f"  Improvement: {perf.get('improvement', 'N/A')}")
        lines.append("")

        trend = report.get("consistency_trend", {})
        lines.append(f"CONSISTENCY TREND: {trend.get('status', 'UNKNOWN')}")
        for note in trend.get("notes", []):
            lines.append(f"  • {note}")

        lines.append("=" * 70)
        return "\n".join(lines)
