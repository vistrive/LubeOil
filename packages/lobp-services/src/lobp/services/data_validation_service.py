"""
Data Validation Pipeline for Historical Batch Imports.

Implements the Data Preparation Checklist from Section 3.3 of the
AI Recipe Optimization document:
- Remove duplicates
- Handle missing values
- Validate ranges
- Check timestamps
- Unit consistency
- Outlier review
- Format compliance

Ensures data quality for AI model training.
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class ValidationSeverity(str, Enum):
    """Severity level for validation issues."""

    ERROR = "error"  # Must be fixed before import
    WARNING = "warning"  # Should be reviewed
    INFO = "info"  # Informational


class ValidationAction(str, Enum):
    """Recommended action for validation issues."""

    REJECT = "reject"  # Reject the record
    FIX_AUTO = "fix_auto"  # Can be auto-fixed
    FIX_MANUAL = "fix_manual"  # Requires manual fix
    REVIEW = "review"  # Manual review recommended
    ACCEPT = "accept"  # Accept as-is


@dataclass
class ValidationIssue:
    """Single validation issue found in data."""

    row_number: int
    field_name: str
    issue_type: str
    message: str
    severity: ValidationSeverity
    action: ValidationAction
    current_value: Any = None
    suggested_value: Any = None


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    total_rows: int
    valid_rows: int
    invalid_rows: int
    issues: list[ValidationIssue] = field(default_factory=list)
    duplicates_found: int = 0
    missing_values_found: int = 0
    outliers_found: int = 0
    fixed_automatically: int = 0
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""

    generated_at: datetime
    source_file: str
    validation_result: ValidationResult
    completeness_score: float  # 0-100%
    consistency_score: float  # 0-100%
    accuracy_score: float  # 0-100%
    overall_score: float  # 0-100%
    recommendations: list[str] = field(default_factory=list)
    ready_for_training: bool = False


class DataValidationService:
    """
    Service for validating historical batch data before AI model training.

    Implements all validation rules from the document's Data Preparation Checklist.
    """

    # Default validation ranges (from document)
    VISCOSITY_40C_RANGE = (5.0, 500.0)  # cSt
    VISCOSITY_100C_RANGE = (1.0, 50.0)  # cSt
    VISCOSITY_INDEX_RANGE = (50, 200)
    TBN_RANGE = (0.0, 20.0)  # mgKOH/g
    TAN_RANGE = (0.0, 10.0)  # mgKOH/g
    FLASH_POINT_RANGE = (100.0, 350.0)  # °C
    POUR_POINT_RANGE = (-60.0, 20.0)  # °C
    DENSITY_RANGE = (0.7, 1.1)  # g/mL
    WATER_CONTENT_RANGE = (0.0, 1000.0)  # ppm
    FOAM_TEST_RANGE = (0.0, 500.0)  # mL
    ADDITIVE_PERCENTAGE_RANGE = (0.0, 100.0)
    BLEND_VOLUME_RANGE = (100.0, 100000.0)  # Liters
    TEMPERATURE_RANGE = (-20.0, 150.0)  # °C

    # Outlier detection thresholds (standard deviations)
    OUTLIER_STD_THRESHOLD = 3.0

    def __init__(self):
        """Initialize validation service."""
        self._validation_stats = {
            "total_validated": 0,
            "total_passed": 0,
            "total_failed": 0,
        }

    def validate_batch_history_csv(
        self,
        csv_content: str,
        source_name: str = "batch_history.csv",
        auto_fix: bool = True,
    ) -> tuple[ValidationResult, list[dict[str, Any]]]:
        """
        Validate batch history CSV data.

        Args:
            csv_content: CSV content as string
            source_name: Source file name for reporting
            auto_fix: Automatically fix minor issues

        Returns:
            Tuple of (ValidationResult, cleaned_data)
        """
        logger.info("Starting batch history CSV validation", source=source_name)

        issues: list[ValidationIssue] = []
        cleaned_data: list[dict[str, Any]] = []

        # Parse CSV
        try:
            reader = csv.DictReader(io.StringIO(csv_content))
            rows = list(reader)
        except Exception as e:
            issues.append(ValidationIssue(
                row_number=0,
                field_name="file",
                issue_type="parse_error",
                message=f"Failed to parse CSV: {str(e)}",
                severity=ValidationSeverity.ERROR,
                action=ValidationAction.REJECT,
            ))
            return ValidationResult(
                is_valid=False,
                total_rows=0,
                valid_rows=0,
                invalid_rows=0,
                issues=issues,
            ), []

        total_rows = len(rows)

        # Step 1: Check for duplicates
        seen_batch_ids = set()
        duplicates = 0

        for row_num, row in enumerate(rows, start=2):  # Start at 2 (after header)
            batch_id = row.get("Batch_ID", "").strip()

            if batch_id in seen_batch_ids:
                duplicates += 1
                issues.append(ValidationIssue(
                    row_number=row_num,
                    field_name="Batch_ID",
                    issue_type="duplicate",
                    message=f"Duplicate Batch_ID: {batch_id}",
                    severity=ValidationSeverity.ERROR,
                    action=ValidationAction.REJECT,
                    current_value=batch_id,
                ))
                continue

            if batch_id:
                seen_batch_ids.add(batch_id)

            # Step 2: Validate required fields
            row_issues, row_fixed = self._validate_batch_row(row, row_num, auto_fix)
            issues.extend(row_issues)

            # Only include if no critical errors
            has_critical = any(
                i.severity == ValidationSeverity.ERROR and i.action == ValidationAction.REJECT
                for i in row_issues
            )

            if not has_critical:
                cleaned_data.append(row_fixed)

        # Step 3: Check timestamp ordering
        timestamp_issues = self._validate_timestamp_ordering(cleaned_data)
        issues.extend(timestamp_issues)

        # Step 4: Detect outliers
        outlier_issues = self._detect_outliers(cleaned_data)
        issues.extend(outlier_issues)

        # Calculate statistics
        valid_rows = len(cleaned_data)
        invalid_rows = total_rows - valid_rows
        fixed_auto = sum(1 for i in issues if i.action == ValidationAction.FIX_AUTO)

        result = ValidationResult(
            is_valid=invalid_rows == 0 and duplicates == 0,
            total_rows=total_rows,
            valid_rows=valid_rows,
            invalid_rows=invalid_rows,
            issues=issues,
            duplicates_found=duplicates,
            missing_values_found=sum(1 for i in issues if i.issue_type == "missing_value"),
            outliers_found=sum(1 for i in issues if i.issue_type == "outlier"),
            fixed_automatically=fixed_auto,
            summary={
                "total_rows": total_rows,
                "valid_rows": valid_rows,
                "duplicate_rows": duplicates,
                "error_count": sum(1 for i in issues if i.severity == ValidationSeverity.ERROR),
                "warning_count": sum(1 for i in issues if i.severity == ValidationSeverity.WARNING),
            },
        )

        logger.info(
            "Batch history validation complete",
            total=total_rows,
            valid=valid_rows,
            invalid=invalid_rows,
            duplicates=duplicates,
        )

        return result, cleaned_data

    def _validate_batch_row(
        self,
        row: dict[str, Any],
        row_num: int,
        auto_fix: bool,
    ) -> tuple[list[ValidationIssue], dict[str, Any]]:
        """Validate a single batch row."""
        issues = []
        fixed_row = dict(row)

        # Required fields validation
        required_fields = ["Batch_ID", "Blend_Date", "Base_Oil_Type", "Blend_Volume_L"]
        for field in required_fields:
            value = row.get(field, "").strip() if row.get(field) else ""
            if not value:
                issues.append(ValidationIssue(
                    row_number=row_num,
                    field_name=field,
                    issue_type="missing_value",
                    message=f"Required field '{field}' is missing",
                    severity=ValidationSeverity.ERROR,
                    action=ValidationAction.REJECT,
                ))

        # Validate date format
        blend_date = row.get("Blend_Date", "")
        if blend_date:
            try:
                # Try common date formats
                parsed_date = None
                for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"]:
                    try:
                        parsed_date = datetime.strptime(blend_date, fmt)
                        break
                    except ValueError:
                        continue

                if not parsed_date:
                    issues.append(ValidationIssue(
                        row_number=row_num,
                        field_name="Blend_Date",
                        issue_type="invalid_format",
                        message=f"Invalid date format: {blend_date}",
                        severity=ValidationSeverity.ERROR,
                        action=ValidationAction.FIX_MANUAL,
                        current_value=blend_date,
                    ))
                elif auto_fix:
                    # Standardize to YYYY-MM-DD
                    fixed_row["Blend_Date"] = parsed_date.strftime("%Y-%m-%d")
            except Exception as e:
                issues.append(ValidationIssue(
                    row_number=row_num,
                    field_name="Blend_Date",
                    issue_type="invalid_format",
                    message=f"Date parsing error: {str(e)}",
                    severity=ValidationSeverity.ERROR,
                    action=ValidationAction.FIX_MANUAL,
                    current_value=blend_date,
                ))

        # Validate numeric ranges
        numeric_validations = [
            ("Base_Oil_Viscosity_cSt", self.VISCOSITY_40C_RANGE),
            ("Blend_Volume_L", self.BLEND_VOLUME_RANGE),
            ("Temperature_Blending_C", self.TEMPERATURE_RANGE),
            ("Viscosity_40C_cSt", self.VISCOSITY_40C_RANGE),
            ("Viscosity_100C_cSt", self.VISCOSITY_100C_RANGE),
            ("Viscosity_Index", self.VISCOSITY_INDEX_RANGE),
            ("TBN_mgKOH", self.TBN_RANGE),
            ("Pour_Point_C", self.POUR_POINT_RANGE),
            ("Flash_Point_C", self.FLASH_POINT_RANGE),
            ("Water_Content_ppm", self.WATER_CONTENT_RANGE),
            ("Foam_Test_ML", self.FOAM_TEST_RANGE),
        ]

        for field_name, (min_val, max_val) in numeric_validations:
            value = row.get(field_name, "")
            if value is not None and str(value).strip():
                try:
                    num_value = float(value)
                    if num_value < min_val or num_value > max_val:
                        issues.append(ValidationIssue(
                            row_number=row_num,
                            field_name=field_name,
                            issue_type="out_of_range",
                            message=f"Value {num_value} out of range [{min_val}, {max_val}]",
                            severity=ValidationSeverity.WARNING,
                            action=ValidationAction.REVIEW,
                            current_value=num_value,
                        ))
                except ValueError:
                    issues.append(ValidationIssue(
                        row_number=row_num,
                        field_name=field_name,
                        issue_type="invalid_type",
                        message=f"Expected numeric value, got: {value}",
                        severity=ValidationSeverity.ERROR,
                        action=ValidationAction.FIX_MANUAL,
                        current_value=value,
                    ))

        # Validate additive percentages sum to ~100%
        additive_fields = [
            "Additive1_Qty_wt%", "Additive2_Qty_wt%",
            "Additive3_Qty_wt%", "Additive4_Qty_wt%",
        ]
        base_oil_pct = self._safe_float(row.get("Base_Oil_Percentage", 0))
        additive_sum = sum(
            self._safe_float(row.get(f, 0)) for f in additive_fields
        )

        total_pct = base_oil_pct + additive_sum
        if total_pct > 0 and abs(total_pct - 100.0) > 2.0:  # Allow 2% tolerance
            issues.append(ValidationIssue(
                row_number=row_num,
                field_name="percentages",
                issue_type="sum_mismatch",
                message=f"Component percentages sum to {total_pct:.1f}%, expected ~100%",
                severity=ValidationSeverity.WARNING,
                action=ValidationAction.REVIEW,
                current_value=total_pct,
                suggested_value=100.0,
            ))

        # Handle missing values with forward-fill option
        if auto_fix:
            fixed_row = self._handle_missing_values(fixed_row, issues)

        return issues, fixed_row

    def _handle_missing_values(
        self,
        row: dict[str, Any],
        issues: list[ValidationIssue],
    ) -> dict[str, Any]:
        """Handle missing values using appropriate strategies."""
        # Fields that can use default values
        defaults = {
            "Off_Spec_Flag": 0,
            "Specification_Met": "PASS",
        }

        for field, default in defaults.items():
            if not row.get(field) or str(row.get(field, "")).strip() == "":
                row[field] = default
                issues.append(ValidationIssue(
                    row_number=0,
                    field_name=field,
                    issue_type="missing_value",
                    message=f"Missing value filled with default: {default}",
                    severity=ValidationSeverity.INFO,
                    action=ValidationAction.FIX_AUTO,
                    suggested_value=default,
                ))

        return row

    def _validate_timestamp_ordering(
        self,
        data: list[dict[str, Any]],
    ) -> list[ValidationIssue]:
        """Validate that timestamps are in ascending order."""
        issues = []

        prev_date = None
        for i, row in enumerate(data):
            date_str = row.get("Blend_Date", "")
            if date_str:
                try:
                    current_date = datetime.strptime(date_str, "%Y-%m-%d")
                    if prev_date and current_date < prev_date:
                        issues.append(ValidationIssue(
                            row_number=i + 2,
                            field_name="Blend_Date",
                            issue_type="timestamp_order",
                            message=f"Date {date_str} is earlier than previous row",
                            severity=ValidationSeverity.WARNING,
                            action=ValidationAction.REVIEW,
                            current_value=date_str,
                        ))
                    prev_date = current_date
                except ValueError:
                    pass

        return issues

    def _detect_outliers(
        self,
        data: list[dict[str, Any]],
    ) -> list[ValidationIssue]:
        """Detect statistical outliers in numeric fields."""
        issues = []

        if len(data) < 10:  # Need minimum data for outlier detection
            return issues

        numeric_fields = [
            "Base_Oil_Viscosity_cSt",
            "Viscosity_40C_cSt",
            "Viscosity_100C_cSt",
            "TBN_mgKOH",
            "Flash_Point_C",
            "Blend_Volume_L",
        ]

        for field in numeric_fields:
            values = []
            for row in data:
                val = self._safe_float(row.get(field))
                if val is not None:
                    values.append(val)

            if len(values) < 5:
                continue

            # Calculate mean and std
            import statistics
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0

            if std == 0:
                continue

            # Find outliers
            for i, row in enumerate(data):
                val = self._safe_float(row.get(field))
                if val is not None:
                    z_score = abs(val - mean) / std
                    if z_score > self.OUTLIER_STD_THRESHOLD:
                        issues.append(ValidationIssue(
                            row_number=i + 2,
                            field_name=field,
                            issue_type="outlier",
                            message=f"Outlier detected: {val} (z-score: {z_score:.2f})",
                            severity=ValidationSeverity.WARNING,
                            action=ValidationAction.REVIEW,
                            current_value=val,
                            suggested_value=f"Mean: {mean:.2f}, Std: {std:.2f}",
                        ))

        return issues

    def validate_quality_results_csv(
        self,
        csv_content: str,
        source_name: str = "quality_results.csv",
    ) -> tuple[ValidationResult, list[dict[str, Any]]]:
        """
        Validate quality results CSV data.

        Validates the format from Section 3.1B of the document.
        """
        logger.info("Starting quality results CSV validation", source=source_name)

        issues: list[ValidationIssue] = []
        cleaned_data: list[dict[str, Any]] = []

        try:
            reader = csv.DictReader(io.StringIO(csv_content))
            rows = list(reader)
        except Exception as e:
            issues.append(ValidationIssue(
                row_number=0,
                field_name="file",
                issue_type="parse_error",
                message=f"Failed to parse CSV: {str(e)}",
                severity=ValidationSeverity.ERROR,
                action=ValidationAction.REJECT,
            ))
            return ValidationResult(
                is_valid=False,
                total_rows=0,
                valid_rows=0,
                invalid_rows=0,
                issues=issues,
            ), []

        for row_num, row in enumerate(rows, start=2):
            row_issues = []

            # Validate Batch_ID
            batch_id = row.get("Batch_ID", "").strip()
            if not batch_id:
                row_issues.append(ValidationIssue(
                    row_number=row_num,
                    field_name="Batch_ID",
                    issue_type="missing_value",
                    message="Batch_ID is required",
                    severity=ValidationSeverity.ERROR,
                    action=ValidationAction.REJECT,
                ))

            # Validate Off_Spec_Flag
            off_spec = row.get("Off_Spec_Flag", "")
            if off_spec and off_spec not in ["0", "1", 0, 1]:
                row_issues.append(ValidationIssue(
                    row_number=row_num,
                    field_name="Off_Spec_Flag",
                    issue_type="invalid_value",
                    message=f"Off_Spec_Flag must be 0 or 1, got: {off_spec}",
                    severity=ValidationSeverity.WARNING,
                    action=ValidationAction.FIX_MANUAL,
                    current_value=off_spec,
                ))

            # Validate Specification_Met
            spec_met = row.get("Specification_Met", "")
            if spec_met and spec_met.upper() not in ["PASS", "FAIL"]:
                row_issues.append(ValidationIssue(
                    row_number=row_num,
                    field_name="Specification_Met",
                    issue_type="invalid_value",
                    message=f"Specification_Met must be PASS or FAIL, got: {spec_met}",
                    severity=ValidationSeverity.WARNING,
                    action=ValidationAction.FIX_MANUAL,
                    current_value=spec_met,
                ))

            issues.extend(row_issues)

            has_critical = any(
                i.severity == ValidationSeverity.ERROR and i.action == ValidationAction.REJECT
                for i in row_issues
            )

            if not has_critical:
                cleaned_data.append(row)

        return ValidationResult(
            is_valid=len(cleaned_data) == len(rows),
            total_rows=len(rows),
            valid_rows=len(cleaned_data),
            invalid_rows=len(rows) - len(cleaned_data),
            issues=issues,
        ), cleaned_data

    def validate_cost_data_csv(
        self,
        csv_content: str,
        source_name: str = "cost_data.csv",
    ) -> tuple[ValidationResult, list[dict[str, Any]]]:
        """
        Validate cost data CSV.

        Validates the format from Section 3.1C of the document.
        """
        logger.info("Starting cost data CSV validation", source=source_name)

        issues: list[ValidationIssue] = []
        cleaned_data: list[dict[str, Any]] = []

        try:
            reader = csv.DictReader(io.StringIO(csv_content))
            rows = list(reader)
        except Exception as e:
            issues.append(ValidationIssue(
                row_number=0,
                field_name="file",
                issue_type="parse_error",
                message=f"Failed to parse CSV: {str(e)}",
                severity=ValidationSeverity.ERROR,
                action=ValidationAction.REJECT,
            ))
            return ValidationResult(
                is_valid=False,
                total_rows=0,
                valid_rows=0,
                invalid_rows=0,
                issues=issues,
            ), []

        for row_num, row in enumerate(rows, start=2):
            row_issues = []

            # Required fields
            for field in ["Material_Code", "Material_Name", "Supplier", "Cost_per_Unit", "Unit_Type"]:
                if not row.get(field, "").strip():
                    row_issues.append(ValidationIssue(
                        row_number=row_num,
                        field_name=field,
                        issue_type="missing_value",
                        message=f"Required field '{field}' is missing",
                        severity=ValidationSeverity.ERROR,
                        action=ValidationAction.REJECT,
                    ))

            # Validate cost is positive number
            cost = row.get("Cost_per_Unit", "")
            if cost:
                try:
                    cost_val = float(str(cost).replace("$", "").replace(",", ""))
                    if cost_val <= 0:
                        row_issues.append(ValidationIssue(
                            row_number=row_num,
                            field_name="Cost_per_Unit",
                            issue_type="invalid_value",
                            message=f"Cost must be positive, got: {cost_val}",
                            severity=ValidationSeverity.ERROR,
                            action=ValidationAction.FIX_MANUAL,
                            current_value=cost_val,
                        ))
                except ValueError:
                    row_issues.append(ValidationIssue(
                        row_number=row_num,
                        field_name="Cost_per_Unit",
                        issue_type="invalid_type",
                        message=f"Invalid cost value: {cost}",
                        severity=ValidationSeverity.ERROR,
                        action=ValidationAction.FIX_MANUAL,
                        current_value=cost,
                    ))

            # Validate unit type
            unit_type = row.get("Unit_Type", "").lower()
            valid_units = ["per_liter", "per_kg", "per_drum", "per_gallon", "liter", "kg"]
            if unit_type and unit_type not in valid_units:
                row_issues.append(ValidationIssue(
                    row_number=row_num,
                    field_name="Unit_Type",
                    issue_type="invalid_value",
                    message=f"Invalid unit type: {unit_type}",
                    severity=ValidationSeverity.WARNING,
                    action=ValidationAction.REVIEW,
                    current_value=unit_type,
                ))

            issues.extend(row_issues)

            has_critical = any(
                i.severity == ValidationSeverity.ERROR and i.action == ValidationAction.REJECT
                for i in row_issues
            )

            if not has_critical:
                cleaned_data.append(row)

        return ValidationResult(
            is_valid=len(cleaned_data) == len(rows),
            total_rows=len(rows),
            valid_rows=len(cleaned_data),
            invalid_rows=len(rows) - len(cleaned_data),
            issues=issues,
        ), cleaned_data

    def generate_quality_report(
        self,
        validation_result: ValidationResult,
        source_file: str,
    ) -> DataQualityReport:
        """Generate comprehensive data quality report."""
        # Calculate scores
        completeness = (
            validation_result.valid_rows / validation_result.total_rows * 100
            if validation_result.total_rows > 0 else 0
        )

        error_count = sum(
            1 for i in validation_result.issues
            if i.severity == ValidationSeverity.ERROR
        )
        consistency = max(0, 100 - (error_count / max(validation_result.total_rows, 1) * 100))

        outlier_count = validation_result.outliers_found
        accuracy = max(0, 100 - (outlier_count / max(validation_result.total_rows, 1) * 50))

        overall = (completeness + consistency + accuracy) / 3

        # Generate recommendations
        recommendations = []

        if validation_result.duplicates_found > 0:
            recommendations.append(
                f"Remove {validation_result.duplicates_found} duplicate records"
            )

        if validation_result.missing_values_found > 0:
            recommendations.append(
                f"Address {validation_result.missing_values_found} missing values"
            )

        if validation_result.outliers_found > 0:
            recommendations.append(
                f"Review {validation_result.outliers_found} outlier values for accuracy"
            )

        if overall >= 90:
            recommendations.append("Data quality is excellent - ready for AI model training")
        elif overall >= 70:
            recommendations.append("Data quality is acceptable - review flagged issues before training")
        else:
            recommendations.append("Data quality needs improvement before AI model training")

        return DataQualityReport(
            generated_at=datetime.now(),
            source_file=source_file,
            validation_result=validation_result,
            completeness_score=completeness,
            consistency_score=consistency,
            accuracy_score=accuracy,
            overall_score=overall,
            recommendations=recommendations,
            ready_for_training=overall >= 80 and error_count == 0,
        )

    def _safe_float(self, value: Any) -> float | None:
        """Safely convert value to float."""
        if value is None:
            return None
        try:
            return float(str(value).replace(",", "").replace("$", "").strip())
        except (ValueError, TypeError):
            return None
