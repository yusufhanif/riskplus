"""Guided Streamlit workflow helpers for RiskPlus."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from riskplus_core.constants import DEFAULT_CONFIDENCE, MIN_OBSERVATIONS
from riskplus_core.data import infer_fund_name_from_file, read_uploaded_file
from riskplus_core.data_sources import (
    prepare_combined_wide_file,
    prepare_separate_fund_files,
    prepare_wide_fund_file_plus_factor_file,
)
from riskplus_core.models import NormalizedDataSource
from riskplus_core.quality import validate_selected_panel
from riskplus_core.weights import (
    build_asset_weight_series,
    detect_weight_columns,
    match_weight_names_to_assets,
    normalize_portfolio_weights,
    prepare_weights_table,
    validate_portfolio_weights,
)


@dataclass(slots=True)
class UploadContext:
    mode: str
    combined_upload: Any | None = None
    wide_fund_upload: Any | None = None
    wide_factor_upload: Any | None = None
    fund_uploads: list[Any] = field(default_factory=list)
    factor_upload: Any | None = None


@dataclass(slots=True)
class MappingResult:
    bundle: NormalizedDataSource | None
    status: str
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WeightResult:
    weights: pd.Series | None
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "incomplete"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisSettings:
    report_name: str
    portfolio_value: float
    rf_rate: float
    confidence: float
    num_sims: int
    corr_method: str
    ewma_decay: float
    dist_type: str
    show_factor_buckets: bool
    display_mode: str
    values_in_percent: bool
    max_missing_pct: float


@dataclass(slots=True)
class GuidedWorkflowState:
    upload_context: UploadContext
    mapping_result: MappingResult
    weight_result: WeightResult
    analysis_settings: AnalysisSettings
    run_requested: bool

    @property
    def can_run(self) -> bool:
        return self.mapping_result.bundle is not None and self.weight_result.weights is not None and not self.mapping_result.errors and not self.weight_result.errors


def _render_step_header(step_number: int, title: str, status: str, description: str | None = None) -> None:
    status_labels = {
        "complete": "Complete",
        "warning": "Warning",
        "error": "Error",
        "incomplete": "Incomplete",
    }
    st.markdown(f"### {step_number}. {title}")
    if description:
        st.caption(description)
    label = status_labels.get(status, "Incomplete")
    if status == "complete":
        st.success(label)
    elif status == "warning":
        st.warning(label)
    elif status == "error":
        st.error(label)
    else:
        st.info(label)


def _status_from_messages(errors: list[str], warnings: list[str], has_bundle: bool) -> str:
    if errors:
        return "error"
    if warnings:
        return "warning"
    if has_bundle:
        return "complete"
    return "incomplete"


def _make_weight_preview_table(asset_cols: list[str], raw_weights: pd.Series, normalized_weights: pd.Series, status_message: str) -> pd.DataFrame:
    raw_values = raw_weights.reindex(asset_cols).fillna(0.0).to_numpy(dtype=float)
    normalized_values = normalized_weights.reindex(asset_cols).fillna(0.0).to_numpy(dtype=float)
    preview = pd.DataFrame({'Fund': asset_cols})
    preview['Weight (%)'] = raw_values * 100.0
    preview['Normalized Weight (%)'] = normalized_values * 100.0
    preview['Status'] = status_message
    return preview


def _read_weights_upload(uploaded_file, sheet_name: str | None = None) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith('.csv'):
        return pd.read_csv(BytesIO(uploaded_file.getvalue()))
    if sheet_name is None:
        return pd.read_excel(BytesIO(uploaded_file.getvalue()))
    return pd.read_excel(BytesIO(uploaded_file.getvalue()), sheet_name=sheet_name)


def render_upload_step() -> UploadContext:
    st.markdown("### 1. Upload Data")
    st.caption("Choose one of the supported upload modes.")

    data_source_mode = st.selectbox(
        "Data source mode",
        options=[
            "Combined wide file",
            "Wide fund file + separate factor file",
            "Separate fund files",
        ],
        index=st.session_state.get('workflow_upload_mode_index', 2),
        key="workflow_data_source_mode",
    )
    st.session_state['workflow_upload_mode'] = data_source_mode
    st.session_state['workflow_upload_mode_index'] = [
        "Combined wide file",
        "Wide fund file + separate factor file",
        "Separate fund files",
    ].index(data_source_mode)

    if data_source_mode == "Combined wide file":
        combined_upload = st.file_uploader("Upload combined wide file", type=["csv", "xlsx", "xls"], key="workflow_combined_upload")
        status = "complete" if combined_upload is not None else "incomplete"
        if status == "complete":
            st.success("Upload complete.")
        else:
            st.info("Upload a combined wide file to continue.")
        return UploadContext(mode=data_source_mode, combined_upload=combined_upload)

    if data_source_mode == "Wide fund file + separate factor file":
        wide_fund_upload = st.file_uploader("Upload wide fund file", type=["csv", "xlsx", "xls"], key="workflow_wide_fund_upload")
        wide_factor_upload = st.file_uploader("Upload factor file", type=["csv", "xlsx", "xls"], key="workflow_wide_factor_upload")
        status = "complete" if wide_fund_upload is not None and wide_factor_upload is not None else "incomplete"
        if status == "complete":
            st.success("Upload complete.")
        else:
            st.info("Upload both the fund file and the factor file to continue.")
        return UploadContext(mode=data_source_mode, wide_fund_upload=wide_fund_upload, wide_factor_upload=wide_factor_upload)

    fund_uploads = st.file_uploader(
        "Upload one file per fund",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="workflow_separate_fund_uploads",
    )
    factor_upload = st.file_uploader("Upload factor returns file", type=["csv", "xlsx", "xls"], key="workflow_separate_factor_upload")
    status = "complete" if fund_uploads and factor_upload is not None else "incomplete"
    if status == "complete":
        st.success("Upload complete.")
    else:
        st.info("Upload at least one fund file and the factor returns file to continue.")
    return UploadContext(mode=data_source_mode, fund_uploads=list(fund_uploads or []), factor_upload=factor_upload)


def _build_combined_mapping(upload_context: UploadContext, values_in_percent: bool, max_missing_pct: float) -> MappingResult:
    if upload_context.combined_upload is None:
        return MappingResult(bundle=None, status="incomplete")

    combined_raw = read_uploaded_file(upload_context.combined_upload.name, upload_context.combined_upload.getvalue())
    combined_date_col = st.selectbox("Date column", options=combined_raw.columns.tolist(), index=0, key="workflow_combined_date_col")
    combined_value_cols = [col for col in combined_raw.columns if col != combined_date_col]
    combined_fund_cols = st.multiselect("Fund return columns", options=combined_value_cols, default=[], key="workflow_combined_fund_cols")
    combined_factor_cols = st.multiselect(
        "Factor return columns",
        options=[col for col in combined_value_cols if col not in combined_fund_cols],
        default=[],
        key="workflow_combined_factor_cols",
    )

    errors: list[str] = []
    warnings: list[str] = []
    if not combined_fund_cols:
        errors.append('Select at least one fund return column.')
    if not combined_factor_cols:
        errors.append('Select at least one factor return column.')

    validation_errors, validation_warnings = validate_selected_panel(
        combined_raw,
        combined_date_col,
        [*combined_fund_cols, *combined_factor_cols],
        MIN_OBSERVATIONS,
        max_missing_pct,
    )
    errors.extend(validation_errors)
    warnings.extend(validation_warnings)

    if errors:
        return MappingResult(bundle=None, status=_status_from_messages(errors, warnings, False), warnings=warnings, errors=errors)

    bundle = prepare_combined_wide_file(
        combined_raw,
        combined_date_col,
        combined_fund_cols,
        combined_factor_cols,
        values_in_percent,
    )
    return MappingResult(bundle=bundle, status=_status_from_messages(errors, warnings, True), warnings=warnings, errors=errors)


def _build_wide_fund_plus_factor_mapping(upload_context: UploadContext, values_in_percent: bool, max_missing_pct: float) -> MappingResult:
    if upload_context.wide_fund_upload is None or upload_context.wide_factor_upload is None:
        return MappingResult(bundle=None, status="incomplete")

    fund_raw = read_uploaded_file(upload_context.wide_fund_upload.name, upload_context.wide_fund_upload.getvalue())
    factor_raw = read_uploaded_file(upload_context.wide_factor_upload.name, upload_context.wide_factor_upload.getvalue())

    fund_date_col = st.selectbox("Fund file date column", options=fund_raw.columns.tolist(), index=0, key="workflow_fund_date_col")
    fund_cols = st.multiselect(
        "Fund return columns",
        options=[col for col in fund_raw.columns if col != fund_date_col],
        default=[],
        key="workflow_fund_cols",
    )
    factor_date_col = st.selectbox("Factor file date column", options=factor_raw.columns.tolist(), index=0, key="workflow_factor_date_col")
    factor_cols = st.multiselect(
        "Factor return columns",
        options=[col for col in factor_raw.columns if col != factor_date_col],
        default=[],
        key="workflow_factor_cols",
    )

    errors: list[str] = []
    warnings: list[str] = []
    if not fund_cols:
        errors.append('Select at least one fund return column.')
    if not factor_cols:
        errors.append('Select at least one factor return column.')

    fund_errors, fund_warnings = validate_selected_panel(fund_raw, fund_date_col, fund_cols, MIN_OBSERVATIONS, max_missing_pct)
    factor_errors, factor_warnings = validate_selected_panel(factor_raw, factor_date_col, factor_cols, MIN_OBSERVATIONS, max_missing_pct)
    errors.extend(fund_errors)
    errors.extend(factor_errors)
    warnings.extend(fund_warnings)
    warnings.extend(factor_warnings)

    if errors:
        return MappingResult(bundle=None, status=_status_from_messages(errors, warnings, False), warnings=warnings, errors=errors)

    bundle = prepare_wide_fund_file_plus_factor_file(
        fund_raw,
        fund_date_col,
        fund_cols,
        factor_raw,
        factor_date_col,
        factor_cols,
        values_in_percent,
    )
    return MappingResult(bundle=bundle, status=_status_from_messages(errors, warnings, True), warnings=warnings, errors=errors)


def _build_separate_fund_mapping(upload_context: UploadContext, values_in_percent: bool, max_missing_pct: float) -> MappingResult:
    if not upload_context.fund_uploads or upload_context.factor_upload is None:
        return MappingResult(bundle=None, status="incomplete")

    fund_file_specs: list[dict[str, Any]] = []
    warnings: list[str] = []
    errors: list[str] = []

    for upload in upload_context.fund_uploads:
        st.caption(upload.name)
        fund_raw = read_uploaded_file(upload.name, upload.getvalue())
        date_col = st.selectbox(
            f"Date column for {upload.name}",
            options=fund_raw.columns.tolist(),
            index=0,
            key=f"workflow_date_{upload.name}",
        )
        numeric_guess = [
            col for col in fund_raw.columns
            if col != date_col and pd.to_numeric(fund_raw[col], errors='coerce').notna().mean() > 0.5
        ]
        if not numeric_guess:
            errors.append(f"{upload.name} has no numeric return columns.")
            continue

        return_col = st.selectbox(
            f"Return column for {upload.name}",
            options=numeric_guess,
            index=0,
            key=f"workflow_return_{upload.name}",
        )
        fund_name = st.text_input(
            f"Fund name for {upload.name}",
            value=infer_fund_name_from_file(upload.name),
            key=f"workflow_name_{upload.name}",
        )

        fund_errors, fund_warnings = validate_selected_panel(fund_raw, date_col, [return_col], MIN_OBSERVATIONS, max_missing_pct)
        warnings.extend(fund_warnings)
        errors.extend(fund_errors)
        fund_file_specs.append(
            {
                'raw_df': fund_raw,
                'date_col': date_col,
                'return_col': return_col,
                'fund_name': fund_name,
            }
        )

    factor_raw = read_uploaded_file(upload_context.factor_upload.name, upload_context.factor_upload.getvalue())
    factor_date_col = st.selectbox("Factor file date column", options=factor_raw.columns.tolist(), index=0, key="workflow_separate_factor_date_col")
    factor_cols = st.multiselect(
        "Factor return columns",
        options=[col for col in factor_raw.columns if col != factor_date_col],
        default=[],
        key="workflow_separate_factor_cols",
    )

    if not factor_cols:
        errors.append('Select at least one factor return column.')

    factor_errors, factor_warnings = validate_selected_panel(
        factor_raw,
        factor_date_col,
        factor_cols,
        MIN_OBSERVATIONS,
        max_missing_pct,
    )
    warnings.extend(factor_warnings)
    errors.extend(factor_errors)

    if errors:
        return MappingResult(bundle=None, status=_status_from_messages(errors, warnings, False), warnings=warnings, errors=errors)

    bundle = prepare_separate_fund_files(
        fund_file_specs,
        factor_raw,
        factor_date_col,
        factor_cols,
        values_in_percent,
    )
    return MappingResult(bundle=bundle, status=_status_from_messages(errors, warnings, True), warnings=warnings, errors=errors)


def render_mapping_step(upload_context: UploadContext, values_in_percent: bool, max_missing_pct: float) -> MappingResult:
    if upload_context.mode == "Combined wide file":
        result = _build_combined_mapping(upload_context, values_in_percent, max_missing_pct)
    elif upload_context.mode == "Wide fund file + separate factor file":
        result = _build_wide_fund_plus_factor_mapping(upload_context, values_in_percent, max_missing_pct)
    else:
        result = _build_separate_fund_mapping(upload_context, values_in_percent, max_missing_pct)

    _render_step_header(
        2,
        "Map Columns",
        result.status,
        "Choose the date, fund, and factor columns for the selected upload mode.",
    )

    if result.bundle is None:
        if not result.errors and not result.warnings:
            st.info("Upload the required files to continue.")
        return result

    st.caption(f"Mapped funds: {', '.join(result.bundle.asset_cols)}")
    st.caption(f"Mapped factors: {', '.join(result.bundle.factor_cols)}")
    return result


def render_validation_step(mapping_result: MappingResult) -> None:
    _render_step_header(
        3,
        "Validate Data",
        mapping_result.status,
        "Review warnings and errors before running the analysis.",
    )

    if mapping_result.bundle is None:
        if mapping_result.errors:
            for error in mapping_result.errors:
                st.error(error)
        elif mapping_result.warnings:
            for warning in mapping_result.warnings:
                st.warning(warning)
        else:
            st.info("Complete the mapping step to see validation feedback.")
        return

    metadata = mapping_result.bundle.data_source_metadata
    validation_table = pd.DataFrame(
        [
            {'metric': 'source mode', 'value': metadata.get('mode', 'unknown')},
            {'metric': 'alignment method', 'value': metadata.get('alignment_method', 'unknown')},
            {'metric': 'fund history', 'value': metadata.get('fund_history_label', 'unknown')},
            {'metric': 'factor history', 'value': metadata.get('factor_history_label', 'unknown')},
            {'metric': 'overlap period', 'value': metadata.get('overlap_label', 'unknown')},
            {'metric': 'fund duplicate dates', 'value': metadata.get('fund_duplicate_date_count', 0)},
            {'metric': 'factor duplicate dates', 'value': metadata.get('factor_duplicate_date_count', 0)},
            {'metric': 'fund extreme returns', 'value': metadata.get('fund_extreme_return_count', 0)},
            {'metric': 'factor extreme returns', 'value': metadata.get('factor_extreme_return_count', 0)},
        ]
    )
    st.dataframe(validation_table, hide_index=True, use_container_width=True)

    if mapping_result.warnings:
        for warning in mapping_result.warnings:
            st.warning(warning)
    else:
        st.success("Selected data passed validation checks.")

    if mapping_result.errors:
        for error in mapping_result.errors:
            st.error(error)


def render_weights_step(asset_cols: list[str], base_weights: pd.Series | None = None) -> WeightResult:
    status = "complete" if asset_cols else "incomplete"
    _render_step_header(
        4,
        "Set Portfolio Weights",
        status,
        "Choose equal, manual, or uploaded portfolio weights.",
    )

    if not asset_cols:
        st.info("Map the fund columns first.")
        return WeightResult(weights=None, status="incomplete")

    weight_method = st.selectbox(
        "Weight input method",
        options=["Equal weight", "Manual editable table", "Upload weights file", "Read weights from workbook sheet"],
        index=0,
        key="workflow_weight_input_method",
    )

    initial_weights = base_weights if base_weights is not None and not base_weights.empty else pd.Series(1.0 / len(asset_cols), index=asset_cols, dtype=float)
    raw_weights = initial_weights.reindex(asset_cols).fillna(0.0).astype(float)
    metadata: dict[str, Any] = {
        'weight_input_method': weight_method,
        'weights_sum_before_normalization': float(raw_weights.sum()),
        'weights_sum_after_normalization': None,
        'missing_weight_funds': [],
        'extra_weight_rows': [],
        'normalization_applied': False,
        'negative_weight_detected': False,
        'zero_weight_funds': [],
    }

    if weight_method == "Equal weight":
        normalized_weights = normalize_portfolio_weights(pd.Series(1.0, index=asset_cols, dtype=float))
        metadata['weights_sum_after_normalization'] = float(normalized_weights.sum())
        metadata['normalization_applied'] = True
        preview = _make_weight_preview_table(asset_cols, normalized_weights, normalized_weights, 'Equal weight')
        st.dataframe(preview, hide_index=True, use_container_width=True)
        st.success("Weights normalized to 100%.")
        return WeightResult(weights=normalized_weights, metadata=metadata, status="complete")

    if weight_method == "Manual editable table":
        editor_table = pd.DataFrame({'Fund': asset_cols, 'Weight (%)': (raw_weights * 100.0).round(6)})
        edited_table = st.data_editor(
            editor_table,
            hide_index=True,
            num_rows='fixed',
            use_container_width=True,
            disabled=['Fund'],
            key='workflow_manual_weight_editor',
        )
        normalize_to_100 = st.checkbox('Normalize weights to 100%', value=True, key='workflow_manual_weight_normalize')
        entered_decimal = pd.Series(pd.to_numeric(edited_table['Weight (%)'], errors='coerce').fillna(0.0).values / 100.0, index=edited_table['Fund'])
        metadata['weights_sum_before_normalization'] = float(entered_decimal.sum())
        if (entered_decimal < 0).any():
            st.error('Negative weights are not allowed.')
            return WeightResult(weights=None, metadata=metadata, status='error', errors=['Negative weights are not allowed.'])
        if normalize_to_100:
            normalized_weights = normalize_portfolio_weights(entered_decimal)
            metadata['normalization_applied'] = True
            st.success('Weights normalized to 100%.')
        else:
            total = float(entered_decimal.sum())
            if not np.isclose(total, 1.0, atol=0.005):
                error = f'Weights sum to {total * 100:.1f}%. Normalize to 100% or revise the weights.'
                st.error(error)
                return WeightResult(weights=None, metadata=metadata, status='error', errors=[error])
            normalized_weights = entered_decimal.reindex(asset_cols).fillna(0.0).astype(float)
        metadata['weights_sum_after_normalization'] = float(normalized_weights.sum())
        preview = _make_weight_preview_table(asset_cols, entered_decimal, normalized_weights, 'Editable table')
        st.dataframe(preview, hide_index=True, use_container_width=True)
        validation_errors, validation_warnings = validate_portfolio_weights(entered_decimal, asset_cols)
        for warning in validation_warnings:
            st.warning(warning)
        if validation_errors:
            for error in validation_errors:
                st.error(error)
            return WeightResult(weights=None, metadata=metadata, status='error', warnings=validation_warnings, errors=validation_errors)
        metadata['negative_weight_detected'] = bool((entered_decimal < 0).any())
        metadata['zero_weight_funds'] = [asset for asset in asset_cols if float(normalized_weights.get(asset, 0.0)) == 0.0]
        return WeightResult(weights=normalized_weights.reindex(asset_cols).fillna(0.0).astype(float), metadata=metadata, status='complete', warnings=validation_warnings)

    weights_upload = st.file_uploader('Upload weights file', type=['csv', 'xlsx', 'xls'], key='workflow_weights_upload_file')
    if weights_upload is None:
        return WeightResult(weights=None, metadata=metadata, status='incomplete')

    sheet_name: str | None = None
    if weights_upload.name.lower().endswith(('.xlsx', '.xls')):
        workbook = pd.ExcelFile(BytesIO(weights_upload.getvalue()))
        sheet_name = st.selectbox('Weights sheet', options=workbook.sheet_names, index=0, key='workflow_weights_sheet_name')
    weights_df = _read_weights_upload(weights_upload, sheet_name=sheet_name)
    if weights_df.empty:
        error = 'Weights file is empty.'
        st.error(error)
        return WeightResult(weights=None, metadata=metadata, status='error', errors=[error])

    detected = detect_weight_columns(weights_df)
    column_options = weights_df.columns.tolist()
    default_fund_col = detected['fund_column'] if detected['fund_column'] in column_options else column_options[0]
    default_weight_col = detected['weight_column'] if detected['weight_column'] in column_options else column_options[min(1, len(column_options) - 1)]
    fund_col = st.selectbox('Fund column', options=column_options, index=column_options.index(default_fund_col), key='workflow_weights_fund_column')
    weight_col = st.selectbox('Weight column', options=column_options, index=column_options.index(default_weight_col), key='workflow_weights_value_column')

    prepared_table = prepare_weights_table(weights_df, fund_col, weight_col)
    prepared_table = prepared_table.dropna(subset=['Fund'])
    if prepared_table.empty:
        error = 'No usable fund names were found in the weights file.'
        st.error(error)
        return WeightResult(weights=None, metadata=metadata, status='error', errors=[error])

    if prepared_table['Fund'].duplicated().any():
        dupes = sorted(set(prepared_table.loc[prepared_table['Fund'].duplicated(), 'Fund'].tolist()))
        error = f"Duplicate fund names in weights file: {', '.join(dupes)}"
        st.error(error)
        return WeightResult(weights=None, metadata=metadata, status='error', errors=[error])

    inferred_as_percent = bool(prepared_table['Weight'].abs().max() > 1.5)
    treat_as_percent = st.checkbox('Treat file weights as percentages', value=inferred_as_percent, key='workflow_weights_treat_percent')
    normalize_to_100 = st.checkbox('Normalize weights to 100%', value=True, key='workflow_weights_normalize_upload')

    mapping_table = match_weight_names_to_assets(prepared_table['Fund'].tolist(), asset_cols)
    edited_mapping = st.data_editor(
        mapping_table,
        hide_index=True,
        use_container_width=True,
        disabled=['Uploaded Fund Name', 'Matched Return Column', 'Match Confidence'],
        key='workflow_weights_match_editor',
    )

    fuzzy_rows = edited_mapping[(edited_mapping['Match Confidence'] < 1.0) & (~edited_mapping['User Confirmed'].astype(bool))]
    if not fuzzy_rows.empty:
        error = 'Confirm fuzzy matches before applying uploaded weights.'
        st.error(error)
        return WeightResult(weights=None, metadata=metadata, status='error', errors=[error])

    confirmed_mapping = edited_mapping[['Uploaded Fund Name', 'Matched Return Column', 'User Confirmed']].copy()
    confirmed_mapping.loc[confirmed_mapping['Matched Return Column'].eq(''), 'Matched Return Column'] = confirmed_mapping['Uploaded Fund Name']
    mapping_lookup = dict(zip(confirmed_mapping['Uploaded Fund Name'], confirmed_mapping['Matched Return Column']))

    mapped_table = prepared_table.copy()
    mapped_table['Fund'] = mapped_table['Fund'].map(mapping_lookup).fillna(mapped_table['Fund'])
    mapped_table['Weight'] = pd.to_numeric(mapped_table['Weight'], errors='coerce').fillna(0.0)
    if treat_as_percent:
        mapped_table['Weight'] = mapped_table['Weight'] / 100.0

    raw_weight_series = build_asset_weight_series(asset_cols, mapped_table.rename(columns={'Fund': 'Fund', 'Weight': 'Weight'}), normalize=False)
    metadata['weights_sum_before_normalization'] = float(raw_weight_series.sum())

    validation_errors, validation_warnings = validate_portfolio_weights(raw_weight_series, asset_cols)
    for warning in validation_warnings:
        st.warning(warning)
    if validation_errors:
        for error in validation_errors:
            st.error(error)
        return WeightResult(weights=None, metadata=metadata, status='error', warnings=validation_warnings, errors=validation_errors)

    if normalize_to_100:
        normalized_weights = normalize_portfolio_weights(raw_weight_series)
        metadata['normalization_applied'] = True
        st.success('Weights normalized to 100%.')
    else:
        total = float(raw_weight_series.fillna(0.0).sum())
        if not np.isclose(total, 1.0, atol=0.005):
            error = f'Weights sum to {total * 100:.1f}%. Normalize to 100% or revise the weights.'
            st.error(error)
            return WeightResult(weights=None, metadata=metadata, status='error', errors=[error])
        normalized_weights = raw_weight_series.reindex(asset_cols).fillna(0.0).astype(float)

    normalized_weights = normalized_weights.reindex(asset_cols).fillna(0.0).astype(float)
    metadata['weights_sum_after_normalization'] = float(normalized_weights.sum())
    metadata['negative_weight_detected'] = bool((raw_weight_series < 0).any())
    metadata['missing_weight_funds'] = [asset for asset in asset_cols if asset not in raw_weight_series.index]
    metadata['extra_weight_rows'] = [name for name in raw_weight_series.index.unique() if name not in asset_cols]
    metadata['zero_weight_funds'] = [asset for asset in asset_cols if float(normalized_weights.get(asset, 0.0)) == 0.0]

    if metadata['missing_weight_funds']:
        st.warning(f"{len(metadata['missing_weight_funds'])} selected funds were missing from the weights file and were assigned 0% weight.")
    if metadata['extra_weight_rows']:
        st.warning(f"{len(metadata['extra_weight_rows'])} rows in the weights file do not match any selected fund return column.")

    preview = _make_weight_preview_table(asset_cols, raw_weight_series, normalized_weights, 'Uploaded weights')
    st.dataframe(preview, hide_index=True, use_container_width=True)
    return WeightResult(weights=normalized_weights, metadata=metadata, status='complete', warnings=validation_warnings)


def render_analysis_settings_step(settings: AnalysisSettings) -> None:
    _render_step_header(
        5,
        "Configure Analysis",
        "complete",
        "Review the analysis parameters before running the model.",
    )
    summary = pd.DataFrame(
        [
            {'setting': 'Report name', 'value': settings.report_name},
            {'setting': 'Portfolio value', 'value': f"{settings.portfolio_value:,.2f}"},
            {'setting': 'Risk-free rate', 'value': f"{settings.rf_rate:.2%}"},
            {'setting': 'Confidence level', 'value': f"{settings.confidence:.2%}"},
            {'setting': 'Number of simulations', 'value': f"{settings.num_sims:,}"},
            {'setting': 'Covariance method', 'value': settings.corr_method},
            {'setting': 'EWMA decay', 'value': f"{settings.ewma_decay:.2f}"},
            {'setting': 'Distribution type', 'value': settings.dist_type},
            {'setting': 'Show factor bucket analysis', 'value': 'Yes' if settings.show_factor_buckets else 'No'},
            {'setting': 'Exposure display mode', 'value': settings.display_mode},
            {'setting': 'Values are in percent', 'value': 'Yes' if settings.values_in_percent else 'No'},
            {'setting': 'Max missing data %', 'value': f"{settings.max_missing_pct:.1%}"},
        ]
    )
    st.dataframe(summary, hide_index=True, use_container_width=True)


def render_run_analysis_step(can_run: bool) -> bool:
    _render_step_header(
        6,
        "Run Analysis",
        "complete" if can_run else "incomplete",
        "Run the backend analysis after the earlier steps are complete.",
    )
    if not can_run:
        st.info("Complete the upload, mapping, validation, and weights steps first.")
        return False
    return st.button("Run Analysis", type="primary", key="workflow_run_analysis")


def render_guided_workflow(settings: AnalysisSettings) -> GuidedWorkflowState:
    st.markdown("## Guided Workflow")
    st.caption("Follow each step in order. Validation feedback appears before analysis is run.")

    upload_context = render_upload_step()

    with st.expander("Step 2 and 3: Map and Validate", expanded=True):
        mapping_result = render_mapping_step(upload_context, settings.values_in_percent, settings.max_missing_pct)
        render_validation_step(mapping_result)

    with st.expander("Step 4: Set Portfolio Weights", expanded=True):
        base_weights = mapping_result.bundle.asset_weight_input if mapping_result.bundle is not None else None
        weight_result = render_weights_step(mapping_result.bundle.asset_cols if mapping_result.bundle is not None else [], base_weights)

    with st.expander("Step 5: Configure Analysis", expanded=True):
        render_analysis_settings_step(settings)

    run_requested = render_run_analysis_step(mapping_result.bundle is not None and weight_result.weights is not None and not mapping_result.errors and not weight_result.errors)
    return GuidedWorkflowState(
        upload_context=upload_context,
        mapping_result=mapping_result,
        weight_result=weight_result,
        analysis_settings=settings,
        run_requested=run_requested,
    )
