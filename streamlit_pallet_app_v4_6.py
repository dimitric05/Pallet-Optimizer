
from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from reportlab.lib import colors as rl_colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

def apply_custom_css():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2.2rem !important;
        }

        h1 {
            font-size: 2.1rem !important;
            line-height: 1.25 !important;
            margin-top: 0.2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# Data model

@dataclass
class Job:
    product_family: str
    product_type: str
    depth_option: int
    package_depth: float
    width: float
    height: float
    qty: int
    side_down: str = 'Auto'


@dataclass
class ConfigurationItem:
    config_id: int
    label: str
    product_family: str
    product_type: str
    depth_option: int
    package_depth: float
    width: float
    height: float
    qty: int
    side_down: str = 'Auto'


@dataclass
class Pallet:
    pallet_id: str
    base_length: float
    base_width: float
    center_height: float
    center_depth: float
    max_depth_per_side: float
    max_height: float
    max_length: float
    usable_space_per_side: float
    pallet_cost: float = 0.0


@dataclass
class Placement:
    side: str
    row_number: int
    unit_number: int
    x: float
    y: float
    length: float
    depth: float
    orientation: str
    config_label: str
    config_id: int
    pallet_id: str = ''


@dataclass
class EvalResult:
    pallet_id: str
    feasible: bool
    rejection_reason: Optional[str]
    chosen_orientation: Optional[str]
    base_side: Optional[float]
    upright_height: Optional[float]
    package_depth: Optional[float]
    cols_per_row: Optional[int]
    max_rows_per_side: Optional[int]
    max_units_per_side: Optional[int]
    max_units_per_pallet: Optional[int]
    units_on_preview_pallet: Optional[int]
    units_top_side: Optional[int]
    units_bottom_side: Optional[int]
    pallets_needed: Optional[int]
    preview_utilization: Optional[float]
    capacity_utilization: Optional[float]
    ranking_reason: Optional[str]
    placements: Optional[List[Placement]]
    explanation: str
    pallet_cost_each: float = 0.0
    estimated_total_cost: float = 0.0


@dataclass(frozen=True)
class OrientationOption:
    orientation_name: str
    base_side: float
    upright_height: float
    effective_depth: float


@dataclass
class JobPalletLoad:
    pallet_number: int
    pallet_id: str
    placements: List[Placement]
    units_on_pallet: int
    units_top_side: int
    units_bottom_side: int
    preview_utilization: float
    explanation: str
    config_side_counts: Dict[int, Tuple[int, int]]
    balance_status: str
    balance_penalty: float
    pallet_cost_each: float = 0.0


@dataclass
class JobPlanResult:
    feasible: bool
    pallets_needed: Optional[int]
    total_units: int
    overall_utilization: Optional[float]
    avg_pallet_utilization: Optional[float]
    pallet_loads: List[JobPalletLoad]
    explanation: str
    total_balance_penalty: Optional[float]
    pallet_mix_summary: Dict[str, int]
    estimated_total_cost: float = 0.0
    excluded_configs: Optional[List[Tuple[str, str]]] = None  # (label, reason) for configs that fit no pallet


@dataclass
class SideState:
    name: str
    depth_left: float
    counts: Dict[int, int]
    rows: List[Tuple[int, OrientationOption, int]]

import os
import sys as _sys

def _resolve_data_dir() -> Path:

    override = os.environ.get('PALLET_OPTIMIZER_DATA_DIR', '').strip()
    if override:
        return Path(override)
    if getattr(_sys, 'frozen', False):
        base = os.environ.get('LOCALAPPDATA') or str(Path.home())
        return Path(base) / 'PalletOptimizer'
    return Path(__file__).resolve().parent


APP_DIR = _resolve_data_dir()
DEFAULT_CONFIG_PATH    = APP_DIR / 'pallet_config_seeded.json'
DEFAULT_DEPTH_CSV_PATH = APP_DIR / 'product_depths_extracted.csv'



DEFAULT_PALLET_CONFIG_JSON = '''{
  "pallets": [
    {
      "pallet_id": "60x46",
      "base_length": 60.0,
      "base_width": 46.0,
      "center_height": 48.0,
      "center_depth": 4.0,
      "max_depth_per_side": 21.0,
      "max_height": 84.0,
      "max_length": 58.0,
      "usable_space_per_side": 1218.0,
      "pallet_cost": 135.0
    },
    {
      "pallet_id": "72x46",
      "base_length": 72.0,
      "base_width": 46.0,
      "center_height": 48.0,
      "center_depth": 4.0,
      "max_depth_per_side": 21.0,
      "max_height": 84.0,
      "max_length": 70.0,
      "usable_space_per_side": 1470.0,
      "pallet_cost": 144.8
    },
    {
      "pallet_id": "96x46",
      "base_length": 96.0,
      "base_width": 46.0,
      "center_height": 48.0,
      "center_depth": 4.0,
      "max_depth_per_side": 21.0,
      "max_height": 84.0,
      "max_length": 94.0,
      "usable_space_per_side": 1974.0,
      "pallet_cost": 163.2
    },
    {
      "pallet_id": "108x46",
      "base_length": 108.0,
      "base_width": 46.0,
      "center_height": 48.0,
      "center_depth": 4.0,
      "max_depth_per_side": 21.0,
      "max_height": 84.0,
      "max_length": 106.0,
      "usable_space_per_side": 2226.0,
      "pallet_cost": 178.2
    },
    {
      "pallet_id": "120x46",
      "base_length": 120.0,
      "base_width": 46.0,
      "center_height": 48.0,
      "center_depth": 4.0,
      "max_depth_per_side": 21.0,
      "max_height": 84.0,
      "max_length": 118.0,
      "usable_space_per_side": 2478.0,
      "pallet_cost": 182.1
    }
  ],
  "global_rules": {
    "use_two_sides": true,
    "brace_height_ratio_required": 0.6667
  },
  "special_rules": {
    "force_long_side_down_product_types": []
  }
}
'''

DEFAULT_PRODUCT_DEPTHS_CSV = '''product_family,product_type,package_depth_1,package_depth_2
AA4325,PI,3.5,4.0
AA4325,SL,4.0,4.5
AA4325,DH,3.75,4.25
AA4325,CAS,5.0,5.5
AA4325,AW,4.5,5.0
AA4325,PW,3.25,3.75
BB5000,PI,3.5,4.0
BB5000,SL,4.25,4.75
BB5000,DH,4.0,4.5
BB5000,CAS,5.25,5.75
BB5000,AW,4.75,5.25
CC6100,PI,3.625,4.125
CC6100,SL,4.5,5.0
CC6100,DH,4.125,4.625
CC6100,CAS,5.5,6.0
CC6100,BAY,6.0,6.5
DD7200,PI,3.75,4.25
DD7200,SL,4.75,5.25
DD7200,DH,4.25,4.75
DD7200,CAS,5.75,6.25
DD7200,GL,4.0,4.5
EE8300,PI,4.0,4.5
EE8300,SL,5.0,5.5
EE8300,DH,4.5,5.0
EE8300,PW,3.5,4.0
'''


def ensure_data_files() -> None:                                                                                            #how bored are u if u r reading this
    """First-run setup: create the data directory and seed the default pallet
    config and product-depth table if they don't exist yet.  Never overwrites
    files the user has already created or edited.  Returns nothing; safe to
    call on every launch."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_CONFIG_PATH.exists():
        DEFAULT_CONFIG_PATH.write_text(DEFAULT_PALLET_CONFIG_JSON)
    if not DEFAULT_DEPTH_CSV_PATH.exists():
        DEFAULT_DEPTH_CSV_PATH.write_text(DEFAULT_PRODUCT_DEPTHS_CSV)


@st.cache_data
def load_config(config_path: str) -> dict:
    return json.loads(Path(config_path).read_text())


@st.cache_data
def load_depths(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)

class ProductDepthLookup:
    def __init__(self, df: pd.DataFrame):
        self.lookup: Dict[Tuple[str, str, int], Optional[float]] = {}
        for _, row in df.iterrows():
            fam = str(row['product_family']).strip().upper()
            typ = str(row['product_type']).strip().upper()
            d1 = None if pd.isna(row['package_depth_1']) else float(row['package_depth_1'])
            d2 = None if pd.isna(row['package_depth_2']) else float(row['package_depth_2'])
            self.lookup[(fam, typ, 1)] = d1
            self.lookup[(fam, typ, 2)] = d2 if d2 is not None else d1

    def get_default_depth(self, family: str, product_type: str, depth_option: int) -> Optional[float]:
        return self.lookup.get((family.strip().upper(), product_type.strip().upper(), int(depth_option)))


def pair_support_widths_are_nonincreasing(top_widths: List[float], bottom_widths: List[float]) -> bool:
    max_len = max(len(top_widths), len(bottom_widths))
    if max_len <= 1:
        return True
    totals: List[float] = []
    for i in range(max_len):
        totals.append((top_widths[i] if i < len(top_widths) else 0.0) + (bottom_widths[i] if i < len(bottom_widths) else 0.0))
    return all(totals[i] + 1e-9 >= totals[i + 1] for i in range(len(totals) - 1))


def side_widths_are_nonincreasing(widths: List[float]) -> bool:
    # widths[0] is closest to center; widths grow outward with index
    return all(widths[i] + 1e-9 >= widths[i+1] for i in range(len(widths) - 1))


def pair_support_widths_message(top_widths: List[float], bottom_widths: List[float]) -> str:
    max_len = max(len(top_widths), len(bottom_widths))
    totals = [round((top_widths[i] if i < len(top_widths) else 0.0) + (bottom_widths[i] if i < len(bottom_widths) else 0.0), 3) for i in range(max_len)]
    return f'Combined supported base widths from the center outward must be non-increasing. Current pair support widths = {totals} inches'


class BaseOptimizer:
    SIDE_OPTIONS = ['Auto', 'Short Side Down', 'Long Side Down']

    def __init__(self, config: dict, depth_df: pd.DataFrame, allowed_pallet_ids: Optional[List[str]] = None):
        self.config = config
        self.allowed_pallet_ids = None if not allowed_pallet_ids else set(allowed_pallet_ids)
        self.pallets: List[Pallet] = []
        self.pallets_missing_cost: List[str] = []
        for p in config['pallets']:
            if self.allowed_pallet_ids is not None and p.get('pallet_id') not in self.allowed_pallet_ids:
                continue
            known_fields = {f.name for f in fields(Pallet)}
            p = {k: v for k, v in p.items() if k in known_fields}
            # Pallet cost comes exclusively from the JSON config.
            # Missing or zero cost is tracked so the UI can warn the user.
            if 'pallet_cost' not in p or p['pallet_cost'] in (None,):
                p['pallet_cost'] = 0.0
            if float(p['pallet_cost'] or 0.0) <= 0.0:
                self.pallets_missing_cost.append(str(p.get('pallet_id', '?')))
            self.pallets.append(Pallet(**p))
        self.depth_lookup = ProductDepthLookup(depth_df)
        rules = config['global_rules']
        self.use_two_sides = bool(rules.get('use_two_sides', True))
        self.brace_height_ratio_required = float(rules.get('brace_height_ratio_required', 2/3))
        special = config.get('special_rules', {})
        self.force_long_side_down_types = {str(x).strip().upper() for x in special.get('force_long_side_down_product_types', [])}

    @staticmethod
    def normalize_dimensions(width: float, height: float) -> Tuple[float, float]:
        return max(width, height), min(width, height)

    def get_orientations(self, product_type: str, long_side: float, short_side: float, side_down: str = 'Auto') -> List[Tuple[str, float, float]]:
        side_down_norm = (side_down or 'Auto').strip().lower()
        ptype = product_type.strip().upper()
        if side_down_norm == 'short side down':
            return [('short_side_down', short_side, long_side)]
        if side_down_norm == 'long side down':
            return [('long_side_down', long_side, short_side)]
        if ptype in self.force_long_side_down_types:
            return [('long_side_down', long_side, short_side)]
        return [('short_side_down', short_side, long_side), ('long_side_down', long_side, short_side)]
                                                                                                                                    #how bored am i if i wrote this
    @staticmethod
    def split_units_across_sides(units_on_pallet: int, sides_used: int) -> Tuple[int, int]:
        if sides_used <= 1:
            return units_on_pallet, 0
        top = math.ceil(units_on_pallet / 2)
        bottom = units_on_pallet - top
        return top, bottom

    @staticmethod
    def build_row_pattern(units_for_side: int, cols_per_row: int, max_rows_per_side: int) -> Optional[List[int]]:
        if units_for_side < 0 or cols_per_row <= 0 or max_rows_per_side <= 0:
            return None
        if units_for_side > cols_per_row * max_rows_per_side:
            return None
        rows: List[int] = []
        remaining = units_for_side
        while remaining > 0:
            row_count = min(cols_per_row, remaining)
            rows.append(row_count)
            remaining -= row_count
        if len(rows) > max_rows_per_side:
            return None
        return rows

    @staticmethod
    def compute_row_positions(pallet_length: float, base_side: float, row_count: int) -> Tuple[List[float], float]:
        """
        Pack units flush (no gaps between them) and center the entire block
        on the pallet length.  The returned 'gap' is always 0.0 because units
        touch each other; the centering offset is absorbed into x_start.
        """
        if row_count <= 0:
            return [], 0.0
        total_width = row_count * base_side
        if total_width > pallet_length + 1e-9:
            return [], 0.0
        x_start = max((pallet_length - total_width) / 2.0, 0.0)
        positions = [x_start + i * base_side for i in range(row_count)]
        return positions, 0.0

    def pallet_by_id(self, pallet_id: str) -> Pallet:
        return next(p for p in self.pallets if p.pallet_id == pallet_id)

    def usable_area_for_pallet(self, pallet: Pallet) -> float:
        # usable_space_per_side is the area of ONE loading side. When both
        # sides are in use, placements span two sides, so utilization must
        # divide by the combined usable area or it can exceed 100%.
        sides_used = 2 if self.use_two_sides else 1
        return (pallet.usable_space_per_side or 0.0) * sides_used


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, index=False, sheet_name=(sheet_name[:31] or 'Sheet1'))
    output.seek(0)
    return output.getvalue()


def _fig_to_image_bytes(fig: go.Figure, width: int = 900, height: int = 500) -> bytes:
    """Render a Plotly figure to a PNG byte string using kaleido."""
    return pio.to_image(fig, format='png', width=width, height=height, scale=1.5)


def _df_to_rl_table(df: pd.DataFrame, col_widths=None) -> Table:
    """Convert a DataFrame to a ReportLab Table with basic styling."""
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle('cell', parent=styles['Normal'], fontSize=7, leading=9)
    header_style = ParagraphStyle('hdr', parent=styles['Normal'], fontSize=7, leading=9, fontName='Helvetica-Bold')

    header = [Paragraph(str(c), header_style) for c in df.columns]
    rows = [header]
    for _, row in df.iterrows():
        rows.append([Paragraph(str(v) if v is not None else '', cell_style) for v in row])

    tbl = Table(rows, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',  (0, 0), (-1, 0),  rl_colors.HexColor('#2c3e50')),
        ('TEXTCOLOR',   (0, 0), (-1, 0),  rl_colors.white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [rl_colors.HexColor('#f5f5f5'), rl_colors.white]),
        ('GRID',        (0, 0), (-1, -1), 0.35, rl_colors.HexColor('#cccccc')),
        ('TOPPADDING',  (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    return tbl


def _heading(text: str, level: int = 1) -> Paragraph:
    styles = getSampleStyleSheet()
    style = styles['Heading1'] if level == 1 else styles['Heading2']
    return Paragraph(text, style)


def export_pdf_by_configuration(
    job: Job,
    best: EvalResult,
    results: List[EvalResult],
    allowable_pallets: List[str],
    optimizer,
    job_name: str = '',
) -> bytes:
    """
    Build a PDF report for Single-Configuration mode.
    Includes a summary table, all-pallets comparison table, and a pallet
    layout image for every pallet in the best result.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter),
                            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    styles = getSampleStyleSheet()
    story = []

    # --- Title ---
    title_text = 'Pallet Loading Report — By Configuration'
    if job_name.strip():
        title_text = f'{job_name.strip()} — Pallet Loading Report (By Configuration)'
    story.append(Paragraph(title_text, styles['Title']))
    story.append(Spacer(1, 0.15 * inch))

    # --- Summary table ---
    story.append(_heading('Summary', level=2))
    summary_data = {
        'Field': ['Job Name', 'Allowable Pallets', 'Product Family', 'Product Type', 'Depth Option',
                  'Depth Used', 'Side Down', 'Width', 'Height', 'Quantity',
                  'Best Pallet', 'Chosen Orientation', 'Pallets Needed',
                  'Max Units / Pallet', 'Preview Utilization %',
                  'Capacity Utilization %', 'Pallet Cost Each', 'Estimated Total Cost'],
        'Value': [
            job_name.strip() or '-',
            ', '.join(allowable_pallets),
            job.product_family, job.product_type, job.depth_option,
            f'{job.package_depth:.3f}"' if job.package_depth else '-',
            job.side_down, f'{job.width:.3f}"', f'{job.height:.3f}"', str(job.qty),
            best.pallet_id, best.chosen_orientation, str(best.pallets_needed),
            str(best.max_units_per_pallet),
            f'{round((best.preview_utilization or 0.0) * 100, 2)}%',
            f'{round((best.capacity_utilization or 0.0) * 100, 2)}%',
            f'${best.pallet_cost_each:,.2f}',
            f'${best.estimated_total_cost:,.2f}',
        ],
    }
    summary_df = pd.DataFrame(summary_data)
    story.append(_df_to_rl_table(summary_df, col_widths=[2.2 * inch, 4.5 * inch]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f'<b>Explanation:</b> {best.explanation}', styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # --- All pallets comparison ---
    story.append(_heading('All Pallets Evaluated', level=2))
    all_rows = []
    for r in results:
        all_rows.append({
            'Pallet': r.pallet_id,
            'Feasible': str(r.feasible),
            'Orientation': r.chosen_orientation or '-',
            'Pallets Needed': r.pallets_needed,
            'Units / Pallet': r.max_units_per_pallet,
            'Cost Each': f'${r.pallet_cost_each:,.2f}' if r.pallet_cost_each else '-',
            'Total Cost': f'${r.estimated_total_cost:,.2f}' if r.estimated_total_cost else '-',
            'Preview Util %': f'{round((r.preview_utilization or 0.0) * 100, 2)}%' if r.preview_utilization is not None else '-',
            'Reason / Explanation': (r.rejection_reason if not r.feasible else r.explanation) or '-',
        })
    story.append(_df_to_rl_table(pd.DataFrame(all_rows)))
    story.append(PageBreak())

    # --- Pallet layout images ---
    chosen_pallet = optimizer.pallet_by_id(best.pallet_id)
    pallet_count = best.pallets_needed or 1
    story.append(_heading('Pallet Layout Previews', level=1))
    story.append(Spacer(1, 0.1 * inch))
    for pallet_num in range(1, pallet_count + 1):
        units_on = optimizer.units_for_pallet_sequence(job.qty, best.max_units_per_pallet or 1, pallet_num)
        preview = optimizer.build_preview_for_units(chosen_pallet, best, units_on, pallet_num)
        img_bytes = _pallet_png_bytes(
            chosen_pallet,
            preview.placements or [],
            f'Pallet #{pallet_num} — {chosen_pallet.pallet_id}  ({units_on} units)',
        )
        img_buf = io.BytesIO(img_bytes)
        rl_img = RLImage(img_buf, width=9 * inch, height=5 * inch)
        story.append(rl_img)
        if pallet_num < pallet_count:
            story.append(PageBreak())

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def export_pdf_by_job(
    configs: List[ConfigurationItem],
    result: JobPlanResult,
    allowable_pallets: List[str],
    optimizer,
    job_name: str = '',
) -> bytes:
    """
    Build a PDF report for Mixed-Job mode.
    Includes summary, config list, pallet mix table, per-pallet details,
    and a layout image for every pallet in the job plan.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(letter),
                            leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    styles = getSampleStyleSheet()
    story = []

    # --- Title ---
    title_text = 'Pallet Loading Report — By Job'
    if job_name.strip():
        title_text = f'{job_name.strip()} — Pallet Loading Report (By Job)'
    story.append(Paragraph(title_text, styles['Title']))
    story.append(Spacer(1, 0.15 * inch))

    # --- Job summary ---
    story.append(_heading('Job Summary', level=2))
    summary_data = {
        'Field': ['Job Name', 'Allowable Pallets', 'Total Units', 'Pallets Needed',
                  'Overall Utilization %', 'Avg Pallet Utilization %',
                  'Estimated Total Cost', 'Pallet Types Used', 'Balance Warning Total'],
        'Value': [
            job_name.strip() or '-',
            ', '.join(allowable_pallets),
            str(result.total_units), str(result.pallets_needed),
            f'{round((result.overall_utilization or 0.0) * 100, 2)}%',
            f'{round((result.avg_pallet_utilization or 0.0) * 100, 2)}%',
            f'${result.estimated_total_cost:,.2f}',
            str(len(result.pallet_mix_summary)),
            f'{(result.total_balance_penalty or 0.0):.1f}',
        ],
    }
    story.append(_df_to_rl_table(pd.DataFrame(summary_data), col_widths=[2.5 * inch, 4.0 * inch]))
    story.append(Spacer(1, 0.1 * inch))
    story.append(Paragraph(f'<b>Explanation:</b> {result.explanation}', styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # --- Configurations ---
    story.append(_heading('Configurations', level=2))
    configs_df = pd.DataFrame([{
        'Label': c.label, 'Family': c.product_family, 'Type': c.product_type,
        'Depth Option': c.depth_option,
        'Depth Used': f'{c.package_depth:.3f}"' if c.package_depth else '-',
        'Side Down': c.side_down,
        'Width': f'{c.width:.3f}"', 'Height': f'{c.height:.3f}"', 'Qty': c.qty,
    } for c in configs])
    story.append(_df_to_rl_table(configs_df))
    story.append(Spacer(1, 0.2 * inch))

    # --- Pallet mix ---
    story.append(_heading('Pallet Mix', level=2))
    mix_rows = []
    for pid, cnt in sorted(result.pallet_mix_summary.items()):
        cost_each = next((pl.pallet_cost_each for pl in result.pallet_loads if pl.pallet_id == pid), 0.0)
        mix_rows.append({'Pallet Type': pid, 'Count': cnt,
                         'Cost Each': f'${cost_each:,.2f}',
                         'Total Cost': f'${cnt * cost_each:,.2f}'})
    story.append(_df_to_rl_table(pd.DataFrame(mix_rows)))
    story.append(Spacer(1, 0.2 * inch))

    # --- Per-pallet detail ---
    story.append(_heading('Per-Pallet Detail', level=2))
    per_pallet_df = pd.DataFrame([{
        'Pallet #': load.pallet_number, 'Pallet Type': load.pallet_id,
        'Units': load.units_on_pallet,
        'Top Units': load.units_top_side, 'Bottom Units': load.units_bottom_side,
        'Utilization %': f'{round(load.preview_utilization * 100, 2)}%',
        'Balance': load.balance_status,
        'Cost Each': f'${load.pallet_cost_each:,.2f}',
    } for load in result.pallet_loads])
    story.append(_df_to_rl_table(per_pallet_df))
    story.append(PageBreak())

    # --- Pallet layout images ---
    story.append(_heading('Pallet Layout Previews', level=1))
    story.append(Spacer(1, 0.1 * inch))
    for load in result.pallet_loads:
        chosen_pallet = optimizer.pallet_by_id(load.pallet_id)
        img_bytes = _pallet_png_bytes(
            chosen_pallet,
            load.placements,
            f'Job Pallet #{load.pallet_number} — {chosen_pallet.pallet_id}  ({load.units_on_pallet} units)',
        )
        img_buf = io.BytesIO(img_bytes)
        rl_img = RLImage(img_buf, width=9 * inch, height=5 * inch)
        story.append(rl_img)
        if load.pallet_number < len(result.pallet_loads):
            story.append(PageBreak())

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()


def export_by_configuration(job: Job, best: EvalResult, results: List[EvalResult], allowable_pallets: List[str]) -> bytes:
    summary = pd.DataFrame([{
        'Mode': 'By Configuration',
        'Allowable Pallets': ', '.join(allowable_pallets),
        'Product Family': job.product_family,
        'Product Type': job.product_type,
        'Depth Option': job.depth_option,
        'Depth Used': job.package_depth,
        'Side Down': job.side_down,
        'Width': job.width,
        'Height': job.height,
        'Quantity': job.qty,
        'Best Pallet': best.pallet_id,
        'Chosen Orientation': best.chosen_orientation,
        'Pallets Needed': best.pallets_needed,
        'Max Units/Pallet': best.max_units_per_pallet,
        'Preview Utilization %': round((best.preview_utilization or 0.0) * 100, 2),
        'Capacity Utilization %': round((best.capacity_utilization or 0.0) * 100, 2),
        'Pallet Cost Each': best.pallet_cost_each,
        'Estimated Total Cost': best.estimated_total_cost,
        'Explanation': best.explanation,
    }])
    all_rows = []
    for r in results:
        all_rows.append({
            'Pallet': r.pallet_id,
            'Feasible': r.feasible,
            'Orientation': r.chosen_orientation,
            'Pallets Needed': r.pallets_needed,
            'Max Units/Pallet': r.max_units_per_pallet,
            'Pallet Cost Each': r.pallet_cost_each,
            'Estimated Total Cost': r.estimated_total_cost,
            'Preview Utilization %': None if r.preview_utilization is None else round(r.preview_utilization * 100, 2),
            'Capacity Utilization %': None if r.capacity_utilization is None else round(r.capacity_utilization * 100, 2),
            'Reason / Explanation': r.rejection_reason if not r.feasible else r.explanation,
        })
    return to_excel_bytes({'Summary': summary, 'All Pallets': pd.DataFrame(all_rows)})


def export_by_job(configs: List[ConfigurationItem], result: JobPlanResult, allowable_pallets: List[str]) -> bytes:
    configs_df = pd.DataFrame([{
        'Label': c.label,
        'Family': c.product_family,
        'Type': c.product_type,
        'Depth Option': c.depth_option,
        'Depth Used': c.package_depth,
        'Side Down': c.side_down,
        'Width': c.width,
        'Height': c.height,
        'Qty': c.qty,
    } for c in configs])
    summary = pd.DataFrame([{
        'Mode': 'By Job',
        'Allowable Pallets': ', '.join(allowable_pallets),
        'Total Units': result.total_units,
        'Pallets Needed': result.pallets_needed,
        'Overall Utilization %': round((result.overall_utilization or 0.0) * 100, 2),
        'Avg Pallet Utilization %': round((result.avg_pallet_utilization or 0.0) * 100, 2),
        'Estimated Total Cost': result.estimated_total_cost,
        'Pallet Types Used': len(result.pallet_mix_summary),
        'Balance Warning Total': result.total_balance_penalty,
        'Explanation': result.explanation,
    }])
    mix_df = pd.DataFrame([{
        'Pallet Type': pid,
        'Count': cnt,
        'Cost Each': next((pl.pallet_cost_each for pl in result.pallet_loads if pl.pallet_id == pid), 0.0),
    } for pid, cnt in sorted(result.pallet_mix_summary.items())])
    mix_df['Total Cost'] = mix_df['Count'] * mix_df['Cost Each']
    per_pallet = pd.DataFrame([{
        'Pallet #': load.pallet_number,
        'Pallet Type': load.pallet_id,
        'Units on Pallet': load.units_on_pallet,
        'Top Side Units': load.units_top_side,
        'Bottom Side Units': load.units_bottom_side,
        'Selected Pallet Utilization %': round(load.preview_utilization * 100, 2),
        'Balance Status': load.balance_status,
        'Balance Warning Total': load.balance_penalty,
        'Pallet Cost Each': load.pallet_cost_each,
        'Explanation': load.explanation,
    } for load in result.pallet_loads])
    side_rows = []
    label_lookup = {c.config_id: c.label for c in configs}
    for load in result.pallet_loads:
        for cid, (top_count, bottom_count) in load.config_side_counts.items():
            diff = abs(top_count - bottom_count)
            status = 'Balanced' if diff <= 1 else ('Slight imbalance' if diff <= 2 else 'High risk')
            side_rows.append({'Pallet #': load.pallet_number, 'Pallet Type': load.pallet_id, 'Configuration': label_lookup.get(cid, f'Config {cid}'), 'Top Side': top_count, 'Bottom Side': bottom_count, 'Difference': diff, 'Status': status})
    return to_excel_bytes({'Summary': summary, 'Configurations': configs_df, 'Pallet Mix': mix_df, 'Per Pallet': per_pallet, 'Side Split': pd.DataFrame(side_rows)})

class SingleConfigOptimizer(BaseOptimizer):
    def build_preview_placements(self, pallet: Pallet, orientation_name: str, base_side: float, ship_depth: float,
                                 top_rows: List[int], bottom_rows: List[int], config_label: str = 'Config 1', config_id: int = 1) -> List[Placement]:
        placements: List[Placement] = []
        unit_number = 1
        # Top side: cursor starts at the center beam top edge and moves outward
        # (increasing Y) by each row's depth, so rows never overlap regardless
        # of whether all rows share the same depth.
        top_y_cursor = pallet.max_depth_per_side + pallet.center_depth
        for row_index, row_count in enumerate(top_rows, start=1):
            row_y = top_y_cursor
            top_y_cursor += ship_depth
            x_positions, _ = self.compute_row_positions(pallet.max_length, base_side, row_count)
            for x in x_positions:
                placements.append(Placement('top', row_index, unit_number, x, row_y, base_side, ship_depth, orientation_name, config_label, config_id, pallet.pallet_id))
                unit_number += 1
        # Bottom side: cursor starts at the center beam bottom edge and moves
        # outward (decreasing Y) by each row's depth.
        bottom_y_cursor = pallet.max_depth_per_side
        for row_index, row_count in enumerate(bottom_rows, start=1):                        
            bottom_y_cursor -= ship_depth
            row_y = bottom_y_cursor
            x_positions, _ = self.compute_row_positions(pallet.max_length, base_side, row_count)
            for x in x_positions:
                placements.append(Placement('bottom', row_index, unit_number, x, row_y, base_side, ship_depth, orientation_name, config_label, config_id, pallet.pallet_id))
                unit_number += 1
        return placements

    def build_preview_for_units(self, pallet: Pallet, result: EvalResult, units_on_this_pallet: int, pallet_number: int) -> EvalResult:
        sides_used = 2 if self.use_two_sides else 1
        units_top_side, units_bottom_side = self.split_units_across_sides(units_on_this_pallet, sides_used)
        top_rows = self.build_row_pattern(units_top_side, result.cols_per_row or 0, result.max_rows_per_side or 0)
        bottom_rows = self.build_row_pattern(units_bottom_side, result.cols_per_row or 0, result.max_rows_per_side or 0)
        placements = self.build_preview_placements(pallet, result.chosen_orientation or 'unknown', result.base_side or 0.0, result.package_depth or 0.0, top_rows or [], bottom_rows or [])
        total_usable_area = self.usable_area_for_pallet(pallet)
        preview_window_area = units_on_this_pallet * (result.base_side or 0.0) * (result.package_depth or 0.0)
        preview_util = (preview_window_area / total_usable_area) if total_usable_area else 0.0
        return replace(result, units_on_preview_pallet=units_on_this_pallet, units_top_side=units_top_side, units_bottom_side=units_bottom_side, preview_utilization=preview_util, placements=placements, explanation=f'{result.explanation} This viewport currently shows pallet #{pallet_number} with {units_on_this_pallet} units.')

    def units_for_pallet_sequence(self, total_qty: int, max_units_per_pallet: int, pallet_number: int) -> int:
        if max_units_per_pallet <= 0:
            return 0
        pallets_needed = math.ceil(total_qty / max_units_per_pallet)
        pallet_number = max(1, min(pallet_number, pallets_needed))
        if pallet_number < pallets_needed:
            return max_units_per_pallet
        remainder = total_qty % max_units_per_pallet
        return remainder if remainder != 0 else max_units_per_pallet

    def evaluate_job(self, job: Job) -> Tuple[Optional[EvalResult], List[EvalResult], Optional[float]]:
        if not self.pallets:
            return None, [], job.package_depth
        ship_depth = job.package_depth
        long_side, short_side = self.normalize_dimensions(job.width, job.height)
        results = [self.evaluate_against_pallet(job, pallet, long_side, short_side, ship_depth) for pallet in self.pallets]
        feasible = [r for r in results if r.feasible]
        if not feasible:
            return None, results, ship_depth
        best = sorted(feasible, key=lambda r: (r.pallets_needed if r.pallets_needed is not None else 10**9, r.estimated_total_cost, -(r.capacity_utilization or 0.0), -(r.preview_utilization or 0.0)))[0]
        return best, results, ship_depth

    def evaluate_against_pallet(self, job: Job, pallet: Pallet, long_side: float, short_side: float, ship_depth: float) -> EvalResult:
        reasons: List[str] = []
        valid_results: List[EvalResult] = []
        sides_used = 2 if self.use_two_sides else 1
        if ship_depth > pallet.max_depth_per_side:
            return EvalResult(pallet.pallet_id, False, f'Input depth {ship_depth:.3f} exceeds max depth per side {pallet.max_depth_per_side:.3f}.', None, None, None, ship_depth, None, None, None, None, None, None, None, None, None, None, None, None, f'Pallet {pallet.pallet_id} rejected because input depth is too large for one pallet side.', pallet.pallet_cost, 0.0)
        for orientation_name, base_side, upright_height in self.get_orientations(job.product_type, long_side, short_side, job.side_down):
            if base_side > pallet.max_length:
                reasons.append(f'{orientation_name}: base side {base_side:.3f} exceeds max length {pallet.max_length:.3f}')
                continue
            if upright_height > pallet.max_height:
                reasons.append(f'{orientation_name}: upright height {upright_height:.3f} exceeds max height {pallet.max_height:.3f}')
                continue
            brace_ratio_actual = pallet.center_height / upright_height if upright_height > 0 else 0.0
            if brace_ratio_actual <= self.brace_height_ratio_required:
                reasons.append(f'{orientation_name}: brace rule failed ({brace_ratio_actual:.3f} <= {self.brace_height_ratio_required:.3f})')
                continue
            cols_per_row = math.floor(pallet.max_length / base_side)
            max_rows_per_side = math.floor(pallet.max_depth_per_side / ship_depth)
            if cols_per_row <= 0 or max_rows_per_side <= 0:
                reasons.append(f'{orientation_name}: no valid row pattern')
                continue
            max_units_per_side = cols_per_row * max_rows_per_side
            max_units_per_pallet = max_units_per_side * sides_used
            if max_units_per_pallet <= 0:
                reasons.append(f'{orientation_name}: max pallet capacity is zero')
                continue
            pallets_needed = math.ceil(job.qty / max_units_per_pallet)
            units_on_preview_pallet = min(job.qty, max_units_per_pallet)
            units_top_side, units_bottom_side = self.split_units_across_sides(units_on_preview_pallet, sides_used)
            top_rows = self.build_row_pattern(units_top_side, cols_per_row, max_rows_per_side)
            bottom_rows = self.build_row_pattern(units_bottom_side, cols_per_row, max_rows_per_side)
            if top_rows is None or bottom_rows is None:
                reasons.append(f'{orientation_name}: could not build a valid row pattern for the preview pallet')
                continue
            top_widths = [row_count * base_side for row_count in top_rows]
            bottom_widths = [row_count * base_side for row_count in bottom_rows]
            if not pair_support_widths_are_nonincreasing(top_widths, bottom_widths):
                reasons.append(f'{orientation_name}: {pair_support_widths_message(top_widths, bottom_widths)}')
                continue
            placements = self.build_preview_placements(pallet, orientation_name, base_side, ship_depth, top_rows, bottom_rows)
            total_usable_area = self.usable_area_for_pallet(pallet)
            preview_window_area = units_on_preview_pallet * base_side * ship_depth
            capacity_window_area = max_units_per_pallet * base_side * ship_depth
            preview_utilization = (preview_window_area / total_usable_area) if total_usable_area else 0.0
            capacity_utilization = (capacity_window_area / total_usable_area) if total_usable_area else 0.0
            total_cost = pallet.pallet_cost * pallets_needed
            ranking_reason = 'Ranked by: lowest pallets needed, then lowest estimated cost, then highest capacity utilization, then highest preview utilization.'
            explanation = f'Pallet {pallet.pallet_id} selected with orientation {orientation_name}. Base side on pallet = {base_side:.3f}\", upright height = {upright_height:.3f}\". Depth used = {ship_depth:.3f}\". Max columns per row = {cols_per_row}, max rows per side = {max_rows_per_side}. Max units / side = {max_units_per_side}, max units / pallet = {max_units_per_pallet}. Each side independently satisfies the non-increasing taper rule (row widths do not increase as rows move outward from the center). Preview utilization = {preview_utilization:.2%}. Capacity utilization = {capacity_utilization:.2%}. Pallets needed = {pallets_needed}. Estimated total pallet cost = ${total_cost:,.2f}.'
            valid_results.append(EvalResult(pallet.pallet_id, True, None, orientation_name, base_side, upright_height, ship_depth, cols_per_row, max_rows_per_side, max_units_per_side, max_units_per_pallet, units_on_preview_pallet, units_top_side, units_bottom_side, pallets_needed, preview_utilization, capacity_utilization, ranking_reason, placements, explanation, pallet.pallet_cost, total_cost))
        if not valid_results:
            reason_text = ' | '.join(reasons) if reasons else 'No valid orientation found.'
            return EvalResult(pallet.pallet_id, False, reason_text, None, None, None, ship_depth, None, None, None, None, None, None, None, None, None, None, None, None, f'Pallet {pallet.pallet_id} rejected. {reason_text}', pallet.pallet_cost, 0.0)
        return sorted(valid_results, key=lambda r: (r.pallets_needed if r.pallets_needed is not None else 10**9, r.estimated_total_cost, -(r.capacity_utilization or 0.0), -(r.preview_utilization or 0.0)))[0]


class MixedJobOptimizer(BaseOptimizer):
    def all_feasible_options_by_pallet(self, configs: List[ConfigurationItem]) -> Dict[str, Dict[int, List[OrientationOption]]]:
        out: Dict[str, Dict[int, List[OrientationOption]]] = {}
        for pallet in self.pallets:
            per_cfg: Dict[int, List[OrientationOption]] = {}
            for c in configs:
                if c.package_depth > pallet.max_depth_per_side:
                    per_cfg[c.config_id] = []
                    continue
                long_side, short_side = self.normalize_dimensions(c.width, c.height)
                opts: List[OrientationOption] = []
                for orientation_name, base_side, upright_height in self.get_orientations(c.product_type, long_side, short_side, c.side_down):
                    if base_side > pallet.max_length:
                        continue
                    if upright_height > pallet.max_height:
                        continue
                    brace_ratio_actual = pallet.center_height / upright_height if upright_height > 0 else 0.0
                    if brace_ratio_actual <= self.brace_height_ratio_required:
                        continue
                    opts.append(OrientationOption(orientation_name, base_side, upright_height, c.package_depth))
                opts.sort(key=lambda o: (-o.base_side, o.effective_depth))
                per_cfg[c.config_id] = opts
            out[pallet.pallet_id] = per_cfg
        return out

    def config_difficulty_scores(self, configs: List[ConfigurationItem], all_feasible: Dict[str, Dict[int, List[OrientationOption]]]) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for c in configs:
            feasible_pallet_count = sum(1 for pid in all_feasible if all_feasible[pid].get(c.config_id))
            footprint = max(c.width, c.height) * min(c.width, c.height)
            scores[c.config_id] = 1e9 if feasible_pallet_count <= 0 else (1000.0 / feasible_pallet_count) + footprint
        return scores

    def build_balance_warning(self, config_side_counts: Dict[int, Tuple[int, int]]) -> Tuple[str, float]:
        diffs = [abs(t - b) for t, b in config_side_counts.values()]
        max_diff = max(diffs) if diffs else 0
        total_diff = float(sum(diffs))
        if max_diff <= 1:
            return 'Balanced', total_diff
        if max_diff <= 2:
            return 'Slight imbalance', total_diff
        return 'High imbalance risk', total_diff

    @staticmethod
    def _row_widths_from_side_rows(rows: List[Tuple[int, OrientationOption, int]]) -> List[float]:
        return [opt.base_side * row_count for _, opt, row_count in rows]

    def _can_add_row_without_rule_break(self, top_rows: List[Tuple[int, OrientationOption, int]], bottom_rows: List[Tuple[int, OrientationOption, int]], side_name: str, opt: OrientationOption, row_count: int) -> bool:
        top_widths = self._row_widths_from_side_rows(top_rows)
        bottom_widths = self._row_widths_from_side_rows(bottom_rows)
        proposed_width = opt.base_side * row_count
        if side_name == 'top':
            top_widths = top_widths + [proposed_width]
        else:
            bottom_widths = bottom_widths + [proposed_width]        
        return (
            side_widths_are_nonincreasing(top_widths) and
            side_widths_are_nonincreasing(bottom_widths)
        )


    def candidate_rows_for_side(self, side, other_side, remaining, options_by_config, pallet):
        candidates = []
    
        # last row width on this side (closest-to-center is row 1; outward increases)
        if side.rows:
            last_width = self._row_widths_from_side_rows(side.rows)[-1]
        else:
            last_width = pallet.max_length
    
        def any_source_has_two(opt):
            # any config with >=2 remaining that matches this geometry
            for cid2, q2 in remaining.items():
                if q2 >= 2:
                    for opt2 in options_by_config.get(cid2, []):
                        if (abs(opt2.base_side - opt.base_side) <= 1e-9 and
                            abs(opt2.effective_depth - opt.effective_depth) <= 1e-9 and
                            opt2.orientation_name == opt.orientation_name):
                            return True
            return False
    
        for cid, qty_left in remaining.items():
            if qty_left <= 0:
                continue
            for opt in options_by_config.get(cid, []):
                # Do not pre-filter by prev_base here.  The authoritative taper
                # check (_can_add_row_without_rule_break) re-validates the full
                # accumulated row list and is the single source of truth.
                # A prev_base fast-gate can disagree with it when mixed-depth
                # configs produce rows whose widths pass the full check but were
                # being incorrectly blocked by the stale scalar.
                if opt.effective_depth > side.depth_left + 1e-9:
                    continue
    
                max_fit_len = math.floor(pallet.max_length / opt.base_side)
                if max_fit_len <= 0:
                    continue
    
                max_fit_taper = math.floor(last_width / opt.base_side) if side.rows else max_fit_len
    
                # this is the TRUE row-slot capacity
                row_slot_max = min(max_fit_len, max_fit_taper)
                if row_slot_max <= 0:
                    continue
    
                # clamp to remaining qty for this cid
                max_count = min(row_slot_max, qty_left)
    
                # --- Strong skinny-row guard:
                # If the slot can take 2 and SOMEONE can supply 2, forbid 1-wide rows here.
                min_count = 1
                if side.rows and row_slot_max >= 2 and any_source_has_two(opt):
                    min_count = 2
    
                if max_count < min_count:
                    continue
    
                for row_count in range(max_count, min_count - 1, -1):
                    if self._can_add_row_without_rule_break(
                        top_rows=side.rows if side.name == 'top' else other_side.rows,
                        bottom_rows=other_side.rows if side.name == 'top' else side.rows,
                        side_name=side.name,
                        opt=opt,
                        row_count=row_count,
                    ):
                        candidates.append((cid, opt, row_count))
    
        return candidates

    def build_single_pallet_candidate(self, pallet: Pallet, remaining: Dict[int, int], config_map: Dict[int, ConfigurationItem], options_by_config: Dict[int, List[OrientationOption]], difficulty: Dict[int, float]) -> Optional[JobPalletLoad]:
        feasible_seed_ids = [cid for cid, qty in remaining.items() if qty > 0 and options_by_config.get(cid)]
        if not feasible_seed_ids:
            return None
        seed_id = sorted(
            feasible_seed_ids,
            key=lambda cid: (
                # Skip seeds that can only contribute 1 unit total to this pallet —
                # a 1-unit seed anchors a side badly and forces singleton rows.
                # Prefer seeds with at least 2 units remaining.
                0 if remaining[cid] >= 2 else 1,
                -difficulty.get(cid, 0.0),
                -remaining[cid],
                -max((o.base_side for o in options_by_config[cid]), default=0.0)
            )
        )[0]
        top = SideState('top', pallet.max_depth_per_side, {cid: 0 for cid in remaining}, [])
        bottom = SideState('bottom', pallet.max_depth_per_side, {cid: 0 for cid in remaining}, [])
        local_remaining = dict(remaining)

        def choose_best_candidate(side: SideState, other_side: SideState) -> Optional[Tuple[int, OrientationOption, int, float]]:
            cands = self.candidate_rows_for_side(side, other_side, local_remaining, options_by_config, pallet)
            if not cands:
                return None
            best = None
            best_score = None
            for cid, opt, row_count in cands:
                # Clamp to actual remaining supply before scoring so that the
                # imbalance penalty reflects the real placement, not an
                # optimistic unclamped count that gets reduced at commit time.
                effective_count = min(row_count, local_remaining.get(cid, 0))
                if effective_count <= 0:
                    continue
                area = effective_count * opt.base_side * opt.effective_depth
                units_side = sum(side.counts.values())
                units_other = sum(other_side.counts.values())
                imbalance_after = abs((units_side + effective_count) - units_other)
                score = area + (effective_count * 1000.0) + (difficulty.get(cid, 0.0) * 2.0)
                score += opt.base_side * 500.0
                if cid == seed_id:
                    score += 250.0 * effective_count
                score -= 1200.0 * imbalance_after
                if best_score is None or score > best_score:
                    best_score = score
                    best = (cid, opt, effective_count, score)
            return best

        while True:
            top_best = choose_best_candidate(top, bottom)
            bottom_best = choose_best_candidate(bottom, top)
            if top_best is None and bottom_best is None:
                break
            if top_best is not None and bottom_best is not None:
                units_top    = sum(top.counts.values())
                units_bottom = sum(bottom.counts.values())
                diff_if_top    = abs((units_top    + top_best[2])    - units_bottom)
                diff_if_bottom = abs( units_top    - (units_bottom   + bottom_best[2]))
                if diff_if_top < diff_if_bottom:
                    chosen_side = 'top'
                elif diff_if_bottom < diff_if_top:
                    chosen_side = 'bottom'
                else:
                    # Imbalance outcome is equal — prefer whichever side currently
                    # has fewer units (the lagging side).  This enforces genuine
                    # alternation and prevents bottom from accumulating rows while
                    # top sits empty just because bottom scores marginally higher.
                    # Only fall back to raw score when both sides are exactly even.
                    if units_top < units_bottom:
                        chosen_side = 'top'
                    elif units_bottom < units_top:
                        chosen_side = 'bottom'
                    else:
                        chosen_side = 'top' if top_best[3] >= bottom_best[3] else 'bottom'
            elif top_best is not None:
                chosen_side = 'top'
            else:
                chosen_side = 'bottom'
            chosen = top_best if chosen_side == 'top' else bottom_best
            cid, opt, row_count, _ = chosen
            row_count = min(row_count, local_remaining[cid])
            if row_count <= 0:
                break
            if chosen_side == 'top':
                top.rows.append((cid, opt, row_count))
                top.depth_left  -= opt.effective_depth
                top.counts[cid] += row_count
            else:
                bottom.rows.append((cid, opt, row_count))
                bottom.depth_left  -= opt.effective_depth
                bottom.counts[cid] += row_count
            local_remaining[cid] -= row_count
        # Per-side taper rule is enforced during row addition; no extra combined (top+bottom) pyramid check here.

        units_on_pallet = sum(top.counts.values()) + sum(bottom.counts.values())
        if units_on_pallet <= 0:
            return None
        placements: List[Placement] = []
        unit_number = 1
        # Top side: cursor starts at the center beam top edge and moves outward
        # (increasing Y). Each row advances by its own effective_depth so that
        # mixed-depth configs never overlap.
        top_y_cursor = pallet.max_depth_per_side + pallet.center_depth
        for row_index, (cid, opt, row_count) in enumerate(top.rows, start=1):
            row_y = top_y_cursor
            top_y_cursor += opt.effective_depth
            x_positions, _ = self.compute_row_positions(pallet.max_length, opt.base_side, row_count)
            label = config_map[cid].label or f'Config {cid}'
            for x in x_positions:
                placements.append(Placement('top', row_index, unit_number, x, row_y, opt.base_side, opt.effective_depth, opt.orientation_name, label, cid, pallet.pallet_id))
                unit_number += 1
        # Accumulate each row's depth outward from the center beam so that
        # mixed-depth configurations (different effective_depth per row) are
        # rendered at the correct Y coordinates in the Plotly preview.
        bottom_y_cursor = pallet.max_depth_per_side
        for row_index, (cid, opt, row_count) in enumerate(bottom.rows, start=1):
            bottom_y_cursor -= opt.effective_depth
            row_y = bottom_y_cursor
            x_positions, _ = self.compute_row_positions(pallet.max_length, opt.base_side, row_count)
            label = config_map[cid].label or f'Config {cid}'
            for x in x_positions:
                placements.append(Placement('bottom', row_index, unit_number, x, row_y, opt.base_side, opt.effective_depth, opt.orientation_name, label, cid, pallet.pallet_id))
                unit_number += 1
        used_area = sum(p.length * p.depth for p in placements)
        total_usable_area = self.usable_area_for_pallet(pallet)
        preview_utilization = (used_area / total_usable_area) if total_usable_area else 0.0
        config_side_counts = {cid: (top.counts.get(cid, 0), bottom.counts.get(cid, 0)) for cid in config_map if top.counts.get(cid, 0) or bottom.counts.get(cid, 0)}
        balance_status, balance_penalty = self.build_balance_warning(config_side_counts)
        return JobPalletLoad(0, pallet.pallet_id, placements, units_on_pallet, sum(top.counts.values()), sum(bottom.counts.values()), preview_utilization, f'Candidate pallet {pallet.pallet_id} removes {units_on_pallet} windows from remaining job inventory. Each side independently satisfies the non-increasing taper rule (row widths do not increase as rows move outward from the center).', config_side_counts, balance_status, balance_penalty, pallet.pallet_cost)

    def candidate_score(self, load: JobPalletLoad, difficulty: Dict[int, float]) -> float:
        removed_difficulty = 0.0
        removed_units = 0
        for cid, (top_count, bottom_count) in load.config_side_counts.items():
            removed = top_count + bottom_count
            removed_units += removed
            removed_difficulty += removed * difficulty.get(cid, 0.0)
        used_area = sum(p.length * p.depth for p in load.placements)
        # Favor stable pallets by penalizing top/bottom imbalance (balance_penalty is sum of per-config side diffs).
        return removed_difficulty * 80.0 + used_area + removed_units * 1000.0 - (getattr(load, 'balance_penalty', 0.0) * 2500.0)

    def build_job_plan(self, configs: List[ConfigurationItem]) -> JobPlanResult:
        if not self.pallets:
            return JobPlanResult(False, None, sum(c.qty for c in configs), None, None, [], 'No allowable pallets are currently selected.', None, {}, 0.0)
        config_map = {c.config_id: c for c in configs}
        remaining = {c.config_id: int(c.qty) for c in configs}
        all_feasible = self.all_feasible_options_by_pallet(configs)
        difficulty = self.config_difficulty_scores(configs, all_feasible)
        # Separate configs that fit no pallet from those that do.  Rather than
        # aborting the whole job when one window is infeasible, we plan the
        # feasible configs and report the excluded ones so the user knows
        # exactly which windows must be handled outside the tool.
        excluded: List[Tuple[str, str]] = []
        feasible_configs: List[ConfigurationItem] = []
        for c in configs:
            feasible_any = any(all_feasible[pallet.pallet_id].get(c.config_id) for pallet in self.pallets)
            if feasible_any:
                feasible_configs.append(c)
            else:
                # Build a concise reason by inspecting the largest pallet.
                widest = max(self.pallets, key=lambda p: p.max_length)
                long_side = max(c.width, c.height)
                if long_side > widest.max_length:
                    reason = (f'long side {long_side:.0f}" exceeds largest pallet length '
                              f'{widest.max_length:.0f}"')
                else:
                    reason = 'fails height or brace rule on all pallets'
                excluded.append((c.label, reason))

        if not feasible_configs:
            detail = '; '.join(f'{lbl} ({rsn})' for lbl, rsn in excluded)
            return JobPlanResult(False, None, sum(c.qty for c in configs), None, None, [],
                'No configuration fits any selected pallet under the current rules. '
                f'Excluded: {detail}', None, {}, 0.0, excluded)

        # From here on, only plan the feasible configs.
        configs = feasible_configs
        config_map = {c.config_id: c for c in configs}
        remaining = {c.config_id: int(c.qty) for c in configs}
        pallet_loads: List[JobPalletLoad] = []
        pallet_mix_summary: Dict[str, int] = {}
        total_used_area = 0.0
        total_balance_penalty = 0.0
        total_cost = 0.0
        pallet_number = 1
        while sum(remaining.values()) > 0:
            candidate_loads: List[Tuple[float, JobPalletLoad]] = []
            for pallet in self.pallets:
                load = self.build_single_pallet_candidate(pallet, remaining, config_map, all_feasible[pallet.pallet_id], difficulty)
                if load is None or load.units_on_pallet <= 0:
                    continue
                score = self.candidate_score(load, difficulty)
                candidate_loads.append((score, load))
            if not candidate_loads:
                return JobPlanResult(False, None, sum(c.qty for c in configs), None, None, pallet_loads, 'The mixed-pallet job planner could not place all remaining configurations while satisfying the selected pallet list and the pyramid support rule.', None, pallet_mix_summary, total_cost)
            candidate_loads.sort(key=lambda t: (-t[0], -t[1].units_on_pallet, -(t[1].preview_utilization or 0.0), t[1].pallet_cost_each))
            chosen = candidate_loads[0][1]
            chosen.pallet_number = pallet_number
            pallet_loads.append(chosen)
            pallet_mix_summary[chosen.pallet_id] = pallet_mix_summary.get(chosen.pallet_id, 0) + 1
            total_balance_penalty += chosen.balance_penalty
            total_cost += chosen.pallet_cost_each
            total_used_area += sum(p.length * p.depth for p in chosen.placements)
            for cid, (top_count, bottom_count) in chosen.config_side_counts.items():
                remaining[cid] -= (top_count + bottom_count)
            pallet_number += 1
        total_units = sum(c.qty for c in configs)
        total_usable_area = sum(self.usable_area_for_pallet(self.pallet_by_id(load.pallet_id)) for load in pallet_loads)
        pallets_needed = len(pallet_loads)
        overall_utilization = (total_used_area / total_usable_area) if total_usable_area else 0.0
        avg_utilization = (sum(load.preview_utilization for load in pallet_loads) / pallets_needed) if pallets_needed else 0.0
        explanation = f'Job plan built pallet-by-pallet using mixed pallet sizes. Total pallets needed = {pallets_needed}. Overall utilization = {overall_utilization:.2%}. Average pallet utilization = {avg_utilization:.2%}. Estimated total pallet cost = ${total_cost:,.2f}. Combined supported base widths from the center outward satisfy the non-increasing pyramid support rule on each pallet. Balance information is warning-only and does not drive pallet selection. By Job objective: fit all windows across the fewest selected pallets possible using the allowable pallet sizes.'
        if excluded:
            excl_detail = '; '.join(f'{lbl} ({rsn})' for lbl, rsn in excluded)
            explanation += (f' NOTE: {len(excluded)} configuration(s) were excluded because they '
                            f'fit no selected pallet and must be handled outside the tool: {excl_detail}.')
        return JobPlanResult(True, pallets_needed, total_units, overall_utilization, avg_utilization, pallet_loads, explanation, total_balance_penalty, pallet_mix_summary, total_cost, excluded)

    def evaluate_job(self, configs: List[ConfigurationItem]) -> JobPlanResult:
        return self.build_job_plan(configs)


# Preview and entry helpers

def get_config_color(config_id: int) -> str:
    palette = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f']
    return palette[(config_id - 1) % len(palette)]


def _pallet_png_bytes(
    pallet: Pallet,
    placements: List[Placement],
    title: str,
    width: int = 900,
    height: int = 500,
) -> bytes:
    """Render a pallet bottom-view layout to PNG bytes using matplotlib.

    This is a browser-free replacement for the kaleido/Plotly image path,
    so PDF generation works on headless servers (e.g. Streamlit Cloud)
    without Chrome/Edge/Chromium installed. It mirrors build_plotly_preview.
    """
    import matplotlib
    matplotlib.use('Agg')  # headless backend, no display/browser required
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dpi = 100
    fig, ax = plt.subplots(figsize=(width / dpi, height / dpi), dpi=dpi)

    # Pallet outline
    ax.add_patch(Rectangle((0, 0), pallet.base_length, pallet.base_width,
                           fill=False, edgecolor='black', linewidth=2.5))
    # Bottom side band (usable)
    ax.add_patch(Rectangle((0, 0), pallet.base_length, pallet.max_depth_per_side,
                           facecolor='#ffc18c', alpha=0.18, edgecolor='none'))
    # Center frame
    ax.add_patch(Rectangle((0, pallet.max_depth_per_side), pallet.base_length, pallet.center_depth,
                           facecolor='#a0a0a0', alpha=0.65, edgecolor='black', linewidth=1.5))
    # Top side band (usable)
    top_y = pallet.max_depth_per_side + pallet.center_depth
    ax.add_patch(Rectangle((0, top_y), pallet.base_length, pallet.base_width - top_y,
                           facecolor='#ffc18c', alpha=0.18, edgecolor='none'))
    ax.text(pallet.base_length / 2, pallet.max_depth_per_side + pallet.center_depth / 2,
            'Center Frame', ha='center', va='center', fontsize=11, color='black')

    # Placed units
    for p in placements:
        ax.add_patch(Rectangle((p.x, p.y), p.length, p.depth,
                               fill=False, edgecolor=get_config_color(p.config_id), linewidth=1.8))
        ax.text(p.x + p.length / 2, p.y + p.depth / 2,
                f'{p.config_label}\n{p.orientation}',
                ha='center', va='center', fontsize=7, color=(0, 0, 0, 0.75))

    ax.set_xlim(0, pallet.base_length)
    ax.set_ylim(0, pallet.base_width)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Pallet Length (inches)')
    ax.set_ylabel('Pallet Width / Side Depth (inches)')
    ax.set_title(title)
    ax.grid(True, color='#e6e6e6', linewidth=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()

    out = io.BytesIO()
    fig.savefig(out, format='png', dpi=dpi)
    plt.close(fig)
    out.seek(0)
    return out.getvalue()



def build_plotly_preview(pallet: Pallet, placements: List[Placement], title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_shape(type='rect', x0=0, y0=0, x1=pallet.base_length, y1=pallet.base_width, line=dict(color='black', width=3), fillcolor='rgba(0,0,0,0)')
    fig.add_shape(type='rect', x0=0, y0=0, x1=pallet.base_length, y1=pallet.max_depth_per_side, line=dict(color='rgba(0,0,0,0)'), fillcolor='rgba(255,193,140,0.18)')
    fig.add_shape(type='rect', x0=0, y0=pallet.max_depth_per_side, x1=pallet.base_length, y1=pallet.max_depth_per_side + pallet.center_depth, line=dict(color='black', width=2), fillcolor='rgba(160,160,160,0.65)')
    fig.add_shape(type='rect', x0=0, y0=pallet.max_depth_per_side + pallet.center_depth, x1=pallet.base_length, y1=pallet.base_width, line=dict(color='rgba(0,0,0,0)'), fillcolor='rgba(255,193,140,0.18)')
    fig.add_annotation(x=pallet.base_length / 2, y=pallet.max_depth_per_side + pallet.center_depth / 2, text='Center Frame', showarrow=False, font=dict(size=13, color='black'))
    for p in placements:
        cx, cy = p.x + p.length / 2, p.y + p.depth / 2
        fig.add_shape(type='rect', x0=p.x, y0=p.y, x1=p.x + p.length, y1=p.y + p.depth,
                      line=dict(color=get_config_color(p.config_id), width=2),
                      fillcolor='rgba(255,255,255,0)')
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers',
                      marker=dict(size=8, color='rgba(0,0,0,0)'),
                      hovertemplate=(
                          f'Config: {p.config_label}<br>'
                          f'Pallet: {p.pallet_id}<br>'
                          f'Unit #{p.unit_number}<br>'
                          f'Side: {p.side}<br>'
                          f'Row: {p.row_number}<br>'
                          f'Orientation: {p.orientation}<br>'
                          f'Base side: {p.length:.3f}"<br>'
                          f'Depth used: {p.depth:.3f}"<extra></extra>'
                      ),
                      showlegend=False))
                
        fig.add_trace(go.Scatter(
            x=[cx],
            y=[cy],
            text=[f"{p.config_label}<br>{p.orientation}"],
            mode='text',
            textfont=dict(
                size=9,  # bump slightly for two lines
                color='rgba(0,0,0,0.7)'
            ),
            textposition='middle center',
            hoverinfo='skip',
            showlegend=False
        ))
    fig.update_xaxes(title='Pallet Length (inches)', range=[0, pallet.base_length], showgrid=True, zeroline=False, scaleanchor='y', scaleratio=1)
    fig.update_yaxes(title='Pallet Width / Side Depth (inches)', range=[0, pallet.base_width], showgrid=True, zeroline=False)
    fig.update_layout(title=title, height=560, margin=dict(l=20, r=20, t=50, b=20), plot_bgcolor='white', hovermode='closest')
    return fig

def clean_string_values(series: pd.Series) -> List[str]:
    return sorted(series.astype('string').dropna().unique().tolist())


def build_default_job_items(lookup: ProductDepthLookup) -> List[dict]:
    seeds = [('Cfg 1', 'AA4325', 'PI', 1, 33.438, 18.0, 28)]
    out = []
    for label, fam, typ, depth_option, width, height, qty in seeds:
        d = lookup.get_default_depth(fam, typ, depth_option)
        out.append({'label': label, 'product_family': fam, 'product_type': typ, 'depth_option': depth_option, 'package_depth': 0.0 if d is None else float(d), 'side_down': 'Auto', 'width': width, 'height': height, 'qty': qty})
    return out


def build_job_items_df(items: List[dict]) -> pd.DataFrame:
    rows = []
    for i, item in enumerate(items, start=1):
        rows.append({'Row': i, 'Label': item.get('label', ''), 'Family': item.get('product_family', ''), 'Type': item.get('product_type', ''), 'Depth Option': item.get('depth_option', 1), 'Depth': item.get('package_depth', 0.0), 'Side Down': item.get('side_down', 'Auto'), 'Width': item.get('width', 0.0), 'Height': item.get('height', 0.0), 'Qty': item.get('qty', 0)})
    return pd.DataFrame(rows)


def ensure_job_form_defaults(lookup: ProductDepthLookup) -> None:
    defaults = {'jf_label': 'Cfg 1', 'jf_family': 'AA4325', 'jf_type': 'PI', 'jf_depth_option': 1, 'jf_depth': float(lookup.get_default_depth('AA4325', 'PI', 1) or 0.0), 'jf_side_down': 'Auto', 'jf_width': 33.438, 'jf_height': 18.0, 'jf_qty': 1}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def load_form_from_item(item: dict) -> None:
    st.session_state['jf_label'] = item.get('label', '')
    st.session_state['jf_family'] = item.get('product_family', '')
    st.session_state['jf_type'] = item.get('product_type', '')
    st.session_state['jf_depth_option'] = int(item.get('depth_option', 1) or 1)
    st.session_state['jf_depth'] = float(item.get('package_depth', 0.0) or 0.0)
    st.session_state['jf_side_down'] = item.get('side_down', 'Auto')
    st.session_state['jf_width'] = float(item.get('width', 0.0) or 0.0)
    st.session_state['jf_height'] = float(item.get('height', 0.0) or 0.0)
    st.session_state['jf_qty'] = int(item.get('qty', 1) or 1)


def current_form_item(lookup: ProductDepthLookup) -> dict:
    fam = str(st.session_state.get('jf_family', '')).strip()
    typ = str(st.session_state.get('jf_type', '')).strip()
    depth_option = int(st.session_state.get('jf_depth_option', 1))
    depth_used = float(st.session_state.get('jf_depth', 0.0) or 0.0)
    if depth_used <= 0:
        default_depth = lookup.get_default_depth(fam, typ, depth_option)
        depth_used = 0.0 if default_depth is None else float(default_depth)
    return {'label': str(st.session_state.get('jf_label', '')).strip() or 'Config', 'product_family': fam, 'product_type': typ, 'depth_option': depth_option, 'package_depth': depth_used, 'side_down': str(st.session_state.get('jf_side_down', 'Auto') or 'Auto'), 'width': float(st.session_state.get('jf_width', 0.0) or 0.0), 'height': float(st.session_state.get('jf_height', 0.0) or 0.0), 'qty': int(st.session_state.get('jf_qty', 0) or 0)}


def items_to_configs(items: List[dict]) -> List[ConfigurationItem]:
    out: List[ConfigurationItem] = []
    for item in items:
        fam = str(item.get('product_family', '')).strip()
        typ = str(item.get('product_type', '')).strip()
        qty = int(item.get('qty', 0) or 0)
        depth = float(item.get('package_depth', 0.0) or 0.0)
        if not fam or not typ or qty <= 0 or depth <= 0:
            continue
        out.append(ConfigurationItem(config_id=len(out) + 1, label=str(item.get('label', '')).strip() or f'Config {len(out)+1}', product_family=fam, product_type=typ, depth_option=int(item.get('depth_option', 1) or 1), package_depth=depth, width=float(item.get('width', 0.0) or 0.0), height=float(item.get('height', 0.0) or 0.0), qty=qty, side_down=str(item.get('side_down', 'Auto') or 'Auto')))
    return out


def configs_to_rows(configs: List[ConfigurationItem]) -> Tuple[Tuple, ...]:
    return tuple(
        (c.config_id, c.label, c.product_family, c.product_type, c.depth_option,
         c.package_depth, c.width, c.height, c.qty, c.side_down)
        for c in configs
    )


@st.cache_data(show_spinner=False)
def run_job_plan_cached(config: dict, depth_csv_path: str, allowed: Tuple[str, ...], config_rows: Tuple[Tuple, ...]) -> JobPlanResult:
    """Cache the mixed-job optimization so it only recomputes when an input
    that affects the plan actually changes (pallet config contents, allowed
    pallet list, or the configuration rows) — not on every widget interaction."""
    depth_df = load_depths(depth_csv_path)
    optimizer = MixedJobOptimizer(config, depth_df, allowed_pallet_ids=list(allowed))
    configs = [ConfigurationItem(*row) for row in config_rows]
    return optimizer.build_job_plan(configs)


def balance_table_for_load(load: JobPalletLoad, configs: List[ConfigurationItem]) -> pd.DataFrame:
    rows = []
    config_lookup = {c.config_id: c for c in configs}
    for cid, (top_count, bottom_count) in load.config_side_counts.items():
        if top_count == 0 and bottom_count == 0:
            continue
        diff = abs(top_count - bottom_count)
        status = 'Balanced' if diff <= 1 else ('Slight imbalance' if diff <= 2 else 'High risk')
        label = config_lookup[cid].label if cid in config_lookup else f'Config {cid}'
        rows.append({'Configuration': label, 'Top Side': top_count, 'Bottom Side': bottom_count, 'Difference': diff, 'Status': status})
    return pd.DataFrame(rows)


def pallet_mix_table(pallet_mix: Dict[str, int], optimizer: BaseOptimizer) -> pd.DataFrame:
    rows = []
    for pid, cnt in sorted(pallet_mix.items()):
        p = optimizer.pallet_by_id(pid)
        rows.append({'Pallet Type': pid, 'Count': cnt, 'Cost Each': p.pallet_cost, 'Total Cost': p.pallet_cost * cnt})
    return pd.DataFrame(rows)


# Pallet settings editor

PALLET_NUMERIC_FIELDS = ['base_length', 'base_width', 'center_height', 'center_depth',
                         'max_depth_per_side', 'max_height', 'max_length',
                         'usable_space_per_side', 'pallet_cost']


def render_pallet_settings(config: dict) -> None:
    st.subheader('Pallet Settings')
    saved_msg = st.session_state.pop('pallet_settings_saved_msg_v45', None)
    if saved_msg:
        st.success(saved_msg)
    st.caption(
        'Edit pallet sizes and costs directly in the table. Use the empty row at the bottom to add a new '
        'pallet size; select a row and use the trash icon to remove one. Press Save to write the changes '
        f'to {DEFAULT_CONFIG_PATH.name} — they take effect immediately in both optimizer modes.'
    )
    pallets_df = pd.DataFrame([{k: p.get(k) for k in ['pallet_id'] + PALLET_NUMERIC_FIELDS} for p in config['pallets']])
    edited_df = st.data_editor(
        pallets_df,
        num_rows='dynamic',
        use_container_width=True,
        hide_index=True,
        key='pallet_settings_editor_v45',
        column_config={
            'pallet_id': st.column_config.TextColumn('Pallet ID', required=True, help='Unique name, e.g. 96x46.'),
            'base_length': st.column_config.NumberColumn('Base Length (in)', min_value=0.0, step=0.5),
            'base_width': st.column_config.NumberColumn('Base Width (in)', min_value=0.0, step=0.5),
            'center_height': st.column_config.NumberColumn('Center Height (in)', min_value=0.0, step=0.5),
            'center_depth': st.column_config.NumberColumn('Center Depth (in)', min_value=0.0, step=0.5),
            'max_depth_per_side': st.column_config.NumberColumn('Max Depth / Side (in)', min_value=0.0, step=0.5),
            'max_height': st.column_config.NumberColumn('Max Height (in)', min_value=0.0, step=0.5),
            'max_length': st.column_config.NumberColumn('Max Length (in)', min_value=0.0, step=0.5),
            'usable_space_per_side': st.column_config.NumberColumn('Usable Area / Side (sq in)', min_value=0.0, step=1.0, help='Leave 0 or blank to auto-calculate as Max Length × Max Depth / Side on save.'),
            'pallet_cost': st.column_config.NumberColumn('Pallet Cost ($)', min_value=0.0, step=0.1, format='$%.2f'),
        },
    )
    if st.button('Save Pallet Settings', type='primary'):
        errors: List[str] = []
        cleaned: List[dict] = []
        seen_ids = set()
        required_positive = [f for f in PALLET_NUMERIC_FIELDS if f not in ('usable_space_per_side', 'pallet_cost')]
        for row_no, (_, row) in enumerate(edited_df.iterrows(), start=1):
            pid = str(row.get('pallet_id') or '').strip()
            if not pid or pid.lower() == 'nan':
                errors.append(f'Row {row_no}: Pallet ID is required.')
                continue
            if pid in seen_ids:
                errors.append(f'Row {row_no}: duplicate Pallet ID "{pid}".')
                continue
            seen_ids.add(pid)
            entry: dict = {'pallet_id': pid}
            row_ok = True
            for field in PALLET_NUMERIC_FIELDS:
                val = row.get(field)
                val = 0.0 if val is None or pd.isna(val) else float(val)
                if val < 0:
                    errors.append(f'Row {row_no} ({pid}): {field} cannot be negative.')
                    row_ok = False
                entry[field] = val
            if row_ok:
                for field in required_positive:
                    if entry[field] <= 0:
                        errors.append(f'Row {row_no} ({pid}): {field} must be greater than zero.')
                        row_ok = False
            if not row_ok:
                continue
            if entry['usable_space_per_side'] <= 0:
                entry['usable_space_per_side'] = entry['max_length'] * entry['max_depth_per_side']
            cleaned.append(entry)
        if not cleaned and not errors:
            errors.append('At least one pallet is required.')
        if errors:
            for e in errors:
                st.error(e)
            st.warning('Nothing was saved. Fix the errors above and press Save again.')
        else:
            new_config = dict(config)
            new_config['pallets'] = cleaned
            DEFAULT_CONFIG_PATH.write_text(json.dumps(new_config, indent=2))
            load_config.clear()
            run_job_plan_cached.clear()
            st.session_state['pallet_settings_saved_msg_v45'] = f'Saved {len(cleaned)} pallet definition(s) to {DEFAULT_CONFIG_PATH.name}.'
            st.rerun()


# Main UI
def main():
    st.set_page_config(page_title='Pallet Optimizer V4.5', layout='wide')
    apply_custom_css()

    ensure_data_files()  # first-run setup: seed default config + depth files
    config = load_config(str(DEFAULT_CONFIG_PATH))
    depth_df = load_depths(str(DEFAULT_DEPTH_CSV_PATH))
    lookup = ProductDepthLookup(depth_df)

    all_pallet_ids = [p['pallet_id'] for p in config['pallets']]

    if 'job_items_v37' not in st.session_state:
        st.session_state['job_items_v37'] = build_default_job_items(lookup)
    if 'selected_job_row_v37' not in st.session_state:
        st.session_state['selected_job_row_v37'] = 1
    ensure_job_form_defaults(lookup)

    mode = st.sidebar.radio('Mode', ['By Configuration', 'By Job', 'Pallet Settings'], index=0)
    if mode != 'Pallet Settings':
        allowable_pallets = st.sidebar.multiselect('Allowable Pallets', options=all_pallet_ids, default=all_pallet_ids, help='Select the pallet sizes the optimizer is allowed to use in the current mode.')

    if mode == 'Pallet Settings':
        render_pallet_settings(config)
    elif mode == 'By Configuration':
        optimizer = SingleConfigOptimizer(config, depth_df, allowed_pallet_ids=allowable_pallets)
        if optimizer.pallets_missing_cost:
            st.sidebar.warning(f"No pallet_cost set in the config JSON for: {', '.join(optimizer.pallets_missing_cost)}. Costs will show as $0.00. Add a \"pallet_cost\" value to each pallet entry in pallet_config_seeded.json.")
        job_name = st.sidebar.text_input('Job Name', value=st.session_state.get('job_name_v45', ''), help='Optional. Appears on the PDF report title and summary.')
        st.session_state['job_name_v45'] = job_name
        families = clean_string_values(depth_df['product_family'])
        selected_family = st.sidebar.selectbox('Product Family', families, index=0 if families else None)
        type_series = depth_df.loc[depth_df['product_family'].astype('string') == selected_family, 'product_type']
        family_types = clean_string_values(type_series)
        selected_type = st.sidebar.selectbox('Product Type', family_types, index=0 if family_types else None)
        depth_option = st.sidebar.selectbox('Depth Option', [1, 2], index=0)
        default_depth = lookup.get_default_depth(selected_family or '', selected_type or '', int(depth_option))
        selection_signature = (selected_family, selected_type, int(depth_option))
        if st.session_state.get('cfg_depth_sig_v37') != selection_signature:
            st.session_state['cfg_depth_sig_v37'] = selection_signature
            st.session_state['cfg_depth_used_v37'] = 0.0 if default_depth is None else float(default_depth)
        package_depth = st.sidebar.number_input('Depth (inches)', min_value=0.0, value=float(st.session_state.get('cfg_depth_used_v37', 0.0)), step=0.125, help='Prefilled from the selected family / type / depth option. If changed, the optimizer uses the value shown here.')
        st.session_state['cfg_depth_used_v37'] = package_depth
        side_down = st.sidebar.selectbox('Side Down', BaseOptimizer.SIDE_OPTIONS, index=0)
        width = st.sidebar.number_input('Width (inches)', min_value=0.0, value=80.0, step=0.25)
        height = st.sidebar.number_input('Height (inches)', min_value=0.0, value=55.0, step=0.25)
        qty = st.sidebar.number_input('Order Quantity', min_value=1, value=10, step=1)
        show_all = st.sidebar.checkbox('Show all pallet evaluations', value=True)
        job = Job(selected_family, selected_type, int(depth_option), float(package_depth), float(width), float(height), int(qty), side_down)
        best, results, depth_used = optimizer.evaluate_job(job)

        tab1, tab2, tab3 = st.tabs(['Overview', 'Preview', 'All Pallets'])
        with tab1:
            left, right = st.columns([1.0, 1.25])
            with left:
                st.subheader('Input Summary')
                st.write({'Mode': mode, 'Allowable Pallets': allowable_pallets, 'Product Family': selected_family, 'Product Type': selected_type, 'Depth Option': depth_option, 'Default Lookup Depth': default_depth, 'Depth Used': depth_used, 'Side Down Input': side_down, 'Width': width, 'Height': height, 'Quantity': qty})
                
            with right:
                st.subheader('Best Result Summary')
                if not allowable_pallets:
                    st.error('Select at least one allowable pallet to run the optimizer.')
                elif depth_used is None:
                    st.error('No default product depth was found for the selected family / type / depth option.')
                elif best is None:
                    st.error('No selected pallet passed all current rules for this configuration.')
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric('Best Pallet', best.pallet_id)
                    m2.metric('Pallets Needed', best.pallets_needed)
                    m3.metric('Max Units / Pallet', best.max_units_per_pallet)
                    m4.metric('Preview Units', best.units_on_preview_pallet)
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric('Chosen Orientation', best.chosen_orientation)
                    s2.metric('Top Side Units', best.units_top_side)
                    s3.metric('Bottom Side Units', best.units_bottom_side)
                    s4.metric('Depth Used', f'{best.package_depth:.3f}\"' if best.package_depth is not None else '-')
                    c1, c2 = st.columns(2)
                    c1.metric('Pallet Cost Each', f'${best.pallet_cost_each:,.2f}')
                    c2.metric('Estimated Total Cost', f'${best.estimated_total_cost:,.2f}')
                    st.success('A feasible pallet was found under the selected pallets and the corrected pyramid support rule.')
                    st.markdown(f'**Explanation:** {best.explanation}')
                    if best.ranking_reason:
                        st.caption(best.ranking_reason)
                    dl_col1, dl_col2 = st.columns(2)
                    dl_col1.download_button('Export Summary (.xlsx)', data=export_by_configuration(job, best, results, allowable_pallets), file_name='pallet_summary_by_configuration.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                    # PDF is expensive (kaleido renders every pallet image).
                    # Only generate when explicitly requested. A single fixed
                    # session key holds the latest PDF (with a signature of the
                    # inputs) so old reports don't pile up in session memory.
                    cfg_pdf_sig = (best.pallet_id, job.qty, job.package_depth, job_name)
                    if dl_col2.button('Generate PDF Report', key='gen_pdf_cfg'):
                        try:
                            with st.spinner('Rendering pallet images…'):
                                st.session_state['pdf_cfg_bytes_v45'] = export_pdf_by_configuration(job, best, results, allowable_pallets, optimizer, job_name=job_name)
                                st.session_state['pdf_cfg_sig_v45'] = cfg_pdf_sig
                        except Exception as exc:
                            st.error(f'Could not render the PDF report: {exc}. The Excel export above always works regardless.')
                    if st.session_state.get('pdf_cfg_sig_v45') == cfg_pdf_sig and 'pdf_cfg_bytes_v45' in st.session_state:
                        safe_name = ''.join(ch for ch in job_name.strip() if ch.isalnum() or ch in (' ', '-', '_')).strip().replace(' ', '_')
                        pdf_filename = f'{safe_name}_pallet_report.pdf' if safe_name else 'pallet_report_by_configuration.pdf'
                        dl_col2.download_button('Download PDF Report (.pdf)', data=st.session_state['pdf_cfg_bytes_v45'], file_name=pdf_filename, mime='application/pdf', key='dl_pdf_cfg')
        with tab2:
            if best is not None:
                chosen_pallet = optimizer.pallet_by_id(best.pallet_id)
                pallet_count = best.pallets_needed or 1
                selector_key = f'cfg_preview_v37_{best.pallet_id}_{job.package_depth}_{job.qty}'
                if selector_key not in st.session_state or st.session_state[selector_key] > pallet_count:
                    st.session_state[selector_key] = 1
                selected_pallet_num = st.radio('Pallet to Preview', options=list(range(1, pallet_count + 1)), index=st.session_state[selector_key] - 1, horizontal=True, key=f'{selector_key}_radio') if pallet_count <= 12 else st.selectbox('Pallet to Preview', options=list(range(1, pallet_count + 1)), index=st.session_state[selector_key] - 1, key=f'{selector_key}_select')
                st.session_state[selector_key] = selected_pallet_num
                units_on_selected_pallet = optimizer.units_for_pallet_sequence(job.qty, best.max_units_per_pallet or 1, selected_pallet_num)
                selected_preview = optimizer.build_preview_for_units(chosen_pallet, best, units_on_selected_pallet, selected_pallet_num)
                st.plotly_chart(build_plotly_preview(chosen_pallet, selected_preview.placements or [], f'Interactive Bottom View — {chosen_pallet.pallet_id} — Pallet #{selected_pallet_num}'), use_container_width=True)
        with tab3:
            if show_all:
                rows = []
                for r in results:
                    rows.append({'Pallet': r.pallet_id, 'Feasible': r.feasible, 'Orientation': r.chosen_orientation, 'Pallets Needed': r.pallets_needed, 'Max Units/Pallet': r.max_units_per_pallet, 'Pallet Cost Each': r.pallet_cost_each, 'Estimated Total Cost': r.estimated_total_cost, 'Preview Utilization %': None if r.preview_utilization is None else round(r.preview_utilization * 100, 2), 'Capacity Utilization %': None if r.capacity_utilization is None else round(r.capacity_utilization * 100, 2), 'Reason / Explanation': r.rejection_reason if not r.feasible else r.explanation})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    else:
        optimizer = MixedJobOptimizer(config, depth_df, allowed_pallet_ids=allowable_pallets)
        if optimizer.pallets_missing_cost:
            st.sidebar.warning(f"No pallet_cost set in the config JSON for: {', '.join(optimizer.pallets_missing_cost)}. Costs will show as $0.00. Add a \"pallet_cost\" value to each pallet entry in pallet_config_seeded.json.")

        st.subheader('Mixed Configuration Job Setup')
        setup_c1, setup_c2, setup_c3 = st.columns([1.2, 0.8, 1.4])
        with setup_c1:
            job_name = st.text_input('Job Name', value=st.session_state.get('job_name_v45', ''), help='Optional. Appears on the PDF report title and summary.')
            st.session_state['job_name_v45'] = job_name
        with setup_c2:
            job_payload = {
                'job_name': job_name,
                'saved_at': pd.Timestamp.now().isoformat(),
                'items': st.session_state['job_items_v37'],
            }
            safe_job_file = ''.join(ch for ch in job_name.strip() if ch.isalnum() or ch in (' ', '-', '_')).strip().replace(' ', '_')
            st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
            st.download_button(
                'Save Job (.json)',
                data=json.dumps(job_payload, indent=2),
                file_name=f'{safe_job_file or "pallet_job"}.json',
                mime='application/json',
                use_container_width=True,
                help='Downloads the current configuration list (and job name) as a JSON file you can reload later.',
            )
        with setup_c3:
            uploaded_job = st.file_uploader('Load Job (.json)', type=['json'], key='job_loader_v45',
                                            help='Restores a previously saved job, replacing the current configuration list.')
        if uploaded_job is not None:
            load_sig = (uploaded_job.name, uploaded_job.size)
            if st.session_state.get('job_loaded_sig_v45') != load_sig:
                try:
                    payload = json.loads(uploaded_job.getvalue().decode('utf-8'))
                    loaded_items = payload.get('items', [])
                    if not isinstance(loaded_items, list):
                        raise ValueError('Invalid job file: "items" must be a list.')
                    required_keys = {'label', 'product_family', 'product_type', 'depth_option',
                                     'package_depth', 'side_down', 'width', 'height', 'qty'}
                    for it in loaded_items:
                        if not isinstance(it, dict) or not required_keys.issubset(it.keys()):
                            raise ValueError('Invalid job file: a configuration entry is missing required fields.')
                    st.session_state['job_items_v37'] = loaded_items
                    st.session_state['selected_job_row_v37'] = 1 if loaded_items else 1
                    loaded_name = str(payload.get('job_name', '') or '')
                    if loaded_name:
                        st.session_state['job_name_v45'] = loaded_name
                    if loaded_items:
                        load_form_from_item(loaded_items[0])
                    st.session_state['job_loaded_sig_v45'] = load_sig
                    st.success(f'Loaded job with {len(loaded_items)} configuration(s).')
                    st.rerun()
                except Exception as exc:
                    st.error(f'Could not load job file: {exc}')
                    st.session_state['job_loaded_sig_v45'] = load_sig

        # ── Reset all configurations (sits next to Save / Load) ──────────────
        _reset_items = st.session_state['job_items_v37']
        if not st.session_state.get('confirm_reset_v46', False):
            if st.button('Reset All Configurations', use_container_width=True,
                         disabled=not bool(_reset_items),
                         help='Clears every configuration currently loaded into the Pallet Optimizer.'):
                st.session_state['confirm_reset_v46'] = True
                st.rerun()
        else:
            st.warning(
                f'This will clear all {len(_reset_items)} configuration(s) loaded into the Pallet Optimizer. '
                'This cannot be undone. Use Save Job (.json) first if you want to keep this job.'
            )
            reset_c1, reset_c2 = st.columns(2)
            if reset_c1.button('Yes, clear all configurations', type='primary', use_container_width=True):
                st.session_state['job_items_v37'] = []
                st.session_state['selected_job_row_v37'] = 1
                st.session_state['confirm_reset_v46'] = False
                for k in list(st.session_state.keys()):
                    if str(k).startswith('pdf_job_'):
                        del st.session_state[k]
                st.success('All configurations cleared.')
                st.rerun()
            if reset_c2.button('Cancel', use_container_width=True):
                st.session_state['confirm_reset_v46'] = False
                st.rerun()

        items = st.session_state['job_items_v37']
        if items and st.session_state['selected_job_row_v37'] > len(items):
            st.session_state['selected_job_row_v37'] = len(items)
        if not items:
            st.session_state['selected_job_row_v37'] = 1

        st.subheader('Add / Edit Configuration')
        ensure_job_form_defaults(lookup)
        current_sig = (str(st.session_state.get('jf_family', '')).strip(), str(st.session_state.get('jf_type', '')).strip(), int(st.session_state.get('jf_depth_option', 1)))
        if st.session_state.get('jf_last_sig_v37') != current_sig:
            looked_up = lookup.get_default_depth(current_sig[0], current_sig[1], current_sig[2])
            if looked_up is not None and float(st.session_state.get('jf_depth', 0.0) or 0.0) <= 0:
                st.session_state['jf_depth'] = float(looked_up)
            st.session_state['jf_last_sig_v37'] = current_sig
        c1, c2, c3, c4 = st.columns(4)
        st.session_state['jf_label'] = c1.text_input('Label', value=st.session_state.get('jf_label', 'Cfg 1'))
        st.session_state['jf_family'] = c2.text_input('Product Family', value=st.session_state.get('jf_family', 'AA4325'))
        st.session_state['jf_type'] = c3.text_input('Product Type', value=st.session_state.get('jf_type', 'PI'))
        st.session_state['jf_depth_option'] = int(c4.number_input('Depth Option', min_value=1, max_value=2, value=int(st.session_state.get('jf_depth_option', 1)), step=1))
        default_depth = lookup.get_default_depth(str(st.session_state.get('jf_family', '')).strip(), str(st.session_state.get('jf_type', '')).strip(), int(st.session_state.get('jf_depth_option', 1)))
        c5, c6, c7, c8, c9 = st.columns(5)
        st.session_state['jf_depth'] = c5.number_input('Depth (inches)', min_value=0.0, value=float(st.session_state.get('jf_depth', 0.0)), step=0.125, help='Prefilled from lookup when available. If changed, the optimizer uses the entered depth.')
        current_side = st.session_state.get('jf_side_down', 'Auto')
        current_idx = BaseOptimizer.SIDE_OPTIONS.index(current_side) if current_side in BaseOptimizer.SIDE_OPTIONS else 0
        st.session_state['jf_side_down'] = c6.selectbox('Side Down', BaseOptimizer.SIDE_OPTIONS, index=current_idx)
        st.session_state['jf_width'] = c7.number_input('Width', min_value=0.0, value=float(st.session_state.get('jf_width', 0.0)), step=0.125)
        st.session_state['jf_height'] = c8.number_input('Height', min_value=0.0, value=float(st.session_state.get('jf_height', 0.0)), step=0.125)
        st.session_state['jf_qty'] = int(c9.number_input('Qty', min_value=0, value=int(st.session_state.get('jf_qty', 1)), step=1))
        st.caption(f'Default lookup depth for the current form selection: {default_depth if default_depth is not None else "Not found"}')
        buttons = st.columns(5)
        if buttons[0].button('Use Lookup Depth', use_container_width=True):
            if default_depth is not None:
                st.session_state['jf_depth'] = float(default_depth)
                st.rerun()
        if buttons[1].button('Add New', use_container_width=True):
            st.session_state['job_items_v37'].append(current_form_item(lookup))
            st.session_state['selected_job_row_v37'] = len(st.session_state['job_items_v37'])
            st.rerun()
        if buttons[2].button('Update Selected', use_container_width=True, disabled=not bool(items)):
            idx = st.session_state.get('selected_job_row_v37', 1) - 1
            if 0 <= idx < len(st.session_state['job_items_v37']):
                st.session_state['job_items_v37'][idx] = current_form_item(lookup)
            st.rerun()
        if buttons[3].button('Duplicate Selected', use_container_width=True, disabled=not bool(items)):
            idx = st.session_state.get('selected_job_row_v37', 1) - 1
            if 0 <= idx < len(st.session_state['job_items_v37']):
                dup = dict(st.session_state['job_items_v37'][idx])
                dup['label'] = f"{dup.get('label', 'Config')} Copy"
                st.session_state['job_items_v37'].insert(idx + 1, dup)
                st.session_state['selected_job_row_v37'] = idx + 2
            st.rerun()
        if buttons[4].button('Delete Selected', use_container_width=True, disabled=not bool(items)):
            idx = st.session_state.get('selected_job_row_v37', 1) - 1
            if 0 <= idx < len(st.session_state['job_items_v37']):
                st.session_state['job_items_v37'].pop(idx)
                if not st.session_state['job_items_v37']:
                    st.session_state['selected_job_row_v37'] = 1
                else:
                    st.session_state['selected_job_row_v37'] = max(1, min(idx + 1, len(st.session_state['job_items_v37'])))
                    load_form_from_item(st.session_state['job_items_v37'][st.session_state['selected_job_row_v37'] - 1])
            st.rerun()

        st.subheader('Configuration List')
        st.dataframe(build_job_items_df(items), use_container_width=True, hide_index=True)
        if items:
            selected_row = st.selectbox('Select Configuration Row', options=list(range(1, len(items) + 1)), index=max(st.session_state['selected_job_row_v37'] - 1, 0), key='selected_job_row_selector_v37')
            if selected_row != st.session_state.get('selected_job_row_v37', 1):
                st.session_state['selected_job_row_v37'] = selected_row
                load_form_from_item(items[selected_row - 1])
                st.rerun()
        else:
            st.info('No configurations are currently in the job. Use the form above to add one.')

        configs = items_to_configs(st.session_state['job_items_v37'])
        best_job = run_job_plan_cached(config, str(DEFAULT_DEPTH_CSV_PATH), tuple(sorted(allowable_pallets)), configs_to_rows(configs)) if configs else None

        st.markdown('---')
        job_tab1, job_tab2, job_tab3 = st.tabs(['Overview', 'Preview', 'Job Details'])
        with job_tab1:
            st.subheader('Best Result Summary')
            if not allowable_pallets:
                st.error('Select at least one allowable pallet to run the optimizer.')
            elif not configs:
                st.info('Add one or more configurations using the form to evaluate a mixed-configuration job.')
            elif best_job is None or not best_job.feasible:
                st.error('The mixed-pallet job planner could not place all configurations under the current rules, selected pallets, and corrected pyramid support rule.')
                if best_job is not None:
                    st.caption(best_job.explanation)
            else:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric('Total Pallets Needed', best_job.pallets_needed)
                m2.metric('Total Units', best_job.total_units)
                m3.metric('Avg Pallet Utilization', f'{(best_job.avg_pallet_utilization or 0.0) * 100:.1f}%')
                m4.metric('Overall Job Utilization', f'{(best_job.overall_utilization or 0.0) * 100:.1f}%')
                c1, c2, c3 = st.columns(3)
                c1.metric('Estimated Total Cost', f'${best_job.estimated_total_cost:,.2f}')
                c2.metric('Pallet Types Used', len(best_job.pallet_mix_summary))
                c3.metric('Balance Warning Total', f'{(best_job.total_balance_penalty or 0.0):.1f}')
                st.success('A mixed-pallet job plan was found under the selected pallet list and the corrected pyramid support rule.')
                if best_job.excluded_configs:
                    excl_lines = '\n'.join(f'- **{lbl}** — {rsn}' for lbl, rsn in best_job.excluded_configs)
                    st.warning(
                        f'{len(best_job.excluded_configs)} configuration(s) could not be placed on any '
                        f'selected pallet and were excluded from this plan. Handle these outside the tool:\n\n'
                        + excl_lines
                    )
                st.markdown(f'**Explanation:** {best_job.explanation}')
                st.dataframe(pallet_mix_table(best_job.pallet_mix_summary, optimizer), use_container_width=True, hide_index=True)
                dl_col1, dl_col2 = st.columns(2)
                dl_col1.download_button('Export Summary (.xlsx)', data=export_by_job(configs, best_job, allowable_pallets), file_name='pallet_summary_by_job.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

                job_pdf_sig = (best_job.pallets_needed, best_job.total_units, best_job.estimated_total_cost, job_name)
                if dl_col2.button('Generate PDF Report', key='gen_pdf_job'):
                    try:
                        with st.spinner('Rendering pallet images…'):
                            st.session_state['pdf_job_bytes_v45'] = export_pdf_by_job(configs, best_job, allowable_pallets, optimizer, job_name=job_name)
                            st.session_state['pdf_job_sig_v45'] = job_pdf_sig
                    except Exception as exc:
                        st.error(f'Could not render the PDF report: {exc}. The Excel export above always works regardless.')
                if st.session_state.get('pdf_job_sig_v45') == job_pdf_sig and 'pdf_job_bytes_v45' in st.session_state:
                    safe_name = ''.join(ch for ch in job_name.strip() if ch.isalnum() or ch in (' ', '-', '_')).strip().replace(' ', '_')
                    pdf_filename = f'{safe_name}_pallet_report.pdf' if safe_name else 'pallet_report_by_job.pdf'
                    dl_col2.download_button('Download PDF Report (.pdf)', data=st.session_state['pdf_job_bytes_v45'], file_name=pdf_filename, mime='application/pdf', key='dl_pdf_job')
        with job_tab2:
            if best_job is not None and best_job.feasible:
                pallet_count = best_job.pallets_needed or len(best_job.pallet_loads) or 1
                selector_key = f'job_preview_v37_{len(configs)}_{sum(c.qty for c in configs)}'
                if selector_key not in st.session_state or st.session_state[selector_key] > pallet_count:
                    st.session_state[selector_key] = 1
                selected_pallet_num = st.radio('Pallet to Preview', options=list(range(1, pallet_count + 1)), index=st.session_state[selector_key] - 1, horizontal=True, key=f'{selector_key}_radio') if pallet_count <= 12 else st.selectbox('Pallet to Preview', options=list(range(1, pallet_count + 1)), index=st.session_state[selector_key] - 1, key=f'{selector_key}_select')
                st.session_state[selector_key] = selected_pallet_num
                selected_load = best_job.pallet_loads[selected_pallet_num - 1]
                chosen_pallet = optimizer.pallet_by_id(selected_load.pallet_id)
                st.plotly_chart(build_plotly_preview(chosen_pallet, selected_load.placements, f'Interactive Bottom View — {chosen_pallet.pallet_id} — Job Pallet #{selected_load.pallet_number}'), use_container_width=True)
            else:
                st.info('Add or update at least one valid configuration to preview a job pallet.')
        with job_tab3:
            if best_job is not None and best_job.feasible:
                selector_key = f'job_preview_v37_{len(configs)}_{sum(c.qty for c in configs)}'
                selected_pallet_num = st.session_state.get(selector_key, 1)
                selected_load = best_job.pallet_loads[selected_pallet_num - 1]
                legend_rows = [{'Configuration': c.label, 'Color': get_config_color(c.config_id), 'Side Down': c.side_down, 'Qty': c.qty, 'Size': f'{c.width:.3f} x {c.height:.3f}', 'Depth Used': c.package_depth} for c in configs]
                st.dataframe(pd.DataFrame(legend_rows), use_container_width=True, hide_index=True)
                st.dataframe(balance_table_for_load(selected_load, configs), use_container_width=True, hide_index=True)
                row_summary = [{'Config': p.config_label, 'Side': p.side, 'Row': p.row_number, 'Base Side': round(p.length, 3), 'Depth Used': round(p.depth, 3), 'Orientation': p.orientation} for p in selected_load.placements]
                st.dataframe(pd.DataFrame(row_summary), use_container_width=True, hide_index=True)
                st.dataframe(pallet_mix_table(best_job.pallet_mix_summary, optimizer), use_container_width=True, hide_index=True)
            else:
                st.info('Job details will appear here once a feasible mixed job plan exists.')

    with st.expander('Show product depth lookup table'):
        st.dataframe(depth_df, use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()
