
from __future__ import annotations

import base64
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
    Image as RLImage, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
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


# Packaging guideline constants
# Source: Kawneer A-Buck packaging model / Veritiv drawing 9646-001 (R0, 8/17/2026).

GUIDELINE_CENTER_WALL_HEIGHT_IN = 56.0   # A-Buck support wall height
GUIDELINE_HEIGHT_ALLOWANCE_IN   = 18.0   # content may extend this far above the wall (74" total)
MAX_UPRIGHT_HEIGHT_IN           = 70.0   # HARD CAP applied by this tool, held below the 74" guideline

BAND_COLOR = '#2e8b57'   # green, matching the band lines on the guideline drawing
BAND_ALPHA = 0.45        # 0 = invisible, 1 = solid; applies to both the PDF images and the interactive preview
BAND_COLOR_RGBA = f'rgba(46,139,87,{BAND_ALPHA})'

BANDING_RULE_TEXT = (
    'Vertical banding: at least 2 vertical bands on every package. Packages 72" to 96" long '
    'receive at least 3 vertical bands. Packages longer than 96" receive 5 vertical bands.'
)
TETHER_NOTE_TEXT = (
    'Each window must be individually tethered to the top of the center wall with poly corded '
    'strapping run through the strapping of each window, so windows cannot fall out once the '
    'outer bands are removed.'
)
HEIGHT_RULE_TEXT = (
    f'Upright height: content is limited to {MAX_UPRIGHT_HEIGHT_IN:.0f}" (guideline allows '
    f'{GUIDELINE_HEIGHT_ALLOWANCE_IN:.0f}" above the {GUIDELINE_CENTER_WALL_HEIGHT_IN:.0f}" '
    f'center wall; this tool holds a {MAX_UPRIGHT_HEIGHT_IN:.0f}" cap).'
)


def required_vertical_bands(package_length: float) -> int:
    """Minimum vertical band count for a package of the given length (inches).
    Implements the A-Buck banding rule: 2 bands under 72", 3 bands from 72" to 96",
    5 bands above 96"."""
    if package_length > 96.0 + 1e-9:
        return 5
    if package_length >= 72.0 - 1e-9:
        return 3
    return 2


def vertical_band_positions(package_length: float, band_count: int) -> List[float]:
    """Evenly spaced X positions (inches from the pallet end) for vertical bands."""
    if band_count <= 0 or package_length <= 0:
        return []
    return [package_length * (i + 1) / (band_count + 1) for i in range(band_count)]


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
    max_length: float
    usable_space_per_side: float
    pallet_cost: float = 0.0
    max_height: float = MAX_UPRIGHT_HEIGHT_IN   # legacy field kept so older config files still load; the global hard cap is what is enforced


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


def _packaging_requirements_flowables() -> list:
    """Packaging requirement notes (height cap, banding rule, tether note) for the PDF."""
    styles = getSampleStyleSheet()
    body = ParagraphStyle('req', parent=styles['Normal'], fontSize=9, leading=12)
    return [
        _heading('Packaging Requirements', level=2),
        Paragraph(f'&bull; {HEIGHT_RULE_TEXT}', body),
        Paragraph(f'&bull; {BANDING_RULE_TEXT}', body),
        Paragraph(f'&bull; <b>Tether note:</b> {TETHER_NOTE_TEXT}', body),
        Spacer(1, 0.15 * inch),
    ]


def _assembly_drawing_flowables() -> list:
    """Closing page: A-Buck pallet assembly drawing (Veritiv dwg 9646-001) embedded in the app."""
    styles = getSampleStyleSheet()
    cap = ParagraphStyle('cap', parent=styles['Normal'], fontSize=8, leading=10)
    img_buf = io.BytesIO(base64.b64decode(A_BUCK_ASSEMBLY_PNG_B64))
    # Native image is 1418 x 1080 px; fit inside the 10" x 7.5" printable area.
    max_w, max_h = 10.0 * inch, 6.5 * inch
    aspect = 1418.0 / 1080.0
    w = max_w
    h = w / aspect
    if h > max_h:
        h = max_h
        w = h * aspect
    return [
        PageBreak(),
        KeepTogether([
            _heading('Pallet Assembly Drawing', level=1),
            Paragraph('A-Buck 96x46x56 assembly, Veritiv / 2K Wholesales drawing 9646-001 (R0, 8/17/2026). '
                      'Reference only. See the full drawing set for the base and support wall details.', cap),
            Spacer(1, 0.08 * inch),
            RLImage(img_buf, width=w, height=h),
        ]),
    ]


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
    summary_data['Field'].append('Vertical Bands Required')
    summary_data['Value'].append(str(required_vertical_bands(optimizer.pallet_by_id(best.pallet_id).base_length)))
    summary_df = pd.DataFrame(summary_data)
    story.append(_df_to_rl_table(summary_df, col_widths=[2.2 * inch, 4.5 * inch]))
    story.append(Spacer(1, 0.2 * inch))

    # --- Packaging requirements ---
    story.extend(_packaging_requirements_flowables())

    # --- All pallets comparison ---
    all_pallets_heading = _heading('All Pallets Evaluated', level=2)
    all_rows = []
    for r in results:
        all_rows.append({
            'Pallet': r.pallet_id,
            'Feasible': str(r.feasible),
            'Orientation': r.chosen_orientation or '-',
            'Pallets Needed': r.pallets_needed,
            'Units / Pallet': r.max_units_per_pallet,
            'Vertical Bands': required_vertical_bands(optimizer.pallet_by_id(r.pallet_id).base_length),
            'Cost Each': f'${r.pallet_cost_each:,.2f}' if r.pallet_cost_each else '-',
            'Total Cost': f'${r.estimated_total_cost:,.2f}' if r.estimated_total_cost else '-',
            'Preview Util %': f'{round((r.preview_utilization or 0.0) * 100, 2)}%' if r.preview_utilization is not None else '-',
        })
    story.append(KeepTogether([all_pallets_heading, _df_to_rl_table(pd.DataFrame(all_rows))]))
    story.append(PageBreak())

    # --- Pallet layout images ---
    chosen_pallet = optimizer.pallet_by_id(best.pallet_id)
    pallet_count = best.pallets_needed or 1
    story.append(_heading('Pallet Layout Previews (Top View)', level=1))
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

    # --- Assembly drawing (last page) ---
    story.extend(_assembly_drawing_flowables())

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
    story.append(Spacer(1, 0.2 * inch))

    # --- Packaging requirements ---
    story.extend(_packaging_requirements_flowables())

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
                         'Vertical Bands': required_vertical_bands(optimizer.pallet_by_id(pid).base_length),
                         'Cost Each': f'${cost_each:,.2f}',
                         'Total Cost': f'${cnt * cost_each:,.2f}'})
    story.append(_df_to_rl_table(pd.DataFrame(mix_rows)))
    story.append(Spacer(1, 0.2 * inch))

    # --- Per-pallet detail ---
    per_pallet_heading = _heading('Per-Pallet Detail', level=2)
    per_pallet_df = pd.DataFrame([{
        'Pallet #': load.pallet_number, 'Pallet Type': load.pallet_id,
        'Units': load.units_on_pallet,
        'Top Units': load.units_top_side, 'Bottom Units': load.units_bottom_side,
        'Utilization %': f'{round(load.preview_utilization * 100, 2)}%',
        'Balance': load.balance_status,
        'Vertical Bands': required_vertical_bands(optimizer.pallet_by_id(load.pallet_id).base_length),
        'Cost Each': f'${load.pallet_cost_each:,.2f}',
    } for load in result.pallet_loads])
    story.append(KeepTogether([per_pallet_heading, _df_to_rl_table(per_pallet_df)]))
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

    # --- Assembly drawing (last page) ---
    story.extend(_assembly_drawing_flowables())

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
    for load in result.pallet_loads:
        side_rows.append({'Pallet #': load.pallet_number, 'Pallet Type': load.pallet_id,
                          'Top Side Units': load.units_top_side, 'Bottom Side Units': load.units_bottom_side,
                          'Difference': abs(load.units_top_side - load.units_bottom_side),
                          'Status': load.balance_status})
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
            height_limit = MAX_UPRIGHT_HEIGHT_IN
            if upright_height > height_limit:
                reasons.append(f'{orientation_name}: upright height {upright_height:.3f} exceeds height limit {height_limit:.3f} (hard cap {MAX_UPRIGHT_HEIGHT_IN:.0f}")')
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
                    if upright_height > MAX_UPRIGHT_HEIGHT_IN:
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

    @staticmethod
    def build_balance_warning(units_top: int, units_bottom: int) -> Tuple[str, float]:
        """Balance is judged on total unit quantity per side only (top vs bottom),
        regardless of which configurations make up each side."""
        diff = abs(int(units_top) - int(units_bottom))
        if diff == 0:
            return 'Balanced', 0.0
        if diff <= 2:
            return 'Slight imbalance', float(diff)
        return 'High imbalance risk', float(diff)

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


    def candidate_rows_for_side(self, side, other_side, remaining, options_by_config, pallet, allow_singletons: bool = False):
        candidates = []
    
        
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
                # In ordered (by-row) mode this guard is relaxed: production order
                # forces odd leftovers to be placed even as singletons.
                min_count = 1
                if not allow_singletons and side.rows and row_slot_max >= 2 and any_source_has_two(opt):
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

    def build_single_pallet_candidate_ordered(self, pallet: Pallet, remaining: Dict[int, int],
                                              config_map: Dict[int, ConfigurationItem],
                                              options_by_config: Dict[int, List[OrientationOption]],
                                              order_index: Dict[int, int]) -> Optional[JobPalletLoad]:
        """
        Strict-order variant of build_single_pallet_candidate.

        Only the CURRENT earliest remaining config (by list order) may be
        placed at each step. When it is exhausted, the next config in order
        becomes current and may stack onto the same pallet's leftover space.
        If the current config fits neither side (taper/depth block it), the
        pallet is closed — the builder never skips ahead to a later config,
        so production order is never violated. The pallet's two sides carry
        independent tapers, so a next row wider than side A's outermost can
        still legitimately seed or continue side B.
        """
        top = SideState('top', pallet.max_depth_per_side, {cid: 0 for cid in remaining}, [])
        bottom = SideState('bottom', pallet.max_depth_per_side, {cid: 0 for cid in remaining}, [])
        local_remaining = dict(remaining)

        def current_earliest() -> Optional[int]:
            # Earliest is judged against ALL remaining configs in global order.
            # If the globally-next config cannot go on THIS pallet type, the
            # pallet must close rather than skip ahead to a later config —
            # otherwise production order would be violated across pallets.
            elig = [cid for cid, q in local_remaining.items() if q > 0]
            if not elig:
                return None
            g = min(elig, key=lambda cid: order_index.get(cid, 10**9))
            if not options_by_config.get(g):
                return None  # next-in-order config infeasible here -> close pallet
            return g

        def best_for_side(side: SideState, other_side: SideState, only_cid: int):
            cands = self.candidate_rows_for_side(side, other_side, local_remaining,
                                                 options_by_config, pallet, allow_singletons=True)
            best = None
            best_score = None
            for cid, opt, row_count in cands:
                if cid != only_cid:
                    continue
                effective_count = min(row_count, local_remaining.get(cid, 0))
                if effective_count <= 0:
                    continue
                units_side = sum(side.counts.values())
                units_other = sum(other_side.counts.values())
                imbalance_after = abs((units_side + effective_count) - units_other)
                score = (effective_count * 1000.0) + (opt.base_side * 500.0) - (1200.0 * imbalance_after)
                if best_score is None or score > best_score:
                    best_score = score
                    best = (cid, opt, effective_count, score)
            return best

        while True:
            cid_now = current_earliest()
            if cid_now is None:
                break
            top_best = best_for_side(top, bottom, cid_now)
            bottom_best = best_for_side(bottom, top, cid_now)
            if top_best is None and bottom_best is None:
                # Current config fits neither side of this pallet — close it.
                # Never skip ahead to a later config.
                break
            if top_best is not None and bottom_best is not None:
                units_top = sum(top.counts.values())
                units_bottom = sum(bottom.counts.values())
                diff_if_top = abs((units_top + top_best[2]) - units_bottom)
                diff_if_bottom = abs(units_top - (units_bottom + bottom_best[2]))
                if diff_if_top < diff_if_bottom:
                    chosen_side = 'top'
                elif diff_if_bottom < diff_if_top:
                    chosen_side = 'bottom'
                else:
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
                top.depth_left -= opt.effective_depth
                top.counts[cid] += row_count
            else:
                bottom.rows.append((cid, opt, row_count))
                bottom.depth_left -= opt.effective_depth
                bottom.counts[cid] += row_count
            local_remaining[cid] -= row_count

        units_on_pallet = sum(top.counts.values()) + sum(bottom.counts.values())
        if units_on_pallet <= 0:
            return None
        return self._assemble_pallet_load(pallet, top, bottom, config_map)

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
        return self._assemble_pallet_load(pallet, top, bottom, config_map)

    def _assemble_pallet_load(self, pallet: Pallet, top: SideState, bottom: SideState,
                              config_map: Dict[int, ConfigurationItem]) -> Optional[JobPalletLoad]:
        """Build the JobPalletLoad (placements, utilization, balance) from filled
        top/bottom SideStates. Shared by both the free and ordered builders."""
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
        balance_status, balance_penalty = self.build_balance_warning(sum(top.counts.values()), sum(bottom.counts.values()))
        return JobPalletLoad(0, pallet.pallet_id, placements, units_on_pallet, sum(top.counts.values()), sum(bottom.counts.values()), preview_utilization, f'Candidate pallet {pallet.pallet_id} removes {units_on_pallet} windows from remaining job inventory. Each side independently satisfies the non-increasing taper rule (row widths do not increase as rows move outward from the center).', config_side_counts, balance_status, balance_penalty, pallet.pallet_cost)

    def candidate_score(self, load: JobPalletLoad, difficulty: Dict[int, float]) -> float:
        removed_difficulty = 0.0
        removed_units = 0
        for cid, (top_count, bottom_count) in load.config_side_counts.items():
            removed = top_count + bottom_count
            removed_units += removed
            removed_difficulty += removed * difficulty.get(cid, 0.0)
        used_area = sum(p.length * p.depth for p in load.placements)
        # Favor stable pallets by penalizing top/bottom imbalance (balance_penalty is |top units - bottom units|).
        return removed_difficulty * 80.0 + used_area + removed_units * 1000.0 - (getattr(load, 'balance_penalty', 0.0) * 2500.0)

    def build_job_plan(self, configs: List[ConfigurationItem], preserve_order: bool = False) -> JobPlanResult:
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

        if preserve_order:
            # ── Ordered (by-row) fill with stacking ─────────────────────────
            # Windows are consumed strictly in configuration-list order: Row 1
            # is placed first, then Row 2, and so on. Crucially, a later row is
            # allowed to STACK onto the same pallet as an earlier row when there
            # is leftover space and the structural rules still pass — Row 2 tops
            # off Row 1's pallet before a new pallet is opened. A new pallet is
            # only started when the current in-order frontier cannot add more.
            #
            # Implementation: at each step the "eligible" configs are the still-
            # remaining ones taken in list order starting at the earliest that
            # still has units. We build a pallet that fills from that ordered
            # frontier. The candidate builder is order-biased so it always
            # exhausts the earliest remaining config before pulling from a later
            # one, which preserves production order while allowing stacking.
            order_index = {c.config_id: i for i, c in enumerate(configs)}
            while sum(remaining.values()) > 0:
                # Ordered frontier: every config that still has units. The builder
                # only ever places the current earliest, advancing in order.
                frontier = {cid: q for cid, q in remaining.items() if q > 0}
                global_earliest = min(frontier, key=lambda c: order_index[c])
                best_load = None
                for pallet in self.pallets:
                    # A pallet type that cannot hold the global earliest config
                    # cannot be opened this round — doing so would palletize a
                    # later row before an earlier one (order violation).
                    if not all_feasible[pallet.pallet_id].get(global_earliest):
                        continue
                    load = self.build_single_pallet_candidate_ordered(
                        pallet, frontier, config_map,
                        all_feasible[pallet.pallet_id], order_index)
                    if load is None or load.units_on_pallet <= 0:
                        continue
                    earliest_removed = sum(load.config_side_counts.get(global_earliest, (0, 0)))
                    key = (earliest_removed, load.units_on_pallet, load.preview_utilization or 0.0, -load.pallet_cost_each)
                    if best_load is None or key > best_load[0]:
                        best_load = (key, load)
                if best_load is None:
                    break
                chosen = best_load[1]
                chosen.pallet_number = pallet_number
                pallet_loads.append(chosen)
                pallet_mix_summary[chosen.pallet_id] = pallet_mix_summary.get(chosen.pallet_id, 0) + 1
                total_balance_penalty += chosen.balance_penalty
                total_cost += chosen.pallet_cost_each
                total_used_area += sum(p.length * p.depth for p in chosen.placements)
                for pcid, (top_count, bottom_count) in chosen.config_side_counts.items():
                    remaining[pcid] -= (top_count + bottom_count)
                pallet_number += 1
            total_units = sum(c.qty for c in configs)
            total_usable_area = sum(self.usable_area_for_pallet(self.pallet_by_id(load.pallet_id)) for load in pallet_loads)
            pallets_needed = len(pallet_loads)
            overall_utilization = (total_used_area / total_usable_area) if total_usable_area else 0.0
            avg_utilization = (sum(load.preview_utilization for load in pallet_loads) / pallets_needed) if pallets_needed else 0.0
            explanation = (f'Ordered (by-row) plan: windows were palletized in configuration-list order — '
                           f'row 1 first, then row 2, and so on — with later rows stacking onto an earlier '
                           f'row\'s pallet when space and the structural rules allow. Total pallets needed = '
                           f'{pallets_needed}. Overall utilization = {overall_utilization:.2%}. Average pallet '
                           f'utilization = {avg_utilization:.2%}. Estimated total pallet cost = ${total_cost:,.2f}. '
                           f'Each side independently satisfies the non-increasing pyramid support rule.')
            if excluded:
                excl_detail = '; '.join(f'{lbl} ({rsn})' for lbl, rsn in excluded)
                explanation += (f' NOTE: {len(excluded)} configuration(s) were excluded because they '
                                f'fit no selected pallet and must be handled outside the tool: {excl_detail}.')
            return JobPlanResult(True, pallets_needed, total_units, overall_utilization, avg_utilization,
                                 pallet_loads, explanation, total_balance_penalty, pallet_mix_summary, total_cost, excluded)

        # ── Free optimizer (default) ────────────────────────────────────────
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
    """Render a pallet top-view layout to PNG bytes using matplotlib.

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

    # Vertical bands (A-Buck banding rule), drawn across the full pallet width
    band_count = required_vertical_bands(pallet.base_length)
    for bx in vertical_band_positions(pallet.base_length, band_count):
        ax.plot([bx, bx], [0, pallet.base_width], color=BAND_COLOR, linewidth=2.2, alpha=BAND_ALPHA, zorder=5)
        ax.text(bx, pallet.base_width - 0.6, 'BAND', ha='center', va='top', fontsize=7,
                color=BAND_COLOR, alpha=BAND_ALPHA, fontweight='bold', zorder=6)

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
    fig.add_annotation(x=pallet.base_length / 2, y=pallet.max_depth_per_side + pallet.center_depth / 2, text='Center Frame', showarrow=False, font=dict(size=10, color='black'))
    for p in placements:
        cx, cy = p.x + p.length / 2, p.y + p.depth / 2
        fig.add_shape(type='rect', x0=p.x, y0=p.y, x1=p.x + p.length, y1=p.y + p.depth,
                      line=dict(color=get_config_color(p.config_id), width=2),
                      fillcolor='rgba(255,255,255,0)')
        fig.add_trace(go.Scatter(x=[cx], y=[cy], mode='markers',
                      marker=dict(size=8, color='rgba(0,0,0,0)'),
                      hovertemplate=(
                          f'Config: {p.config_label}<br>'
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
                size=14,  # bump slightly for two lines
                color='rgba(0,0,0,0.7)'
            ),
            textposition='middle center',
            hoverinfo='skip',
            showlegend=False
        ))
    # Vertical bands (A-Buck banding rule), drawn across the full pallet width
    band_count = required_vertical_bands(pallet.base_length)
    for i, bx in enumerate(vertical_band_positions(pallet.base_length, band_count), start=1):
        fig.add_shape(type='line', x0=bx, y0=0, x1=bx, y1=pallet.base_width,
                      line=dict(color=BAND_COLOR_RGBA, width=3))
        fig.add_annotation(x=bx, y=pallet.base_width, text=f'Band {i}', showarrow=False,
                           yanchor='bottom', font=dict(size=11, color=BAND_COLOR_RGBA))
    fig.update_xaxes(title='Pallet Length (inches)', range=[0, pallet.base_length], showgrid=True, zeroline=False, scaleanchor='y', scaleratio=1)
    fig.update_yaxes(title='Pallet Width / Side Depth (inches)', range=[0, pallet.base_width], showgrid=True, zeroline=False)
    fig.update_layout(title=title, height=560, margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='white', hovermode='closest')
    return fig


def render_packaging_notes(pallet: Pallet) -> None:
    """Banding count + tether reminder shown under the interactive preview."""
    bands = required_vertical_bands(pallet.base_length)
    st.info(
        f'**Banding:** {bands} vertical bands required for the {pallet.base_length:.0f}" package '
        f'(green lines). {BANDING_RULE_TEXT}\n\n'
        f'**Tether:** {TETHER_NOTE_TEXT}\n\n'
        f'**Height:** {HEIGHT_RULE_TEXT}'
    )

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
def run_job_plan_cached(config: dict, depth_csv_path: str, allowed: Tuple[str, ...], config_rows: Tuple[Tuple, ...], preserve_order: bool = False) -> JobPlanResult:
    """Cache the mixed-job optimization so it only recomputes when an input
    that affects the plan actually changes (pallet config contents, allowed
    pallet list, the configuration rows, or the ordered/optimized toggle) —
    not on every widget interaction."""
    depth_df = load_depths(depth_csv_path)
    optimizer = MixedJobOptimizer(config, depth_df, allowed_pallet_ids=list(allowed))
    configs = [ConfigurationItem(*row) for row in config_rows]
    return optimizer.build_job_plan(configs, preserve_order=preserve_order)


def balance_table_for_load(load: JobPalletLoad, configs: List[ConfigurationItem]) -> pd.DataFrame:
    """Side balance is based on total unit quantity per side only."""
    diff = abs(load.units_top_side - load.units_bottom_side)
    return pd.DataFrame([{
        'Pallet #': load.pallet_number,
        'Top Side Units': load.units_top_side,
        'Bottom Side Units': load.units_bottom_side,
        'Difference': diff,
        'Status': load.balance_status,
    }])


def pallet_mix_table(pallet_mix: Dict[str, int], optimizer: BaseOptimizer) -> pd.DataFrame:
    rows = []
    for pid, cnt in sorted(pallet_mix.items()):
        p = optimizer.pallet_by_id(pid)
        rows.append({'Pallet Type': pid, 'Count': cnt, 'Cost Each': p.pallet_cost, 'Total Cost': p.pallet_cost * cnt})
    return pd.DataFrame(rows)


# Pallet settings editor

PALLET_NUMERIC_FIELDS = ['base_length', 'base_width', 'center_height', 'center_depth',
                         'max_depth_per_side', 'max_length',
                         'usable_space_per_side', 'pallet_cost']


def render_pallet_settings(config: dict) -> None:
    st.subheader('Pallet Settings')
    saved_msg = st.session_state.pop('pallet_settings_saved_msg_v45', None)
    if saved_msg:
        st.success(saved_msg)
    st.caption(
        'Edit pallet sizes and costs directly in the table. Use the empty row at the bottom to add a new '
        'pallet size; select a row and use the trash icon to remove one. Press Save to write the changes '
        f'to {DEFAULT_CONFIG_PATH.name} — they take effect immediately in both optimizer modes. '
        f'Max upright height is fixed at {MAX_UPRIGHT_HEIGHT_IN:.0f}" for every pallet and is not editable here.'
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
    st.set_page_config(page_title='Pallet Optimizer V4.8', layout='wide')
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
        st.sidebar.caption(f'Upright height hard cap: {MAX_UPRIGHT_HEIGHT_IN:.0f}" (A-Buck guideline).')
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
                st.plotly_chart(build_plotly_preview(chosen_pallet, selected_preview.placements or [], f'Interactive Top View — {chosen_pallet.pallet_id} — Pallet #{selected_pallet_num}'), use_container_width=True)
                render_packaging_notes(chosen_pallet)
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
        setup_c1, setup_c2, setup_c3, setup_c4 = st.columns([1.2, 0.8, 1.4, 0.8])
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
        with setup_c4:
            st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
            if st.button('Reset', use_container_width=True,
                         disabled=not bool(st.session_state['job_items_v37']),
                         help='Clears every configuration currently loaded into the Pallet Optimizer.'):
                st.session_state['confirm_reset_v46'] = True
                st.rerun()
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

        # ── Reset confirmation (triggered by the Reset button next to Load Job) ──
        if st.session_state.get('confirm_reset_v46', False):
            _reset_items = st.session_state['job_items_v37']
            st.warning(
                f'This will clear all {len(_reset_items)} configuration(s) loaded into the Pallet Optimizer. '
                'This cannot be undone. Use Save Job (.json) first if you want to keep this job.'
            )
            reset_c1, reset_c2, _reset_spacer = st.columns([1, 1, 3])
            if reset_c1.button('Yes, clear all', type='primary', use_container_width=True):
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

        preserve_order = st.checkbox(
            'Palletize in row order',
            value=st.session_state.get('preserve_order_v48', False),
            key='preserve_order_v48',
            help='When checked, windows are loaded onto pallets strictly in the order of the configuration list. '
                 'Each row\'s full quantity is palletized before the next row begins, and each pallet holds a single '
                 'configuration. When unchecked, the optimizer freely mixes configurations to minimize pallet count.'
        )

        configs = items_to_configs(st.session_state['job_items_v37'])
        best_job = run_job_plan_cached(config, str(DEFAULT_DEPTH_CSV_PATH), tuple(sorted(allowable_pallets)), configs_to_rows(configs), preserve_order) if configs else None

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
                c3.metric('Balance Warning Total', f'{(best_job.total_balance_penalty or 0.0):.1f}', help='Sum over all pallets of |top side units - bottom side units|. Balance considers unit quantity per side only.')
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
                st.plotly_chart(build_plotly_preview(chosen_pallet, selected_load.placements, f'Interactive Top View — {chosen_pallet.pallet_id} — Job Pallet #{selected_load.pallet_number}'), use_container_width=True)
                render_packaging_notes(chosen_pallet)
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


# A-Buck pallet assembly drawing (Veritiv / 2K Wholesales dwg 9646-001, R0 8/17/2026),
# rasterized from the packaging model PDF and embedded as PNG so the report works on
# Streamlit Cloud and in the PyInstaller build without any extra data file.
A_BUCK_ASSEMBLY_PNG_B64 = (
    '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'
)


if __name__ == '__main__':
    main()
