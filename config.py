"""
Shared configuration, data loading and styling for all analysis notebooks.
Import with:  from config import *
"""

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# ── Root paths (works whether Jupyter is started from root or notebooks/) ──────
ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
OUTPUT_DIR  = ROOT / 'output'
FIGURES_DIR = ROOT / 'figures'

# ── Styling ───────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', font_scale=1.15)
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 150,
                     'savefig.bbox': 'tight', 'font.family': 'sans-serif'})

AREA_ORDER  = ['Zefkseidos', 'Shaded', 'Waterfront']
AREA_LABELS = {
    'Zefkseidos': 'Area A\n(Dense Canyon)',
    'Shaded':     'Area B\n(Shaded / Vegetation)',
    'Waterfront': 'Area C\n(Open Parking)',
}
AREA_COLORS = {'Zefkseidos': '#D94F3D', 'Shaded': '#3DAD6E', 'Waterfront': '#3A7EC6'}

DATE_ORDER  = ['15Feb', '31Mar', '15Aug', '15Nov']
DATE_LABELS = {
    '15Feb': 'Feb 15 (Winter)',
    '31Mar': 'Mar 31 (Spring)',
    '15Aug': 'Aug 15 (Summer)',
    '15Nov': 'Nov 15 (Autumn)',
}
DATE_COLORS = {
    '15Feb': '#3A7EC6',
    '31Mar': '#3DAD6E',
    '15Aug': '#D94F3D',
    '15Nov': '#E8A838',
}
DATE_MARKERS = {'15Feb': 'o', '31Mar': 's', '15Aug': '^', '15Nov': 'D'}

# ── ENVI-met column names ─────────────────────────────────────────────────────
AIR_TEMP_COL   = 'Potential Air Temperature (°C)'
BUILDING_COL   = 'Objects ( )'

# ── Data loading ──────────────────────────────────────────────────────────────
def load_csvs(mtype, verbose=True):
    """Load all parquet files for a given measurement type."""
    files = sorted(OUTPUT_DIR.glob(f'*_{mtype}.parquet'))
    if not files:
        if verbose: print(f'  [!] No {mtype} parquet files in {OUTPUT_DIR}')
        return pd.DataFrame()
    frames = []
    for p in files:
        df = pd.read_parquet(p)
        # ensure hour is available (int); time may be NaT for old extractions
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df['time'], errors='coerce').dt.hour
        frames.append(df)
        if verbose: print(f'  {p.name}  →  {len(df):,} rows')
    return pd.concat(frames, ignore_index=True)


def detect_surface_cols(srf):
    """Return (surface_temp_col, surface_albedo_col) by scanning column names.

    Handles ENVI-met naming conventions:
      temperature → 'T Surface (°C)'  (contains '°c' not 'temperature')
      albedo      → 'Surface Albedo ()'
    """
    meta = {'area','date','measurement_type','time','hour','i','j','k','x_m','y_m','z_m'}
    cols = [c for c in srf.columns if c not in meta]
    lo   = [c.lower() for c in cols]

    def find(checks):
        for c, l in zip(cols, lo):
            if all(k in l for k in checks):
                return c
        return None

    t = find(['surface', '°c']) or find(['surface', 'temp']) or find(['temperature'])
    a = find(['albedo'])
    return t, a


def savefig(fig, subfolder, filename):
    path = FIGURES_DIR / subfolder / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f'  Saved → figures/{subfolder}/{filename}')
    plt.show()


print(f'Config loaded.  ROOT={ROOT}')
print(f'Output dir : {OUTPUT_DIR}')
print(f'Figures dir: {FIGURES_DIR}')
