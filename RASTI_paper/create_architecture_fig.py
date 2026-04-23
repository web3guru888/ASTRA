"""
ASTRA System Architecture Diagram
Creates fig00_architecture.pdf for the RASTI paper.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Figure dimensions: 7 x 5 inches (single-column width for journal)
fig, ax = plt.subplots(figsize=(8.5, 5.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# ASTRA brand-consistent colors (colorblind-safe, B&W-readable)
BOX_COLOR_INPUT = '#E8F4FD'       # Light blue - input
BOX_COLOR_CORE = '#EEF4EB'        # Light green - core engines
BOX_COLOR_KNOW = '#FDF5E8'        # Light amber - knowledge
BOX_COLOR_OUT = '#F5EEF8'         # Light purple - output
BOX_EDGE = '#2c3e50'              # Dark edge
ARROW_COLOR = '#34495e'           # Arrow color

def draw_box(ax, x, y, w, h, label, sublabel, color, fontsize=9, subfontsize=7.5):
    """Draw a rounded rectangle box with label."""
    box = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=BOX_EDGE, linewidth=1.2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + 0.12, label,
            ha='center', va='center', fontsize=fontsize, fontweight='bold',
            color='#1a1a2e', wrap=True)
    ax.text(x + w/2, y + h/2 - 0.18, sublabel,
            ha='center', va='center', fontsize=subfontsize, color='#4a4a6a',
            style='italic', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label='', color=ARROW_COLOR, curved=False):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.3,
                                connectionstyle='arc3,rad=0.0' if not curved else 'arc3,rad=0.2'))
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.08, label, ha='center', va='bottom', fontsize=6.5,
                color='#555577')

# ─── Row 1: Input and Orchestration ───────────────────────────────────────────
# Box: User Query / Data Input
draw_box(ax, 0.3, 5.5, 2.2, 1.1,
         'Data Input',
         'FITS, CSV, catalogs\nimages, spectra',
         BOX_COLOR_INPUT, fontsize=9, subfontsize=7.5)

# Box: Multi-Module Orchestration (centre-top)
draw_box(ax, 3.9, 5.5, 2.2, 1.1,
         'Orchestration',
         'Meta-cognitive layer\nmodule selection & synthesis',
         '#FBF0F0', fontsize=9, subfontsize=7.5)

# Box: Natural Language Query
draw_box(ax, 7.5, 5.5, 2.2, 1.1,
         'Query Interface',
         'Natural language queries\nautonomous task routing',
         BOX_COLOR_INPUT, fontsize=9, subfontsize=7.5)

# ─── Row 2: Core Engines ──────────────────────────────────────────────────────
draw_box(ax, 0.3, 3.5, 2.0, 1.5,
         'Physics Engine',
         'Dimensional analysis\n(Buckingham π)\nConservation laws\nUnits checking',
         BOX_COLOR_CORE, fontsize=8.5, subfontsize=7)

draw_box(ax, 2.6, 3.5, 2.0, 1.5,
         'Causal Module',
         'PC algorithm\ndo-calculus\nV-structure detection\nConditional indep.',
         BOX_COLOR_CORE, fontsize=8.5, subfontsize=7)

draw_box(ax, 4.9, 3.5, 2.0, 1.5,
         'Bayesian Engine',
         'Evidence computation\nBayes factors\nPSIS-LOO-CV\nMCMC / HME',
         BOX_COLOR_CORE, fontsize=8.5, subfontsize=7)

draw_box(ax, 7.2, 3.5, 2.5, 1.5,
         'Domain Knowledge',
         'MORK ontology\n75 specialist modules\nKnowledge graph\nVector memory',
         BOX_COLOR_KNOW, fontsize=8.5, subfontsize=7)

# ─── Row 3: Data Pipeline ────────────────────────────────────────────────────
draw_box(ax, 0.3, 1.5, 3.6, 1.3,
         'Data Processing Pipeline',
         'Ingestion · Validation · Cleaning · Normalisation\nUncertainty propagation · Multi-wavelength fusion',
         BOX_COLOR_INPUT, fontsize=9, subfontsize=7.5)

draw_box(ax, 4.2, 1.5, 5.5, 1.3,
         'HPC Simulation Manager',
         'Athena++ MHD campaign design · Remote job management\n600+ parallel simulations · Results integration',
         '#F0EEF8', fontsize=9, subfontsize=7.5)

# ─── Row 4: Output ───────────────────────────────────────────────────────────
draw_box(ax, 2.5, 0.1, 5.0, 1.1,
         'Scientific Output',
         'Physical interpretation · Uncertainty quantification · Testable hypotheses · Reproducible workflows',
         BOX_COLOR_OUT, fontsize=9, subfontsize=7.5)

# ─── Arrows ──────────────────────────────────────────────────────────────────
# Data Input → Orchestration
draw_arrow(ax, 2.5, 6.05, 3.9, 6.05)
# Query Interface → Orchestration
draw_arrow(ax, 7.5, 6.05, 6.1, 6.05)

# Orchestration → Core Engines
draw_arrow(ax, 5.0, 5.5, 4.0, 5.0)
draw_arrow(ax, 5.0, 5.5, 3.6, 5.0)
draw_arrow(ax, 5.0, 5.5, 5.9, 5.0)
draw_arrow(ax, 5.0, 5.5, 7.9, 5.0)

# Data Input → Data Pipeline
draw_arrow(ax, 1.4, 5.5, 2.1, 2.8)

# HPC Sim Manager → Output
draw_arrow(ax, 6.95, 1.5, 6.5, 1.2)

# Data Pipeline → Core Engines
draw_arrow(ax, 2.1, 3.5, 1.3, 3.0)
draw_arrow(ax, 2.1, 3.5, 3.6, 3.0)

# Core Engines → Output
draw_arrow(ax, 1.3, 3.5, 4.0, 1.2)
draw_arrow(ax, 3.6, 3.5, 5.0, 1.2)
draw_arrow(ax, 5.9, 3.5, 5.5, 1.2)

# Title annotation
ax.text(5.0, 6.85, 'ASTRA System Architecture',
        ha='center', va='center', fontsize=12, fontweight='bold', color='#1a1a2e')

# Legend-style annotations for capability count
ax.text(0.3, 0.0, '16 integrated capabilities  ·  313K lines Python  ·  620+ modules',
        ha='left', va='bottom', fontsize=7.5, color='#666688', style='italic')

plt.tight_layout(pad=0.3)
plt.savefig('figures/fig00_architecture.pdf', bbox_inches='tight', dpi=200,
            facecolor='white', edgecolor='none')
plt.savefig('figures/fig00_architecture.png', bbox_inches='tight', dpi=150,
            facecolor='white', edgecolor='none')
print("Architecture diagram saved: figures/fig00_architecture.pdf")
