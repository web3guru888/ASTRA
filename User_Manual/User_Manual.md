# ASTRA User Manual
## Autonomous Scientific & Technological Research Agent

**Version**: 8.0
**Date**: April 2026
**Authors**: Glenn J. White, Open University and Rutherford Appleton Laboratory, England
**Repository**: https://github.com/Tilanthi/ASTRA-dev

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Installation and Setup](#3-installation-and-setup)
   - 3.1 System Requirements
   - 3.2 Installation Methods
   - 3.3 Configuration
   - 3.4 Running ASTRA Live Backend ⭐ NEW
4. [Getting Started](#4-getting-started)
5. [Core Capabilities Overview](#5-core-capabilities-overview)
6. [V5.0 Discovery Enhancement System](#6-v50-discovery-enhancement-system)
7. [V7.0 Autonomous Research Scientist](#7-v70-autonomous-research-scientist)
8. [V8.0 Cognitive Architecture (Scientific AGI)](#8-v80-cognitive-architecture-scientific-agi) ⭐ NEW
9. [Use Case Examples](#9-use-case-examples)
10. [Advanced Features](#10-advanced-features)
11. [Domain Modules](#11-domain-modules)
12. [API Reference](#12-api-reference)
13. [Best Practices](#13-best-practices)
14. [Troubleshooting](#14-troubleshooting)
15. [Appendices](#15-appendices)

---

## 1. Introduction

### 1.1 What is ASTRA?

ASTRA (Autonomous Scientific & Technological Research Agent) is an integrated computational framework that combines numerical data analysis, causal reasoning, physical validation, and statistical inference to enable automated scientific discovery in astrophysics and related domains. Unlike traditional machine learning systems that detect patterns without understanding their physical meaning, or large language models that can explain concepts but cannot process numerical data, ASTRA integrates multiple analytical approaches to provide physically interpretable, validated scientific insights.

**Version 8.0** represents a paradigm shift with the introduction of **Scientific AGI capabilities** through a unified cognitive architecture that integrates knowledge graphs, neuro-symbolic reasoning, and meta-cognitive self-awareness.

### 1.2 Key Design Principles

**Physics-Aware Reasoning**: All discoveries are validated against fundamental physical principles including conservation laws, dimensional consistency, and established theoretical frameworks.

**Causal Understanding**: ASTRA distinguishes between correlation and causation using structural causal models, enabling identification of physical mechanisms rather than mere associations.

**Uncertainty Quantification**: Every result includes properly propagated uncertainties, confidence intervals, and statistical significance assessments.

**Reproducibility**: All analyses are fully documented and reproducible, with complete provenance tracking from raw data to final conclusions.

**Cognitive Integration**: V8.0 integrates perception, reasoning, memory, attention, and meta-cognition in a unified architecture approaching Scientific AGI.

### 1.3 Who Should Use This Manual?

This manual is written for expert users including:
- Research astronomers and astrophysicists
- Data scientists working with astronomical data
- Computational scientists requiring physics-aware analysis tools
- Graduate students and postdoctoral researchers in astrophysics
- AI researchers interested in scientific reasoning systems

Users should have familiarity with:
- Python programming
- Basic statistical concepts
- Fundamental astrophysical principles
- Command-line operation
- REST API usage (for live backend)

### 1.4 What's New in Version 8.0

**V8.0 Cognitive Architecture (Scientific AGI)**:
- Dynamic Knowledge Graph for semantic knowledge integration
- Neuro-Symbolic Engine unifying neural and symbolic reasoning
- Meta-Cognitive Architecture for self-awareness and self-improvement
- Cognitive Core Orchestrator for unified discovery pipeline
- State Persistence across restarts
- Enhanced Hypothesis Generation (3-5 per cycle)
- Live Dashboard with real-time status
- 15 new cognitive API endpoints

**Enhanced Capabilities**:
- Automatic discovery recording from statistical tests
- Cross-domain analogy discovery
- Knowledge gap identification
- Method performance tracking and selection
- Reflective self-improvement

---

## 2. System Architecture

### 2.1 Architectural Overview

ASTRA V8.0 implements a layered cognitive architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface Layer                    │
│  (Live Dashboard, REST API, Python API, CLI)               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Cognitive Core (V8.0)                     │
│  ┌──────────────┬──────────────┬──────────────┐            │
│  │   Knowledge  │ Neuro-       │  Meta-       │            │
│  │    Graph     │ Symbolic     │ Cognition    │            │
│  └──────────────┴──────────────┴──────────────┘            │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   Orchestration Layer                       │
│  OODA Discovery Cycle (Orient→Select→Investigate→Update)   │
└─────┬───────────┬───────────┬───────────┬───────────┬─────┘
      │           │           │           │           │
┌─────▼─────┐ ┌─▼──────┐ ┌─▼──────┐ ┌─▼──────┐ ┌─▼────────┐
│  Physics  │ │ Causal │ │Bayesian│ │ Data   │ │ Domain   │
│  Engine   │ │Reasoning│ │Inference│ │Registry│ │Knowledge│
└───────────┘ └────────┘ └────────┘ └────────┘ └──────────┘
      │           │           │           │           │
      └───────────┴───────────┴───────────┴───────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Foundation Layer                         │
│  Memory systems, I/O handling, numerical libraries         │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Core Components

#### 2.2.1 Physics Engine

The Physics Engine implements fundamental physical laws and constraints that govern all analyses.

#### 2.2.2 Causal Reasoning Module

The Causal Reasoning Module enables discovery and inference of causal relationships from observational data.

#### 2.2.3 Bayesian Inference Engine

The Bayesian Inference Engine provides rigorous model comparison and uncertainty quantification.

#### 2.2.4 Data Registry

The Data Registry manages 9 astronomical and socioeconomic data sources with automatic caching and validation.

#### 2.2.5 V8.0 Cognitive Components

**Dynamic Knowledge Graph**:
- Semantic brain integrating all discoveries
- Bayesian belief propagation
- Knowledge gap identification
- Cross-domain analogy discovery

**Neuro-Symbolic Engine**:
- Pattern discovery → symbolic formalization
- Explainable predictions
- Architecture search for problems

**Meta-Cognitive Architecture**:
- Self-awareness and reasoning trace monitoring
- Method performance tracking
- Error pattern detection
- Confidence calibration

---

## 3. Installation and Setup

### 3.1 System Requirements

**Minimum Requirements**:
- Python 3.8 or higher
- 8 GB RAM
- 2 GB free disk space
- Linux, macOS, or Windows with WSL2

**Recommended for Large Datasets**:
- Python 3.10 or higher
- 32 GB RAM
- 20 GB free disk space
- SSD storage for better I/O performance
- Multi-core processor (4+ cores)

**Optional Dependencies**:
- PyTorch for neural network components
- Jupyter for interactive analysis
- Docker for containerized deployment

### 3.2 Installation Methods

#### 3.2.1 Installation from GitHub

```bash
# Clone the repository
git clone https://github.com/Tilanthi/ASTRA-dev.git
cd ASTRA-dev

# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .
```

#### 3.2.2 Verification of Installation

```python
# Test installation
python -c "from astra_live_backend import create_engine; print('ASTRA Live installed successfully')"

# Check version
curl http://localhost:8787/api/status  # After starting server
```

### 3.3 Configuration

#### 3.3.1 Basic Configuration

ASTRA Live uses environment variables and sensible defaults:

```bash
# Optional: Set custom port
export ASTRA_PORT=8787

# Optional: Set data directory
export ASTRA_DATA_DIR=./astronomy_data

# Optional: Enable debug logging
export ASTRA_LOG_LEVEL=DEBUG
```

### 3.4 Running ASTRA Live Backend

**ASTRA Live** is the recommended mode for running ASTRA V8.0 with full cognitive capabilities.

#### 3.4.1 Starting the Server

```bash
# Navigate to ASTRA directory
cd /path/to/ASTRA-dev

# Start the live backend server
python3 -m astra_live_backend.server

# Server starts at http://localhost:8787
# API documentation at http://localhost:8787/docs
```

#### 3.4.2 Accessing the Dashboard

```bash
# Open in browser
open http://localhost:8787

# Or access via curl
curl http://localhost:8787/api/status
```

#### 3.4.3 Initializing Hypotheses

For fresh installations, initialize with baseline hypotheses:

```bash
# Run initialization script
python3 init_hypotheses.py

# This creates 23 baseline hypotheses:
# - 18 in TESTING phase
# - 5 in PROPOSED phase
```

#### 3.4.4 Running a Discovery Cycle

```bash
# Trigger one discovery cycle
curl -X POST http://localhost:8787/api/engine/cycle

# Check status
curl http://localhost:8787/api/status | jq '.engine'
```

#### 3.4.5 Stopping the Server

```bash
# Press Ctrl+C in terminal
# Or use pkill
pkill -f "python.*astra_live_backend.server"
```

---

## 4. Getting Started

### 4.1 Your First Analysis

#### Example 1: Query ASTRA Live API

```bash
# Check system status
curl http://localhost:8787/api/status

# Get hypotheses
curl http://localhost:8787/api/hypotheses | jq '.[] | {id, name, phase}'

# View cognitive architecture status
curl http://localhost:8787/api/cognitive/status
```

#### Example 2: Python API

```python
import requests
import json

API_BASE = "http://localhost:8787"

# Get system status
response = requests.get(f"{API_BASE}/api/status")
status = response.json()
print(f"Cycle: {status['engine']['cycle_count']}")
print(f"Phase: {status['engine']['current_phase']}")
print(f"Confidence: {status['engine']['system_confidence']:.3f}")

# Get hypotheses
response = requests.get(f"{API_BASE}/api/hypotheses")
hypotheses = response.json()
print(f"\nTotal hypotheses: {len(hypotheses)}")

for h in hypotheses[:5]:
    print(f"{h['id']}: {h['name']} ({h['phase']})")
```

### 4.2 Understanding ASTRA's Output

ASTRA Live provides structured JSON output:

**Status Response**:
```json
{
  "status": "running",
  "engine": {
    "running": true,
    "current_phase": "INVESTIGATE",
    "cycle_count": 217,
    "system_confidence": 0.81,
    "funnel": {
      "proposed": 5,
      "testing": 18,
      "validated": 0
    }
  }
}
```

**Hypothesis Response**:
```json
{
  "id": "H001",
  "name": "Pantheon+ SNe Ia Distance Modulus",
  "domain": "Astrophysics",
  "phase": "testing",
  "confidence": 0.55,
  "test_results": []
}
```

---

## 5. Core Capabilities Overview

ASTRA Live integrates 25+ analytical capabilities across the OODA discovery cycle.

### 5.1 OODA Discovery Cycle

ASTRA operates through a continuous OODA loop:

**OBSERVE**: Collect data from 9 sources
- Pantheon+ SNe Ia (1,701 supernovae)
- NASA Exoplanet Archive (2,839 planets)
- Gaia DR3 (4,984 stars)
- SDSS DR18 (2,000+ galaxies)
- LIGO GW (280 events)
- Planck CMB (2,507 data points)
- ZTF Transients (2,000 events)
- TESS Input Catalog
- SDSS Clusters

**ORIENT**: Generate and select hypotheses
- Discovery-guided hypothesis generation
- Cognitive discovery every 15 cycles
- Knowledge gap identification

**INVESTIGATE**: Test hypotheses with statistical methods
- KS test, χ² test, t-test
- Pearson correlation
- Granger causality
- Causal inference (PC/FCI algorithms)
- Bayesian model comparison

**UPDATE**: Learn and improve
- Update hypothesis confidence
- Generate new hypotheses
- Cognitive reflection
- State persistence every 50 cycles

### 5.2 Statistical Tests

| Test | Purpose | Output |
|------|---------|--------|
| Kolmogorov-Smirnov | Distribution comparison | p-value, D statistic |
| Chi-squared | Goodness of fit | p-value, χ² statistic |
| Pearson correlation | Linear correlation | r, p-value |
| Granger causality | Temporal causality | F-statistic, p-value |
| Bayesian model comparison | Model selection | Bayes factor, BIC |

### 5.3 Discovery Methods

| Method | Description | Data Required |
|--------|-------------|---------------|
| Scaling discovery | Find power-law relations | 2+ variables |
| Causal inference | Discover causal structures | 3+ variables |
| Bimodality detection | Find multi-modal distributions | 1 variable |
| Anomaly detection | Identify outliers | 1+ variable |
| Model comparison | Compare theoretical models | Observed + predicted |

---

## 6. V5.0 Discovery Enhancement System

### 6.1 Overview

V5.0 adds advanced discovery capabilities for genuine scientific discovery.

### 6.2 Capabilities

#### 6.2.1 Genuine Discovery Detection

```python
# Via API
curl -X POST http://localhost:8787/api/discovery/genuine \
  -H "Content-Type: application/json" \
  -d '{
    "data_source": "sdss",
    "novelty_threshold": 0.95
  }'
```

#### 6.2.2 Physical Model Discovery

```bash
# Discover physical models
curl -X POST http://localhost:8787/api/discovery/physical-model \
  -H "Content-Type: application/json" \
  -d '{
    "variables": ["mass", "luminosity"],
    "model_space": ["power_law", "exponential"]
  }'
```

---

## 7. V7.0 Autonomous Research Scientist

### 7.1 Overview

The V7.0 Autonomous Research Scientist conducts the entire scientific research cycle from question formulation through publication.

### 7.2 Core Components

#### 7.2.1 Question Generator

```bash
# Generate research questions
curl -X POST http://localhost:8787/api/questions/generate \
  -H "Content-Type: application/json" \
  -d '{
    "domain": "interstellar_medium",
    "context": {"focus": "filament_widths"},
    "num_questions": 5
  }'
```

#### 7.2.2 Hypothesis Formulator

```bash
# Formulate hypotheses
curl -X POST http://localhost:8787/api/hypotheses/formulate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What determines filament width?",
    "num_hypotheses": 3
  }'
```

---

## 8. V8.0 Cognitive Architecture (Scientific AGI) ⭐ NEW

### 8.1 Overview

V8.0 introduces a unified cognitive architecture integrating three foundational systems:

**Dynamic Knowledge Graph**: Semantic brain that integrates all discoveries into a coherent knowledge network

**Neuro-Symbolic Engine**: Unifies neural pattern recognition with symbolic reasoning for explainable discoveries

**Meta-Cognitive Architecture**: Self-awareness system that monitors and improves its own reasoning

### 8.2 Knowledge Graph

#### 8.2.1 Overview

The Knowledge Graph stores and reasons about scientific knowledge as a semantic network of entities and relations.

#### 8.2.2 API Endpoints

```bash
# Get knowledge graph statistics
curl http://localhost:8787/api/knowledge-graph/statistics

# Get knowledge gaps
curl http://localhost:8787/api/knowledge-graph/gaps

# Get cross-domain analogies
curl http://localhost:8787/api/knowledge-graph/analogies
```

#### 8.2.3 Example Output

```json
{
  "statistics": {
    "total_entities": 42,
    "total_relations": 38,
    "domains": ["astrophysics", "gravitation", "cosmology"],
    "graph_density": 0.43
  },
  "knowledge_gaps": [
    {
      "gap_type": "missing_relation",
      "description": "No known relation between dark_matter and filament_structure",
      "importance": 0.82
    }
  ]
}
```

### 8.3 Neuro-Symbolic Engine

#### 8.3.1 Overview

The Neuro-Symbolic Engine combines neural networks (for pattern discovery) with symbolic reasoning (for explanation and generalization).

#### 8.3.2 Pattern Discovery

```bash
# Discover and formalize patterns
curl -X POST http://localhost:8787/api/neuro-symbolic/discover \
  -H "Content-Type: application/json" \
  -d '{
    "data_source": "sdss",
    "variables": ["mass", "luminosity", "star_formation_rate"],
    "reasoning_mode": "causal"
  }'
```

#### 8.3.3 Example: Scaling Relation Discovery

```python
import requests

# Discover scaling relation
response = requests.post(
    "http://localhost:8787/api/neuro-symbolic/discover",
    json={
        "data_source": "exoplanets",
        "variables": ["mass", "radius"],
        "reasoning_mode": "scaling"
    }
)

result = response.json()
print(f"Relation: {result['symbolic_form']}")
# Output: R ∝ M^(0.45±0.08)

print(f"Explanation: {result['explanation']}")
# Output: "Mass-radius relation for exoplanets deviates from 
# theoretical R∝M^(1/3) prediction, suggesting atmospheric 
# inflation effects for hot Jupiters"
```

### 8.4 Meta-Cognitive Architecture

#### 8.4.1 Overview

The Meta-Cognitive Architecture enables ASTRA to reason about its own reasoning processes.

#### 8.4.2 API Endpoints

```bash
# Get meta-cognitive report
curl http://localhost:8787/api/metacognition/report

# Get method performance
curl http://localhost:8787/api/metacognition/methods
```

#### 8.4.3 Example Output

```json
{
  "report": {
    "cognitive_state": "normal",
    "total_reasoning_traces": 156,
    "methods_tracked": 8,
    "best_method": "statistical_test",
    "error_patterns_detected": 2,
    "confidence_calibration": "well_calibrated"
  },
  "method_performance": [
    {
      "method": "statistical_test",
      "success_rate": 0.95,
      "avg_confidence": 0.72
    },
    {
      "method": "causal_inference",
      "success_rate": 0.68,
      "avg_confidence": 0.54
    }
  ]
}
```

### 8.5 Cognitive Core

#### 8.5.1 Overview

The Cognitive Core orchestrates all cognitive systems in a unified discovery pipeline.

#### 8.5.2 Discovery Pipeline

```
PERCEIVE → reason → cross_domain → generate_insights
    ↓
REASON → symbolic_reasoning → pattern_formalization → explanation
    ↓
LEARN → update_beliefs → identify_gaps → propose_experiments
    ↓
DISCOVER → unify_theory_and_data → validate → publish
```

#### 8.5.3 Running Cognitive Discovery

```bash
# Get cognitive dashboard
curl http://localhost:8787/api/cognitive/dashboard

# Trigger cognitive discovery (runs automatically every 15 cycles)
curl -X POST http://localhost:8787/api/cognitive/discover
```

### 8.6 State Persistence

#### 8.6.1 Overview

ASTRA automatically saves state every 50 cycles and restores on restart, preserving all hypotheses and discoveries.

#### 8.6.2 API Endpoints

```bash
# Get persistence status
curl http://localhost:8787/api/state/persistence

# Manual save
curl -X POST http://localhost:8787/api/state/save

# Manual restore
curl -X POST http://localhost:8787/api/state/restore
```

#### 8.6.3 Example: Restoring After Restart

```bash
# Start server
python3 -m astra_live_backend.server

# State automatically restored
curl http://localhost:8787/api/state/persistence

# Output:
{
  "last_saved": "2026-04-06T18:25:00Z",
  "hypotheses_saved": 23,
  "cycle_count": 156,
  "auto_restore": "success"
}
```

### 8.7 Enhanced Hypothesis Generation

#### 8.7.1 Overview

V8.0 generates 3-5 hypotheses per cycle (vs. 1 in previous versions) from:
- Strong discoveries (60% weight)
- Untested variable pairs (25% weight)
- Cross-domain analogies (15% weight)

#### 8.7.2 Example Output

```bash
curl http://localhost:8787/api/hypotheses | jq '[.[] | select(.phase == "proposed")] | length'
# Output: 5
```

### 8.8 Live Dashboard

#### 8.8.1 Overview

The live dashboard provides real-time visualization of ASTRA's status with two modes:
- **LIVE** (green): Connected to API, real-time data
- **CACHED** (amber): Using embedded snapshot data

#### 8.8.2 Dashboard Features

- System status (phase, cycle count, confidence)
- Hypothesis funnel (proposed → testing → validated)
- Recent activity log
- Safety status
- Engine state space
- Knowledge graph statistics
- Meta-cognitive report

#### 8.8.3 Regenerating Dashboard

```bash
# Regenerate with current data
python3 -m astra_live_backend.generate_dashboard
```

---

## 9. Use Case Examples

### 9.1 Interstellar Medium Analysis

**Question**: "Why do filament widths cluster at 0.1 pc?"

```python
import requests

API_BASE = "http://localhost:8787"

# Create hypothesis
hypothesis = {
    "name": "Filament Sonic Scale Hypothesis",
    "domain": "Astrophysics",
    "description": "Test whether filament width is set by sonic scale",
    "confidence": 0.5
}

response = requests.post(f"{API_BASE}/api/hypotheses", json=hypothesis)
print(f"Created: {response.json()['id']}")
```

### 9.2 Exoplanet Analysis

```bash
# Query exoplanet data
curl http://localhost:8787/api/data/exoplanets | jq '.data[:5]'

# Run correlation analysis
curl -X POST http://localhost:8787/api/analyze/correlation \
  -H "Content-Type: application/json" \
  -d '{
    "data_source": "exoplanets",
    "variables": ["mass", "period"]
  }'
```

### 9.3 Galaxy Evolution

```bash
# Get galaxy data statistics
curl http://localhost:8787/api/data/sdss/stats

# Run causal inference
curl -X POST http://localhost:8787/api/analyze/causal \
  -H "Content-Type: application/json" \
  -d '{
    "data_source": "sdss",
    "variables": ["mass", "sfr", "metallicity"]
  }'
```

### 9.4 Cognitive Discovery

```bash
# Get cognitive insights
curl http://localhost:8787/api/cognitive/discoveries | jq '.'

# Find cross-domain analogies
curl http://localhost:8787/api/knowledge-graph/analogies | jq '.[]'
```

---

## 10. Advanced Features

### 10.1 Safety Architecture

ASTRA includes a comprehensive safety subsystem:

```bash
# Get safety status
curl http://localhost:8787/api/engine/safety-status

# Get arbiter verdicts
curl http://localhost:8787/api/engine/arbiter/verdicts
```

### 10.2 Swarm Intelligence

```bash
# Get swarm status
curl http://localhost:8787/api/swarm/status

# Get pheromone field
curl http://localhost:8787/api/swarm/pheromones
```

### 10.3 Literature Integration

```bash
# Search papers
curl http://localhost:8787/api/literature/search?q=filament+width

# Get citation graph
curl http://localhost:8787/api/literature/citation-graph
```

---

## 11. Domain Modules

ASTRA includes 75 specialized domain modules. See V7.0 manual for complete list.

### 11.1 Available Domains (Summary)

**Stellar Astrophysics** (8 domains)
**Interstellar Medium** (8 domains)
**Exoplanets & Solar System** (4 domains)
**High-Energy Astrophysics** (5 domains)
**Galaxy Evolution** (8 domains)
**Compact Objects** (7 domains)
**Observational Techniques** (12 domains)
**Theoretical Physics** (10 domains)
**Radiation & Atomic Physics** (6 domains)
**Solar Physics** (2 domains)
**Cross-Disciplinary** (5 domains)

---

## 12. API Reference

### 12.1 Core Endpoints

#### Status

```bash
GET /api/status
```

Returns system status including phase, cycle count, confidence.

#### Hypotheses

```bash
GET /api/hypotheses           # List all hypotheses
POST /api/hypotheses          # Create hypothesis
GET /api/hypotheses/{id}      # Get specific hypothesis
```

#### Activity

```bash
GET /api/activity             # Get activity log
GET /api/activity?limit=20    # Get last 20 activities
```

### 12.2 Cognitive Endpoints (V8.0)

#### Cognitive Status

```bash
GET /api/cognitive/status     # Cognitive architecture overview
GET /api/cognitive/dashboard  # Unified cognitive dashboard
GET /api/cognitive/discoveries # Cognitive discoveries
```

#### Knowledge Graph

```bash
GET /api/knowledge-graph/statistics   # Graph statistics
GET /api/knowledge-graph/gaps         # Knowledge gaps
GET /api/knowledge-graph/analogies    # Cross-domain analogies
```

#### Meta-Cognition

```bash
GET /api/metacognition/report         # Self-awareness report
GET /api/metacognition/methods        # Method performance
```

#### State Persistence

```bash
GET /api/state/persistence            # Persistence status
POST /api/state/save                  # Manual save
POST /api/state/restore               # Manual restore
```

### 12.3 Data Endpoints

```bash
GET /api/data/{source}                # Get data from source
GET /api/data/{source}/stats          # Data statistics
```

Available sources: `pantheon`, `exoplanets`, `gaia`, `sdss`, `ligo`, `planck`, `ztf`, `tess`, `clusters`

---

## 13. Best Practices

### 13.1 Data Preparation

- Use standard formats (FITS, CSV, HDF5)
- Include proper metadata
- Propagate uncertainties
- Document data provenance

### 13.2 Hypothesis Design

- Make hypotheses falsifiable
- Specify test predictions
- Include domain context
- Set appropriate confidence priors

### 13.3 API Usage

- Use appropriate HTTP methods (GET, POST)
- Handle JSON responses correctly
- Implement retry logic for failures
- Cache responses when appropriate

### 13.4 Monitoring

- Check dashboard regularly
- Monitor cycle progress
- Review safety status
- Track confidence trends

---

## 14. Troubleshooting

### 14.1 Common Issues

**Issue**: Dashboard shows "CACHED" status

**Solution**:
```bash
# Check API connectivity
curl http://localhost:8787/api/status

# If unreachable, restart server
pkill -f "python.*astra_live_backend.server"
python3 -m astra_live_backend.server
```

**Issue**: No hypotheses generated

**Solution**:
```bash
# Check discovery memory
curl http://localhost:8787/api/discovery-memory

# Initialize hypotheses if needed
python3 init_hypotheses.py

# Trigger discovery cycle
curl -X POST http://localhost:8787/api/engine/cycle
```

**Issue**: State not restored after restart

**Solution**:
```bash
# Check state files
ls -la astra_state/

# Manual restore
curl -X POST http://localhost:8787/api/state/restore
```

### 14.2 Getting Help

- Check GitHub issues: https://github.com/Tilanthi/ASTRA-dev/issues
- API documentation: http://localhost:8787/docs
- Dashboard: http://localhost:8787
- Check logs in `astra_state/logs/`

---

## 15. Appendices

### Appendix A: Complete Capability List

1. Bias Detection
2. Scaling Relations Discovery
3. Causal Inference
4. Model Selection
5. Multi-Wavelength Fusion
6. Uncertainty Quantification
7. Temporal Analysis
8. Instrument-Aware Analysis
9. Anomaly Detection
10. Ensemble Prediction
11. Physical Model Discovery
12. Bayesian Model Selection
13. Counterfactual Analysis
14. Genuine Discovery Detection
15. **V8.0: Dynamic Knowledge Graph**
16. **V8.0: Neuro-Symbolic Reasoning**
17. **V8.0: Meta-Cognitive Architecture**
18. **V8.0: Cognitive Core Orchestrator**
19. **V8.0: State Persistence**
20. **V8.0: Enhanced Hypothesis Generation**
21. **V8.0: Live Dashboard**
22. **V8.0: Cross-Domain Analogies**
23. **V8.0: Knowledge Gap Identification**
24. **V8.0: Method Performance Tracking**
25. **V8.0: Reflective Self-Improvement**

### Appendix B: Data Sources

| Source | Records | Description |
|--------|---------|-------------|
| Pantheon+ | 1,701 | Type Ia supernovae distance moduli |
| NASA Exoplanet | 2,839 | Confirmed exoplanets |
| Gaia DR3 | 4,984 | Stellar astrometry and photometry |
| SDSS DR18 | 2,000+ | Galaxy photometry and spectra |
| LIGO | 280 | Gravitational wave events |
| Planck | 2,507 | CMB power spectrum |
| ZTF | 2,000 | Transient light curves |
| TESS | TBD | Exoplanet host stars |
| SDSS Clusters | TBD | Galaxy cluster catalogs |

### Appendix C: Statistical Methods

| Method | Input | Output | Use Case |
|--------|-------|--------|----------|
| KS test | 2 samples | D, p-value | Distribution comparison |
| Chi-squared | Observed, expected | χ², p-value | Goodness of fit |
| Pearson | 2 variables | r, p-value | Linear correlation |
| Granger | Time series | F, p-value | Temporal causality |
| PC algorithm | 3+ variables | DAG | Causal structure |
| BIC | Models, data | BIC, ΔBIC | Model selection |

### Appendix D: Quick Reference

#### Starting ASTRA

```bash
cd /path/to/ASTRA-dev
python3 -m astra_live_backend.server
```

#### Checking Status

```bash
curl http://localhost:8787/api/status | jq '.'
```

#### Running Discovery Cycle

```bash
curl -X POST http://localhost:8787/api/engine/cycle
```

#### Stopping ASTRA

```bash
pkill -f "python.*astra_live_backend.server"
```

---

**Document Version**: 8.0
**Last Updated**: April 2026
**Authors**: Glenn J. White, Open University and Rutherford Appleton Laboratory, England
**License**: MIT License

For the latest version, visit: https://github.com/Tilanthi/ASTRA-dev
