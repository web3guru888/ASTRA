# ASTRA: Autonomous System for Scientific Discovery in Astrophysics

**Version**: 4.7
**AGI Capability Estimate**: 70-75%

ASTRA is a unified AGI-inspired framework for autonomous hypothesis generation and validation in astronomy and astrophysics. The system integrates ~303,000 lines of clean, functional code across modular cognitive capabilities.

## Overview

ASTRA combines advanced AI techniques including:
- **Causal Inference & Discovery**: Structural causal models, PC algorithm, counterfactual reasoning
- **Meta-Learning**: MAML optimization, cross-domain transfer learning
- **Swarm Intelligence**: Multi-agent reasoning, stigmergic coordination
- **Domain Expertise**: 75 specialized astrophysics domain modules
- **V4 Revolutionary Capabilities**: Meta-Context Engine, Autocatalytic Self-Compiler, Cognitive-Relativity Navigator, Multi-Mind Orchestration

## Quick Start

### Paper and Documentation

**Full Paper**: See `RASTI_AI/draft_paper_complete_v9.md` and `RASTI_AI/ASTRA_paper_complete.pdf` for the complete scientific paper describing ASTRA's capabilities with 15 comprehensive test cases using real astronomical data.

### Installation

```bash
# Clone the repository
git clone https://github.com/Tilanthi/ASTRA.git
cd ASTRA

# Install dependencies
pip install -e .
```

### Basic Usage

```python
from stan_core import create_stan_system

# Create system with auto-optimized capabilities
system = create_stan_system()

# Answer queries with automatic capability selection
result = system.answer("What causes supernovae?")
print(result['answer'])
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Entry Points (Top Layer)                     │
│  create_stan_system() | create_v4_system() | process_query()   │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                 V4.0 Revolutionary Capabilities                  │
│  MCE (Context) | ASC (Self-Improvement) | CRN (Abstraction)    │
│  MMOL (7 Specialized Minds)                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                    Domain Architecture                          │
│  BaseDomainModule → DomainRegistry → Specialized Domains        │
│  (75 domains: ISM, Star Formation, Exoplanets, GW, Cosmology,  │
│   Solar System, Time Domain, High-Energy, etc.)                │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                Cross-Domain Meta-Learning                       │
│  MAMLOptimizer | CrossDomainMetaLearner | AdaptationResult      │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                   Physics & Causal Engines                      │
│  UnifiedPhysicsEngine | StructuralCausalModel | PCAlgorithm      │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                  Memory & Knowledge Systems                     │
│  MORK Ontology | Memory Graph | Vector Store | Working Memory   │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 75 Domain Modules
- Interstellar Medium (ISM)
- Star Formation
- Exoplanets
- Gravitational Waves
- Cosmology
- High-Energy Astrophysics
- Solar System
- Time Domain Astronomy
- Galactic Archaeology
- And 66 more specialized domains

### V4 Revolutionary Capabilities
- **Meta-Context Engine (MCE)**: Multi-layered context representation with temporal, perceptual, domain, modality, certainty, social, and epistemic dimensions
- **Autocatalytic Self-Compiler (ASC)**: Self-improving system architecture with version management and safe mutation
- **Cognitive-Relativity Navigator (CRN)**: Adaptive abstraction navigation with 0-100 scale
- **Multi-Mind Orchestration Layer (MMOL)**: 7 specialized minds (Physics, Empathy, Politics, Poetry, Mathematics, Causal, Creative)

### Physics Engine
- Unified Physics Engine with 8 models
- Relativistic Physics
- Quantum Mechanics
- Nuclear Astrophysics
- Differentiable Physics

### Advanced Reasoning
- Causal Discovery (PC Algorithm, V50, V70)
- Swarm Reasoning
- Hierarchical Bayesian Meta-Learning
- Cross-Domain Meta-Learning
- MAML Optimization

## Testing

### Run All Tests

```bash
# Comprehensive system test
python stan_core/comprehensive_system_test.py

# V4 capability tests
python stan_core/tests/v4/run_tests.py

# Specialist capability tests
python stan_core/tests/test_specialist_capabilities.py
```

### Test Results

| Test Suite | Result |
|------------|--------|
| Comprehensive System Test | ✅ 18/18 (100%) |
| V4 Capability Tests | ✅ 5/5 (100%) |
| Specialist Capabilities | ✅ 6/6 (100%) |

## Project Statistics

- **Total Lines**: ~303,000
- **Python Files**: 514+
- **Domain Modules**: 75
- **Specialist Capabilities**: 66+
- **V4 Capabilities**: 4 revolutionary systems

## Citation

If you use ASTRA in your research, please cite:

```bibtex
@software{astra_2024,
  title={ASTRA: Autonomous System for Scientific Discovery in Astrophysics},
  author={[Author Names]},
  year={2024},
  version={4.7},
  url={https://github.com/YOUR_USERNAME/ASTRA}
}
```

## License

[Specify your license here]

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting pull requests.

## Acknowledgments

ASTRA builds upon research in:
- Causal inference and discovery
- Meta-learning and transfer learning
- Swarm intelligence and multi-agent systems
- Cognitive architectures and AGI
- Astrophysics and scientific discovery

## Contact

For questions, issues, or collaborations, please open an issue on GitHub or contact [your contact information].

---

**Note**: ASTRA was previously known as "STAN-XI-ASTRO" internally. The codebase retains the "stan" naming for backward compatibility.
