"""
Autonomous Research System

This module contains the V7.0 autonomous research scientist system.
"""

# Import main system + re-exported types (so astra_core/__init__.py can use
# `from .autonomous_research import ...` for all V7 symbols)
from .v7_autonomous_scientist import (
    V7AutonomousScientist,
    create_v7_scientist,
)

# Shared types (used by all engines — imported here for flat top-level access)
from .types import (
    ResearchCycle,
    ResearchQuestion,
    Hypothesis,
    Experiment,
    ResearchResult,
    Publication,
    QuestionType,
    QuestionImportance,
    HypothesisType,
    HypothesisStatus,
    ExperimentType,
    DesignParameters,
    DataSource,
    ExecutionResult,
    PredictionType,
    PredictionConfidence,
    AnalysisType,
    CausalInferenceResult,
    RevisionType,
    TheoryStatus,
    PaperStructure,
    FigureType,
)

# Engine classes
from .engines.question_generator import QuestionGenerator
from .engines.hypothesis_formulator import HypothesisFormulator
from .engines.experiment_designer import ExperimentDesigner
from .engines.experiment_executor import ExperimentExecutor
from .engines.prediction_engine import (
    PredictionEngine,
    AnalysisEngine,
    TheoryRevisionEngine,
    PublicationEngine,
)

__all__ = [
    # Main system
    'V7AutonomousScientist',
    'create_v7_scientist',
    # Core research types
    'ResearchCycle',
    'ResearchQuestion',
    'Hypothesis',
    'Experiment',
    'ResearchResult',
    'Publication',
    # Question types
    'QuestionGenerator',
    'QuestionType',
    'QuestionImportance',
    # Hypothesis types
    'HypothesisFormulator',
    'HypothesisType',
    'HypothesisStatus',
    # Experiment types
    'ExperimentDesigner',
    'ExperimentType',
    'DesignParameters',
    'ExperimentExecutor',
    'ExecutionResult',
    'DataSource',
    # Prediction & analysis
    'PredictionEngine',
    'PredictionType',
    'PredictionConfidence',
    'AnalysisEngine',
    'AnalysisType',
    'CausalInferenceResult',
    # Theory & publication
    'TheoryRevisionEngine',
    'RevisionType',
    'TheoryStatus',
    'PublicationEngine',
    'PaperStructure',
    'FigureType',
]
