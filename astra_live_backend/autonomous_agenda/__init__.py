"""
ASTRA V9.0 — Autonomous Scientific Agenda
Enables ASTRA to set its own research goals through curiosity metrics.

Components:
- CuriosityEngine: Computes information-theoretic curiosity metrics
- GoalGenerator: Generates research goals from knowledge gaps
- ApprovalWorkflow: Manages human approval for semi-autonomous mode
- AutonomousAgenda: Main system integrating all components

Example usage:
    from astra_live_backend.autonomous_agenda import create_autonomous_agenda

    agenda = create_autonomous_agenda(
        knowledge_graph=kg,
        discovery_memory=dm,
        mode="semi_autonomous"
    )

    goals = agenda.generate_research_agenda(num_goals=5)
    for goal in goals:
        print(f"{goal.title}: {goal.curiosity_score:.2f}")
"""

from .curiosity_engine import (
    CuriosityMetric,
    GoalStatus,
    GoalPriority,
    ResearchGoal,
    CuriosityAssessment,
    CuriosityEngine,
    GoalGenerator,
    ApprovalWorkflow,
    AutonomousAgenda,
    create_autonomous_agenda,
    create_curiosity_engine
)

__all__ = [
    "CuriosityMetric",
    "GoalStatus",
    "GoalPriority",
    "ResearchGoal",
    "CuriosityEngine",
    "GoalGenerator",
    "ApprovalWorkflow",
    "AutonomousAgenda",
    "create_autonomous_agenda"
]
