"""
ASTRA V9.0 — Multi-Agent Scientific Collaboration System
Provides specialized agents that collaborate on scientific problems.

This module implements Phase 1 of V9.0: Multi-Agent Foundation

Components:
- AgentFactory: Creates specialized scientific agents
- CollaborationProtocol: Manages agent communication
- ConsensusEngine: Computes consensus from agent opinions
- DebateOrchestrator: Orchestrates structured scientific debates
- ExpertiseTracker: Tracks agent specializations and performance

Example usage:
    from astra_live_backend.multi_agent import (
        AgentFactory, DebateOrchestrator, create_debate
    )

    # Create agents
    agents = AgentFactory.create_full_team()

    # Start debate
    orchestrator = create_debate()
    debate_id = orchestrator.start_debate(
        "What determines the characteristic width of interstellar filaments?",
        [agent.id for agent in agents]
    )

    # Advance debate through phases
    orchestrator.advance_debate(debate_id)

    # Get result
    result = orchestrator.conclude_debate(debate_id)
    print(result.recommendation)
"""

from .agent_factory import (
    AgentFactory,
    ScientificAgent,
    AgentRole,
    AgentStatus,
    AgentExpertise,
    AgentOpinion,
    TheoristAgent,
    EmpiricistAgent,
    ExperimentalistAgent,
    MathematicianAgent,
    SkepticAgent,
    SynthesizerAgent,
    get_agent_description,
    validate_agent_collaboration
)

from .collaboration_protocol import (
    CollaborationProtocol,
    DebateProtocol,
    EvidenceExchangeProtocol,
    DiscussionContext,
    AgentMessage,
    MessageType,
    MessagePriority,
    create_protocol,
    format_message_for_display,
    summarize_discussion
)

from .consensus_engine import (
    ConsensusEngine,
    ConsensusMethod,
    ConsensusResult,
    ConflictResolver,
    format_consensus_result,
    compare_consensus_methods
)

from .debate_orchestrator import (
    DebateOrchestrator,
    DebateConfig,
    DebatePhase,
    DebateResult,
    DebateMetrics,
    create_debate
)

from .expertise_tracker import (
    ExpertiseTracker,
    ExpertiseLevel,
    SpecializationProfile,
    TaskPerformance,
    create_expertise_tracker,
    compare_agent_performance
)

__all__ = [
    # Factory
    "AgentFactory",
    "create_debate",

    # Agents
    "ScientificAgent",
    "AgentRole",
    "AgentStatus",
    "TheoristAgent",
    "EmpiricistAgent",
    "ExperimentalistAgent",
    "MathematicianAgent",
    "SkepticAgent",
    "SynthesizerAgent",

    # Protocol
    "CollaborationProtocol",
    "DebateProtocol",
    "DiscussionContext",
    "AgentMessage",
    "MessageType",

    # Consensus
    "ConsensusEngine",
    "ConsensusMethod",
    "ConsensusResult",

    # Debate
    "DebateOrchestrator",
    "DebateConfig",
    "DebateResult",

    # Expertise
    "ExpertiseTracker",
    "ExpertiseLevel",

    # Utilities
    "format_message_for_display",
    "format_consensus_result",
    "summarize_discussion"
]
