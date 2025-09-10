from .learner import SelfLearningAgent, KnowledgeBase, CriticalReasoningEngine

# For backward compatibility
try:
    from .autonomous_learner import AutonomousLearningAgent
except ImportError:
    # Fallback if external dependencies are not available
    AutonomousLearningAgent = None

__all__ = ['SelfLearningAgent', 'KnowledgeBase', 'CriticalReasoningEngine', 'AutonomousLearningAgent']
