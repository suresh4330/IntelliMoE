"""
conversation_ai/__init__.py
----------------------------
Intelligent Conversational AI Layer for IntelliMoE.

This module sits between the user input and the Hybrid Router.
It classifies intent and either handles the query conversationally (greetings,
small talk, follow-ups, general knowledge) or passes it to the expert routing
pipeline unchanged.

Public API
----------
>>> from conversation_ai.layer import ConversationLayer, ConversationResult
>>> layer = ConversationLayer()
>>> result = layer.process("How are you?", memory)
>>> if result.is_conversational:
...     print(result.response)
"""

from conversation_ai.layer import ConversationLayer, ConversationResult
from conversation_ai.detector import IntentDetector, IntentResult, IntentType

__all__ = [
    "ConversationLayer",
    "ConversationResult",
    "IntentDetector",
    "IntentResult",
    "IntentType",
]
