"""
router/quality_engine.py
-------------------------
Answer Quality Engine for IntelliMoE.
Implements the multi-stage quality assurance pipeline for generated responses:
1. PromptBuilder - Retrieves the professional system prompt for each expert.
2. ResponsePlanner - Formulates an answer plan before calling the LLM.
3. Context Builder - Integrates conversation history, expert metadata, query, and RAG context.
4. Response Generator - Generates a detailed draft response.
5. Response Reviewer - Critiques the draft response against correctness, completeness, clarity, and formatting.
6. Response Improver - Rewrites the draft to fix any identified weaknesses.
"""

import logging
from typing import Optional, TYPE_CHECKING, Dict, Any
from utils.prompt_manager import PromptManager
from config.settings import GEMINI_MODEL_ID, EXPERT_CONFIGS

if TYPE_CHECKING:
    from utils.memory import ConversationMemory

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds and loads professional system prompts for each expert."""
    
    def __init__(self) -> None:
        self.prompt_manager = PromptManager()

    def get_system_prompt(self, prompt_name: str) -> str:
        """Dynamically load system prompt from the prompts directory."""
        try:
            return self.prompt_manager.get_prompt(prompt_name)
        except Exception as e:
            logger.warning("PromptBuilder: failed to load prompt '%s', using default: %s", prompt_name, e)
            return f"You are a helpful expert in {prompt_name}."


class ResponsePlanner:
    """Generates an answer plan before calling the LLM."""
    
    @staticmethod
    def plan(expert_name: str, system_prompt: str, context: str, query: str) -> str:
        from services.gemini_client import generate_response
        
        planning_prompt = (
            f"You are the Answer Planner for IntelliMoE.\n"
            f"Your job is to formulate a structured answer plan for the {expert_name} expert, "
            f"aiming to perfectly address the user's question.\n\n"
            f"Expert Instructions:\n{system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"User Question: {query}\n\n"
            f"INSTRUCTION: Outline a clear, step-by-step response strategy. Specify exactly what sections to include, "
            f"which technical details or code/math formulas are needed, and how to structure the explanation. "
            f"Return only the step-by-step plan. Do not write the actual answer."
        )
        
        try:
            plan_output = generate_response(
                prompt=planning_prompt,
                system_prompt="You are an expert answer planner. Output a concise markdown plan.",
                model=GEMINI_MODEL_ID,
                temperature=0.2
            )
            return plan_output.strip()
        except Exception as e:
            logger.error("ResponsePlanner: planning LLM call failed: %s", e)
            return "1. Address the query directly.\n2. Provide necessary technical explanations."


class ResponseReviewer:
    """Reviews the generated draft response for correctness, completeness, clarity, and formatting."""
    
    @staticmethod
    def review(draft: str, query: str, context: str) -> str:
        from services.gemini_client import generate_response
        
        reviewer_prompt = (
            f"You are the Response Reviewer for IntelliMoE.\n"
            f"Critically review the draft response to the user's question.\n\n"
            f"Context:\n{context}\n\n"
            f"Original Question: {query}\n\n"
            f"Draft Response:\n{draft}\n\n"
            f"INSTRUCTION: Check the draft response against the following criteria:\n"
            f"1. Correctness (are code snippets, formulas, or concepts technically correct?)\n"
            f"2. Completeness (does it answer all components of the user's question?)\n"
            f"3. Clarity (is it easy to read, logical, and conversational?)\n"
            f"4. Formatting (is it clean, standard markdown, and visually appealing?)\n\n"
            f"Provide a summary of weaknesses or suggestions for improvement. Be concise and direct."
        )
        
        try:
            review_output = generate_response(
                prompt=reviewer_prompt,
                system_prompt="You are a strict response reviewer. Output a concise list of weaknesses and improvements.",
                model=GEMINI_MODEL_ID,
                temperature=0.2
            )
            return review_output.strip()
        except Exception as e:
            logger.error("ResponseReviewer: reviewer LLM call failed: %s", e)
            return "No critical weaknesses identified."


class ResponseImprover:
    """Improves weak sections identified by the reviewer and returns the final polished response."""
    
    @staticmethod
    def improve(draft: str, review: str, query: str, context: str, system_prompt: str) -> str:
        from services.gemini_client import generate_response
        
        improver_prompt = (
            f"You are the Response Improver for IntelliMoE.\n"
            f"Your job is to revise the draft response using the reviewer's feedback to produce the perfect final answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Original Question: {query}\n\n"
            f"Draft Response:\n{draft}\n\n"
            f"Reviewer Feedback:\n{review}\n\n"
            f"INSTRUCTION: Generate a fully polished, corrected, and improved response. "
            f"Incorporate all fixes for correctness, formatting, clarity, and completeness.\n"
            f"Return ONLY the final improved response. No commentary, no introductory or concluding chat metadata."
        )
        
        try:
            final_output = generate_response(
                prompt=improver_prompt,
                system_prompt=system_prompt,
                model=GEMINI_MODEL_ID,
                temperature=0.4
            )
            return final_output.strip()
        except Exception as e:
            logger.error("ResponseImprover: improvement LLM call failed: %s", e)
            return draft


class AnswerQualityEngine:
    """Coordinates PromptBuilder, ResponsePlanner, ResponseReviewer, and ResponseImprover."""
    
    def __init__(self) -> None:
        self.prompt_builder = PromptBuilder()
        self.planner = ResponsePlanner()
        self.reviewer = ResponseReviewer()
        self.improver = ResponseImprover()

    def generate_quality_response(
        self,
        expert: Any,
        expert_name: str,
        query: str,
        memory: Optional["ConversationMemory"] = None,
        image_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Runs the Answer Quality Engine pipeline.
        Returns a dictionary containing the final response, plan, and review.
        """
        # Check if AQE is disabled (Fast Mode)
        enable_aqe = True
        try:
            import streamlit as st
            if "enable_aqe" in st.session_state:
                enable_aqe = st.session_state.enable_aqe
        except Exception:
            pass

        if not enable_aqe:
            logger.info("AnswerQualityEngine [%s]: AQE is disabled (Fast Mode), calling expert directly.", expert_name)
            if expert_name == "vision" and hasattr(expert, "answer"):
                ans = expert.answer(query, memory=memory, image_path=image_path)
            else:
                ans = expert.answer(query, memory=memory)
            return {
                "answer": ans,
                "plan": "Answer Quality Engine bypassed (Fast Mode enabled).",
                "review": "Answer Quality Engine bypassed (Fast Mode enabled)."
            }

        # 1. Build Prompt
        system_prompt = self.prompt_builder.get_system_prompt(expert_name)

        # 2. Build Context
        rag_context = ""
        if expert_name == "research" and hasattr(expert, "_retrieve"):
            try:
                rag_context = expert._retrieve(query)
            except Exception as e:
                logger.warning("AnswerQualityEngine: failed to retrieve research RAG context: %s", e)

        # Format memory context
        history_text = ""
        if memory and not memory.is_empty:
            for turn in memory.get_turns():
                history_text += f"User: {turn.question}\nAssistant: {turn.answer}\n\n"

        context_str = f"Selected Expert: {expert_name}\n"
        if history_text:
            context_str += f"\nConversation History:\n{history_text}"
        if rag_context:
            context_str += f"\nRetrieved Context:\n{rag_context}"
        if image_path:
            context_str += f"\nImage Attached: {image_path}"

        # 3. Planning Phase
        logger.info("AnswerQualityEngine [%s]: Planning response...", expert_name)
        plan_content = self.planner.plan(expert_name, system_prompt, context_str, query)
        logger.info("AnswerQualityEngine [%s]: Plan generated.", expert_name)

        # 4. Draft Generation Phase
        logger.info("AnswerQualityEngine [%s]: Generating draft response...", expert_name)
        draft_prompt = (
            f"Answer Plan:\n{plan_content}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}"
        )
        
        # Select target API matching standard expert preferences
        from services.gemini_client import generate_response as gemini_gen
        from services.groq_client import generate_response as groq_gen
        
        try:
            if expert_name in ["coding", "math"]:
                draft = groq_gen(
                    prompt=draft_prompt,
                    system_prompt=system_prompt,
                    model="llama-3.1-8b-instant",
                    temperature=0.5
                )
            else:
                draft = gemini_gen(
                    prompt=draft_prompt,
                    system_prompt=system_prompt,
                    model=GEMINI_MODEL_ID,
                    temperature=0.5
                )
            logger.info("AnswerQualityEngine [%s]: Draft generated.", expert_name)
        except Exception as e:
            logger.warning("AnswerQualityEngine [%s]: Draft generation failed, falling back to standard answer: %s", expert_name, e)
            # Safe fallback: run the standard expert answer
            if expert_name == "vision" and hasattr(expert, "answer"):
                return {
                    "answer": expert.answer(query, memory=memory, image_path=image_path),
                    "plan": "Standard fallback plan.",
                    "review": "Standard fallback review."
                }
            return {
                "answer": expert.answer(query, memory=memory),
                "plan": "Standard fallback plan.",
                "review": "Standard fallback review."
            }

        # 5. Review Phase
        logger.info("AnswerQualityEngine [%s]: Reviewing draft response...", expert_name)
        review_content = self.reviewer.review(draft, query, context_str)
        logger.info("AnswerQualityEngine [%s]: Review completed.", expert_name)

        # 6. Improvement Phase
        logger.info("AnswerQualityEngine [%s]: Improving response...", expert_name)
        improved_response = self.improver.improve(draft, review_content, query, context_str, system_prompt)
        logger.info("AnswerQualityEngine [%s]: Response improved successfully.", expert_name)

        return {
            "answer": improved_response,
            "plan": plan_content,
            "review": review_content
        }
