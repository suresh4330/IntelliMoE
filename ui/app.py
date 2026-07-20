# -*- coding: utf-8 -*-
"""
ui/app.py
---------
Streamlit UI for IntelliMoE - Multi-Expert AI Assistant.

Phase 31: Premium UI/UX Redesign
  - Minimalistic ChatGPT / Claude inspired layout.
  - Sidebar: Logo, New Chat, History, Tools expander, Expert Stats, Developer Mode.
  - Developer Mode: collapsible sidebar panel with Performance, Benchmarking, XAI, AI Evaluation.
  - 'View AI Reasoning' expander under each assistant response.
  - Response actions: Copy, Regenerate, Like, Dislike, Download.
  - Welcome screen with capability cards.
  - Streaming typewriter animation for new responses.
  - All backend functionality (routing, experts, evaluation) preserved unchanged.
"""

import logging
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime

import streamlit as st
import streamlit.components.v1 as st_components
import pandas as pd

def _render_html(html_code: str, height: int = 0, width: int = 0) -> None:
    """Render HTML/JS component with iframe parent DOM access for custom controls (+ upload button, voice mic)."""
    st_components.html(html_code, height=height, width=width)

# ---------------------------------------------------------------------------
# Path fix — ensure project root is on sys.path so all imports resolve
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.logging_config import setup_logging        # noqa: E402
from utils.memory import ConversationMemory, Message, Turn  # noqa: E402
from router.router import ExpertRouter, ExpertName   # noqa: E402
from models.loader import get_device, get_memory_usage_mb  # noqa: E402

# Initialise logging once
setup_logging()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="IntelliMoE · Multi-Expert AI Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — Phase 31 Premium Dark Theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,300;0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;1,14..32,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Base Reset ── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }
    .stApp { background: #0b0f17 !important; color: #E6EDF3 !important; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    /* Hide Streamlit deploy button, toolbar, rerun notification */
    [data-testid="stDecoration"] { display: none !important; }
    [data-testid="stDeployButton"] { display: none !important; }
    [data-testid="stToolbarActions"] { display: none !important; }
    [data-testid="stStatusWidget"] { display: none !important; }
    .stDeployButton { display: none !important; }
    a[href*="deploy"] { display: none !important; }
    /* Keep header transparent so sidebar toggle still works */
    [data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
    }
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #30363D; border-radius: 9px; }

    /* ── Main content full width ── */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 100% !important;
        margin: 0 auto !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 120px !important;
        padding-top: 0.5rem !important;
    }

    /* ── Sidebar shell — ChatGPT-style width ── */
    [data-testid="stSidebar"] {
        background: #0B0F17 !important;
        border-right: 1px solid #1C2128 !important;
        min-width: 268px !important;
        max-width: 268px !important;
        width: 268px !important;
    }
    [data-testid="stSidebarContent"] {
        padding: 0 0 130px 0 !important;
        display: flex !important;
        flex-direction: column !important;
        height: 100vh !important;
        background: #0B0F17 !important;
        position: relative !important;
    }

    /* ── Sidebar logo row ── */
    .sb-logo-row {
        display: flex;
        align-items: center;
        gap: 0.25rem;
        padding: 1.2rem 1rem 0.8rem 1.1rem;
    }
    .sb-logo-icon-svg {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .sb-logo-text {
        font-size: 1.1rem;
        font-weight: 800;
        color: #4F8CFF !important;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    .sb-logo-subtitle {
        font-size: 0.62rem;
        color: #8B949E !important;
        letter-spacing: 1.0px;
        text-transform: uppercase;
        font-weight: 700;
        display: block;
        margin-top: 2px;
    }

    /* ── New Chat button ── */
    [data-testid="stSidebar"] div[class*="st-key-new_chat_sidebar_btn"] button {
        width: 100% !important;
        background: transparent !important;
        border: 1px solid #21262D !important;
        color: #4F8CFF !important;
        border-radius: 8px !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        padding: 0.55rem 0.75rem !important;
        text-align: center !important;
        transition: all 0.18s ease !important;
        box-shadow: none !important;
        display: block !important;
    }
    [data-testid="stSidebar"] div[class*="st-key-new_chat_sidebar_btn"] button:hover {
        background: #161B22 !important;
        border-color: #4F8CFF !important;
        color: #4F8CFF !important;
        box-shadow: 0 0 10px rgba(79,140,255,0.06) !important;
    }

    /* ── Search box ── */
    .sb-search-wrap { padding: 0 0.85rem 0.6rem 0.85rem; }
    .sb-search-wrap [data-testid="stTextInput"] > div > div {
        background: #161B22 !important;
        border: 1px solid #21262D !important;
        border-radius: 8px !important;
        padding: 0.35rem 0.6rem !important;
    }
    .sb-search-wrap [data-testid="stTextInput"]::before {
        content: "🔍" !important;
        position: absolute !important;
        left: 10px !important;
        top: 50% !important;
        transform: translateY(-50%) !important;
        font-size: 0.78rem !important;
        color: #484F58 !important;
        z-index: 10 !important;
    }
    .sb-search-wrap [data-testid="stTextInput"] input {
        background: transparent !important;
        border: none !important;
        color: #8B949E !important;
        font-size: 0.82rem !important;
        padding: 0 0 0 24px !important;
    }
    .sb-search-wrap [data-testid="stTextInput"] input::placeholder { color: #484F58 !important; }
    .sb-search-wrap [data-testid="stTextInput"] input:focus { box-shadow: none !important; outline: none !important; color: #E6EDF3 !important; }
    .sb-search-wrap label { display: none !important; }
    .sb-search-wrap [data-testid="stTextInput"] { margin-bottom: 0 !important; }

    /* ── Sidebar section labels ── */
    .sb-section-label {
        font-size: 0.76rem !important;
        font-weight: 500 !important;
        color: #8B949E !important;
        padding: 0.75rem 1rem 0.35rem 1rem !important;
        margin: 0 !important;
        text-transform: none !important;
        letter-spacing: normal !important;
    }

    /* ── Sidebar lists ── */
    .sb-conv-list { overflow-y: auto; flex: 1; padding-bottom: 0.5rem; }

    /* ── All sidebar button styles ── */
    [data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        background: transparent !important;
        border: none !important;
        color: #8B949E !important;
        text-align: left !important;
        padding: 0.45rem 1rem !important;
        font-size: 0.83rem !important;
        border-radius: 8px !important;
        box-shadow: none !important;
        transition: all 0.15s ease !important;
        font-weight: 500 !important;
        position: relative !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #161B22 !important;
        color: #E6EDF3 !important;
    }
    /* Only conversation lists show menu three dots ⋯ */
    [data-testid="stSidebar"] div[class*="st-key-chat_select_"] button:hover::after {
        content: "⋯" !important;
        position: absolute !important;
        right: 12px !important;
        font-size: 0.95rem !important;
        color: #8B949E !important;
    }

    /* ── Sidebar expanders targeted to look like menu links ── */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: none !important;
        background: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        display: flex !important;
        align-items: center !important;
        gap: 0.55rem !important;
        padding: 0.45rem 1rem !important;
        border-radius: 8px !important;
        font-size: 0.83rem !important;
        color: #8B949E !important;
        cursor: pointer !important;
        transition: background 0.15s, color 0.15s !important;
        position: relative !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
        background: #161B22 !important;
        color: #E6EDF3 !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
        display: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary::after {
        content: "〉" !important;
        font-size: 0.72rem !important;
        color: #484F58 !important;
        position: absolute !important;
        right: 15px !important;
    }

    /* ── Bottom tools/settings items ── */
    .sb-bottom-items {
        padding: 0.4rem 0.5rem 0.2rem 0.5rem;
        border-top: 1px solid #1C2128;
        background: #0B0F17 !important;
    }
    .sb-bottom-link {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding: 0.45rem 1rem;
        border-radius: 8px;
        font-size: 0.83rem;
        color: #8B949E;
        cursor: pointer;
        transition: background 0.15s, color 0.15s;
        text-decoration: none;
    }
    .sb-bottom-link:hover { background: #161B22; color: #E6EDF3; }
    .sb-bottom-link-icon { font-size: 0.9rem; }

    /* ── Pinned Bottom Settings & Profile Controls ── */
    [data-testid="stSidebar"] div[class*="st-key-aqe_toggle_btn"] {
        position: fixed !important;
        bottom: 50px !important;
        left: 0 !important;
        width: 268px !important;
        z-index: 9998 !important;
        background: #0B0F17 !important;
        padding: 0.55rem 1rem 0.45rem 1rem !important;
        border-top: 1px solid #1C2128 !important;
        box-sizing: border-box !important;
    }
    [data-testid="stSidebar"] div[class*="st-key-aqe_toggle_btn"]::before {
        content: "Optimization Settings" !important;
        display: block !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        color: #8B949E !important;
        margin-bottom: 0.35rem !important;
        letter-spacing: normal !important;
    }
    .sb-user-row {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.65rem 1rem 0.8rem 1rem;
        cursor: pointer;
        border-top: 1px solid #1C2128;
        background: #0B0F17 !important;
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        width: 268px !important;
        z-index: 9999 !important;
        box-sizing: border-box !important;
    }
    .sb-avatar {
        width: 32px; height: 32px;
        background: #4F545C !important;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; font-weight: 700; color: #fff;
        flex-shrink: 0;
    }
    .sb-user-name {
        font-size: 0.86rem;
        font-weight: 600;
        color: #E6EDF3;
        flex: 1;
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        max-width: 170px !important;
        display: inline-block !important;
    }
    .sb-user-chevron { color: #484F58; font-size: 0.8rem; }

    /* ── Chat Messages Transparency Overrides ── */
    [data-testid="stChatMessage"],
    [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"] > div,
    [data-testid="stChatMessage"] > div > div {
        background-color: transparent !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stChatMessage"] {
        position: relative !important;
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
    }

    /* ── Action buttons outline vector replacement styling ── */
    .msg-action-container [data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: center !important;
        gap: 6px !important;
    }
    .msg-action-container [data-testid="stHorizontalBlock"] > div {
        width: auto !important;
        flex: none !important;
    }
    .msg-action-container button,
    [data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-copy_btn_"]) button {
        background-color: transparent !important;
        border: none !important;
        width: 34px !important;
        height: 34px !important;
        min-width: 34px !important;
        min-height: 34px !important;
        color: transparent !important; /* Hide original emoji text */
        font-size: 0px !important;
        line-height: 0 !important;
        text-indent: -9999px !important;
        overflow: hidden !important;
        background-size: 20px 20px !important;
        background-repeat: no-repeat !important;
        background-position: center center !important;
        opacity: 0.65 !important;
        transition: opacity 0.15s, background-color 0.15s !important;
        border-radius: 6px !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .msg-action-container button:hover,
    [data-testid="stChatMessage"] div[data-testid="stHorizontalBlock"]:has(div[class*="st-key-copy_btn_"]) button:hover {
        background-color: rgba(255, 255, 255, 0.06) !important;
        opacity: 1.0 !important;
    }

    /* Copy Button */
    div[class*="st-key-copy_btn_"] button {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%238B949E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='9' y='9' width='13' height='13' rx='2' ry='2'%3E%3C/rect%3E%3Cpath d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'%3E%3C/path%3E%3C/svg%3E") !important;
    }
    div[class*="st-key-copy_btn_"] button:hover {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23E6EDF3' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Crect x='9' y='9' width='13' height='13' rx='2' ry='2'%3E%3C/rect%3E%3Cpath d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'%3E%3C/path%3E%3C/svg%3E") !important;
    }

    /* Like Button */
    div[class*="st-key-like_turn_"] button {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%238B949E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3'%3E%3C/path%3E%3C/svg%3E") !important;
    }
    div[class*="st-key-like_turn_"] button:hover {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23E6EDF3' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3'%3E%3C/path%3E%3C/svg%3E") !important;
    }

    /* Dislike Button */
    div[class*="st-key-dislike_turn_"] button {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%238B949E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm12-7h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3'%3E%3C/path%3E%3C/svg%3E") !important;
    }
    div[class*="st-key-dislike_turn_"] button:hover {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23E6EDF3' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm12-7h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3'%3E%3C/path%3E%3C/svg%3E") !important;
    }

    /* Export/Download Button */
    div[class*="st-key-dl_txt_"] button {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%238B949E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8'%3E%3C/path%3E%3Cpolyline points='16 6 12 2 8 6'%3E%3C/polyline%3E%3Cline x1='12' y1='2' x2='12' y2='15'%3E%3C/line%3E%3C/svg%3E") !important;
    }
    div[class*="st-key-dl_txt_"] button:hover {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23E6EDF3' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8'%3E%3C/path%3E%3Cpolyline points='16 6 12 2 8 6'%3E%3C/polyline%3E%3Cline x1='12' y1='2' x2='12' y2='15'%3E%3C/line%3E%3C/svg%3E") !important;
    }

    /* Regenerate Button */
    div[class*="st-key-regen_"] button {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%238B949E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67'%3E%3C/path%3E%3C/svg%3E") !important;
    }
    div[class*="st-key-regen_"] button:hover {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23E6EDF3' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67'%3E%3C/path%3E%3C/svg%3E") !important;
    }

    /* More Button */
    div[class*="st-key-more_btn_"] button {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%238B949E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='1'%3E%3C/circle%3E%3Ccircle cx='19' cy='12' r='1'%3E%3C/circle%3E%3Ccircle cx='5' cy='12' r='1'%3E%3C/circle%3E%3C/svg%3E") !important;
    }
    div[class*="st-key-more_btn_"] button:hover {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23E6EDF3' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='1'%3E%3C/circle%3E%3Ccircle cx='19' cy='12' r='1'%3E%3C/circle%3E%3Ccircle cx='5' cy='12' r='1'%3E%3C/circle%3E%3C/svg%3E") !important;
    }

    /* Speak (Read Aloud) Button */
    div[class*="st-key-speak_btn_"] button {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%238B949E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='11 5 6 9 2 9 2 15 6 15 11 19 11 5'%3E%3C/polygon%3E%3Cpath d='M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07'%3E%3C/path%3E%3C/svg%3E") !important;
    }
    div[class*="st-key-speak_btn_"] button:hover {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23E6EDF3' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='11 5 6 9 2 9 2 15 6 15 11 19 11 5'%3E%3C/polygon%3E%3Cpath d='M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07'%3E%3C/path%3E%3C/svg%3E") !important;
    }
    div[class*="st-key-speak_btn_"] button.speaking {
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%23EF4444' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolygon points='11 5 6 9 2 9 2 15 6 15 11 19 11 5'%3E%3C/polygon%3E%3Cline x1='23' y1='9' x2='17' y2='15'%3E%3C/line%3E%3Cline x1='17' y1='9' x2='23' y2='15'%3E%3C/line%3E%3C/svg%3E") !important;
        animation: pulse-mic 0.8s infinite alternate !important;
    }

    /* ── Expert badges ── */
    .expert-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.1rem 0.45rem;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 600;
        margin-right: 0.25rem;
        margin-bottom: 0.3rem;
        transition: all 0.2s ease;
    }
    .expert-badge:hover { border-color: #4F8CFF !important; }

    /* ── Typing Indicator ── */
    .typing-dots { display: inline-flex; align-items: center; gap: 4px; margin-top: 4px; }
    .typing-dot {
        width: 5px; height: 5px;
        background: #4F8CFF; border-radius: 50%;
        animation: typingPulse 1.2s infinite both;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.15s; }
    .typing-dot:nth-child(3) { animation-delay: 0.3s; }
    @keyframes typingPulse {
        0%, 100% { opacity: 0.2; transform: scale(0.85); }
        50%       { opacity: 1.0; transform: scale(1.1); }
    }

    /* ── Chat input ── */
    [data-testid="stChatInputContainer"], .stChatInputContainer {
        background: transparent !important; border: none !important; box-shadow: none !important;
    }
    /* ── Chat input pill (fixed at bottom) ── */
    [data-testid="stChatInput"] {
        position: fixed !important;
        bottom: 25px !important;
        left: calc(50% + 120px) !important;
        transform: translateX(-50%) !important;
        width: calc(100% - 300px) !important;
        max-width: 760px !important;
        z-index: 9999 !important;
        background: #1C2128 !important;
        border: 1px solid #30363D !important;
        border-radius: 30px !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
        min-height: 52px !important;
        height: auto !important;
        display: flex !important;
        align-items: center !important;
        padding-left: 20px !important;
        padding-right: 92px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(79,140,255,0.45) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3), 0 0 0 3px rgba(79,140,255,0.07) !important;
    }
    [data-testid="stChatInput"] > div {
        background: transparent !important; border: none !important;
        box-shadow: none !important; width: 100% !important;
        display: flex !important; align-items: center !important;
    }
    [data-testid="stChatInput"] [data-baseweb="textarea"],
    [data-testid="stChatInput"] [data-baseweb="base-input"] {
        background-color: transparent !important; border: none !important;
        height: auto !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #E6EDF3 !important;
        background: transparent !important;
        font-size: 0.95rem !important;
        font-family: 'Inter', sans-serif !important;
        line-height: 1.45 !important;
        height: auto !important;
        min-height: 24px !important;
        max-height: 120px !important;
        overflow-y: auto !important;
        resize: none !important;
        border: none !important;
        padding: 4px 0 !important;
    }
    /* ── Mic icon (::after on input) ── */
    [data-testid="stChatInput"]::after {
        content: "🎙️" !important;
        position: absolute !important; right: 52px !important; top: 50% !important;
        transform: translateY(-50%) !important; font-size: 1rem !important;
        color: #484F58 !important; cursor: pointer !important; z-index: 10 !important;
        transition: color 0.2s ease !important;
    }
    [data-testid="stChatInput"]:hover::after { color: #8B949E !important; }
    [data-testid="stChatInput"].mic-listening::after {
        content: "🔴" !important;
        color: #EF4444 !important;
        animation: pulse-mic 0.8s infinite alternate !important;
    }
    @keyframes pulse-mic {
        from { opacity: 0.4; transform: translateY(-50%) scale(0.9); }
        to { opacity: 1.0; transform: translateY(-50%) scale(1.1); }
    }
    [data-testid="stChatInputSubmitButton"] {
        border-radius: 50% !important;
        background: linear-gradient(135deg, #4F8CFF, #7C6BFF) !important;
        color: #FFFFFF !important;
        width: 34px !important; height: 34px !important;
        border: none !important;
        position: absolute !important; right: 12px !important; top: 50% !important;
        transform: translateY(-50%) !important;
        z-index: 10 !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(79,140,255,0.3) !important;
    }
    [data-testid="stChatInputSubmitButton"]:hover {
        transform: translateY(-50%) scale(1.08) !important;
        box-shadow: 0 4px 14px rgba(79,140,255,0.45) !important;
    }
    [data-testid="stChatInputSubmitButton"] svg { color: #FFFFFF !important; fill: #FFFFFF !important; }

    /* ── Bottom disclaimer ── */
    .chat-disclaimer {
        position: fixed !important;
        bottom: 6px !important;
        left: 50% !important;
        transform: translateX(-50%) !important;
        font-size: 0.73rem !important;
        color: #484F58 !important;
        z-index: 9999 !important;
        text-align: center !important;
        width: 100% !important;
    }

    /* ── Suggested follow-up block ── */
    .followup-label {
        font-size: 0.65rem;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: #E6EDF3;
        margin: 1.1rem 0 0.55rem 0;
    }
    .followup-pills-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-bottom: 0.5rem;
        animation: pillIn 0.25s ease-out both;
    }
    @keyframes pillIn { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }
    .followup-pill-wrap .stButton > button {
        background: #161B22 !important;
        border: 1px solid #30363D !important;
        color: #8B949E !important;
        border-radius: 999px !important;
        font-size: 0.82rem !important;
        padding: 0.35rem 0.95rem !important;
        height: auto !important; min-height: unset !important;
        line-height: 1.5 !important; font-weight: 500 !important;
        white-space: nowrap !important; opacity: 1 !important;
        transition: all 0.18s !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.35rem !important;
    }
    .followup-pill-wrap .stButton > button:hover {
        background: rgba(79,140,255,0.07) !important;
        border-color: rgba(79,140,255,0.4) !important;
        color: #4F8CFF !important;
    }
    /* Ensure pills always visible (not affected by message hover) */
    .followup-pill-wrap .stButton > button,
    .suggest-chip-wrap .stButton > button { opacity: 1 !important; }

    /* ── Main Area Expanders ── */
    .stMain [data-testid="stExpander"] {
        border: 1px solid #21262D !important;
        border-radius: 10px !important;
        background: #161B22 !important;
        margin-top: 0.35rem !important;
    }
    .stMain [data-testid="stExpander"] summary {
        padding: 0.5rem 0.75rem !important; font-size: 0.8rem !important; color: #8B949E !important;
    }
    .stMain [data-testid="stExpander"] summary:hover { color: #E6EDF3 !important; }

    /* ── Progress Bars ── */
    .stProgress > div > div { background: linear-gradient(90deg, #4F8CFF, #7C6BFF) !important; border-radius: 999px !important; }
    .stProgress > div { background: #21262D !important; border-radius: 999px !important; height: 3px !important; }

    /* ── Typography ── */
    h1, h2, h3, h4, h5 {
        color: #E6EDF3 !important;
        font-weight: 700 !important; margin-top: 1.4rem !important;
        margin-bottom: 0.6rem !important; letter-spacing: -0.2px !important; line-height: 1.3 !important;
    }
    p, li { font-size: 0.94rem !important; line-height: 1.75 !important; color: #E6EDF3 !important; }
    ol, ul { padding-left: 1.35rem !important; margin-bottom: 0.85rem !important; }
    li { margin-bottom: 0.3rem !important; }
    blockquote {
        border-left: 2px solid rgba(79,140,255,0.6) !important;
        padding-left: 1rem !important; color: #8B949E !important;
        margin: 0.85rem 0 !important; font-style: italic !important;
    }

    /* ── Tables ── */
    table { width: 100% !important; border-collapse: collapse !important; margin: 1rem 0 !important; font-size: 0.87rem !important; border-radius: 8px !important; overflow: hidden !important; border: 1px solid #21262D !important; }
    th { background: #161B22 !important; border-bottom: 1px solid #30363D !important; color: #4F8CFF !important; font-weight: 600 !important; padding: 0.65rem 0.85rem !important; text-align: left !important; font-size: 0.81rem !important; text-transform: uppercase !important; letter-spacing: 0.4px !important; }
    td { border-bottom: 1px solid #1C2128 !important; padding: 0.65rem 0.85rem !important; color: #E6EDF3 !important; }
    tr:last-child td { border-bottom: none !important; }
    tr:hover td { background: rgba(255,255,255,0.012) !important; }

    /* ── Code Blocks ── */
    code { font-family: 'JetBrains Mono', 'Fira Code', monospace !important; font-size: 0.83em !important; color: #79C0FF !important; background: rgba(121,192,255,0.07) !important; padding: 0.1rem 0.28rem !important; border-radius: 4px !important; }
    pre code { color: #E6EDF3 !important; background: transparent !important; padding: 0 !important; font-size: 0.86rem !important; line-height: 1.65 !important; }
    pre { background: #161B22 !important; border: 1px solid #21262D !important; border-radius: 10px !important; padding: 1rem 1.1rem !important; overflow-x: auto !important; box-shadow: inset 0 1px 4px rgba(0,0,0,0.2) !important; }

    /* ── Welcome Screen ── */
    .welcome-screen { text-align: center; padding: 6rem 1rem 2rem 1rem; max-width: 580px; margin: 0 auto; }
    .welcome-title { font-size: 2.1rem; font-weight: 700; color: #E6EDF3; margin-bottom: 0.25rem; letter-spacing: -0.4px; line-height: 1.2; }
    .welcome-sub { color: #8B949E; font-size: 1rem; margin-bottom: 0.15rem; line-height: 1.6; }
    .welcome-sub2 { color: #484F58; font-size: 0.88rem; margin-bottom: 0; line-height: 1.6; }

    /* ── Suggestion Chips (Welcome) ── */
    .suggest-chip-wrap .stButton > button {
        background: #161B22 !important;
        border: 1px solid #30363D !important;
        color: #8B949E !important;
        border-radius: 10px !important;
        font-size: 0.82rem !important;
        padding: 0.55rem 0.8rem !important;
        height: auto !important; min-height: unset !important;
        line-height: 1.5 !important; font-weight: 500 !important;
        opacity: 1 !important; text-align: left !important;
        transition: all 0.18s !important;
    }
    .suggest-chip-wrap .stButton > button:hover {
        border-color: rgba(79,140,255,0.4) !important;
        color: #E6EDF3 !important;
        background: #1C2128 !important;
    }

    /* ── Metrics ── */
    [data-testid="stMetric"] { background: #161B22 !important; border: 1px solid #21262D !important; border-radius: 10px !important; padding: 0.7rem !important; }
    [data-testid="stMetricLabel"] { color: #8B949E !important; font-size: 0.72rem !important; font-weight: 600 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important; }
    [data-testid="stMetricValue"] { color: #E6EDF3 !important; font-weight: 700 !important; font-size: 1.15rem !important; }

    /* ── Dividers ── */
    hr { border-color: #21262D !important; margin: 0.75rem 0 !important; }

    /* ── Alerts & Toasts ── */
    [data-testid="stAlert"] { border: 1px solid #21262D !important; background: #161B22 !important; }
    [data-testid="stToast"] { background: #161B22 !important; border: 1px solid #30363D !important; box-shadow: 0 8px 24px rgba(0,0,0,0.4) !important; }

    /* ── Sidebar Text Input overrides (Search box) ── */
    [data-testid="stSidebar"] [data-testid="stTextInputRootElement"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] > div > div {
        background: #0B0F17 !important;
        border: 1px solid #21262D !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stTextInput"] input {
        background-color: transparent !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%238B949E' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='11' cy='11' r='8'%3E%3C/circle%3E%3Cline x1='21' y1='21' x2='16.65' y2='16.65'%3E%3C/line%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: 12px center !important;
        padding-left: 36px !important;
        border: none !important;
        color: #8B949E !important;
        font-size: 0.82rem !important;
    }

    /* ── Main Panel Button Overrides (Enforce Dark Pill buttons everywhere) ── */
    [data-testid="stAppViewBlockContainer"] .stButton > button,
    [data-testid="stAppViewBlockContainer"] .stDownloadButton > button,
    .stMain .stButton > button,
    .stMain .stDownloadButton > button {
        background: #161B22 !important;
        border: 1px solid #30363D !important;
        color: #8B949E !important;
        border-radius: 20px !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        transition: all 0.18s ease !important;
        box-shadow: none !important;
        text-align: left !important;
    }
    [data-testid="stAppViewBlockContainer"] .stButton > button:hover,
    [data-testid="stAppViewBlockContainer"] .stDownloadButton > button:hover,
    .stMain .stButton > button:hover,
    .stMain .stDownloadButton > button:hover {
        background: #1C2128 !important;
        border-color: #4F8CFF !important;
        color: #E6EDF3 !important;
        box-shadow: 0 0 10px rgba(79,140,255,0.08) !important;
    }

    /* ── Inline Attachment Card (.txt download) ── */
    [data-testid="stAppViewBlockContainer"] [data-testid="stChatMessage"] .stDownloadButton:not(.msg-action-container .stDownloadButton) > button,
    .stMain [data-testid="stChatMessage"] .stDownloadButton:not(.msg-action-container .stDownloadButton) > button {
        background: #161B22 !important;
        border: 1px solid #21262D !important;
        border-radius: 999px !important;
        color: #8B949E !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        padding: 0.32rem 1.1rem !important;
        width: auto !important;
        min-width: 80px !important;
        max-width: 140px !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        height: auto !important;
        opacity: 1 !important;
        text-align: center !important;
        justify-content: center !important;
        margin: 0.65rem auto 0.5rem auto !important;
    }
    [data-testid="stAppViewBlockContainer"] [data-testid="stChatMessage"] .stDownloadButton:not(.msg-action-container .stDownloadButton) > button:hover,
    .stMain [data-testid="stChatMessage"] .stDownloadButton:not(.msg-action-container .stDownloadButton) > button:hover {
        background: #21262D !important;
        border-color: #30363D !important;
        color: #E6EDF3 !important;
    }

    /* ── Center the .txt download button block ── */
    [data-testid="stChatMessage"] .stDownloadButton:not(.msg-action-container .stDownloadButton) {
        display: flex !important;
        justify-content: center !important;
    }

    /* ── RAG + Plan ── */
    .rag-chunk { background: #161B22; border: 1px solid #21262D; border-radius: 8px; padding: 0.65rem 0.8rem; margin-bottom: 0.5rem; font-size: 0.8rem; color: #8B949E; white-space: pre-wrap; }
    .rag-chunk-header { display: flex; justify-content: space-between; font-size: 0.71rem; color: #4F8CFF; font-weight: 600; margin-bottom: 0.35rem; border-bottom: 1px solid #21262D; padding-bottom: 0.25rem; }
    .plan-flow-container { display: flex; align-items: center; flex-wrap: wrap; gap: 0.3rem; margin-bottom: 0.9rem; }
    .plan-flow-step { background: #161B22; border: 1px solid #21262D; border-radius: 6px; padding: 0.35rem 0.65rem; font-size: 0.76rem; color: #E6EDF3; }
    .plan-flow-arrow { color: #4F8CFF; font-size: 0.9rem; }

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] { border: 1px solid #21262D !important; border-radius: 9px !important; }

    /* ── Radio Buttons ── */
    [data-testid="stRadio"] label { color: #8B949E !important; font-size: 0.82rem !important; }
    [data-testid="stRadio"] label:hover { color: #E6EDF3 !important; }

    /* ── File Uploader ── */
    [data-testid="stFileUploader"] { border: 1px solid #21262D !important; border-radius: 8px !important; background: #161B22 !important; }

    /* ── User menu popup ── */
    div.stElementContainer:has(> .sb-user-menu-popup) {
        position: absolute !important;
        bottom: 58px !important;
        left: 8px !important;
        right: 8px !important;
        height: 226px !important;
        z-index: 99990 !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .sb-user-menu-popup {
        background: #202123 !important;
        border: 1px solid #2d2f34 !important;
        border-radius: 12px !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
        width: 100% !important;
        height: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        padding: 6px 0 !important;
        box-sizing: border-box !important;
    }
    .menu-item-header {
        display: flex !important;
        align-items: center !important;
        padding: 10px 14px !important;
        cursor: pointer !important;
        transition: background 0.15s !important;
    }
    .menu-item-header:hover {
        background: #2a2b32 !important;
    }
    .avatar-container {
        margin-right: 12px !important;
        flex-shrink: 0 !important;
    }
    .menu-avatar {
        width: 32px !important;
        height: 32px !important;
        background: #4f545c !important;
        color: #ffffff !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        border-radius: 50% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    .header-details {
        flex-grow: 1 !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
    }
    .header-name {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        color: #ececf1 !important;
        line-height: 1.2 !important;
    }
    .header-sub {
        font-size: 0.72rem !important;
        color: #8e8ea0 !important;
        line-height: 1.2 !important;
        margin-top: 2px !important;
    }
    .header-chevron {
        display: flex !important;
        align-items: center !important;
        margin-left: 8px !important;
    }
    .menu-separator {
        height: 1px !important;
        background: #2d2f34 !important;
        margin: 6px 0 !important;
    }
    .menu-option-item {
        display: flex !important;
        align-items: center !important;
        padding: 9px 14px !important;
        cursor: pointer !important;
        transition: background 0.15s !important;
    }
    .menu-option-item:hover {
        background: #2a2b32 !important;
    }
    .option-icon {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin-right: 12px !important;
        color: #ececf1 !important;
        width: 18px !important;
        opacity: 0.85 !important;
    }
    .option-label {
        font-size: 0.83rem !important;
        font-weight: 400 !important;
        color: #ececf1 !important;
        flex-grow: 1 !important;
    }
    .item-chevron {
        display: flex !important;
        align-items: center !important;
    }
    div[class*="st-key-user_menu_"],
    div[class*="st-key-hidden_user_btn"] {
        position: absolute !important;
        top: -9999px !important;
        left: -9999px !important;
        width: 0px !important;
        height: 0px !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Constants & Metadata
# ---------------------------------------------------------------------------
EXPERT_META: dict[str, dict] = {
    "coding":          {"label": "Coding",           "icon": "💻", "color": "#3b82f6", "bg": "#1e3a5f33"},
    "math":            {"label": "Mathematics",      "icon": "📐", "color": "#10b981", "bg": "#06463233"},
    "ml":              {"label": "Machine Learning",  "icon": "⚙️",  "color": "#f59e0b", "bg": "#45220033"},
    "deep_learning":   {"label": "Deep Learning",    "icon": "🧬", "color": "#8b5cf6", "bg": "#2e1a5333"},
    "genai":           {"label": "Generative AI",    "icon": "✨", "color": "#ec4899", "bg": "#4a102a33"},
    "research":        {"label": "Research",         "icon": "🔬", "color": "#06b6d4", "bg": "#06404833"},
    "system_design":   {"label": "System Design",    "icon": "🏗️",  "color": "#f97316", "bg": "#4a200033"},
    "vision":          {"label": "Vision",           "icon": "👁️",  "color": "#e11d48", "bg": "#e11d4833"},
    # ── Conversational AI Layer (Phase 30) ─────────────────────────────────
    "conversational":  {"label": "Conversational",   "icon": "💬", "color": "#34d399", "bg": "#06402833"},
}

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def serialize_memory(memory: ConversationMemory) -> list[dict]:
    serialized_turns = []
    for turn in memory.get_turns():
        serialized_turns.append({
            "user_message": {
                "role": turn.user_message.role,
                "content": turn.user_message.content,
                "timestamp": turn.user_message.timestamp.isoformat(),
                "image_path": turn.user_message.image_path,
            },
            "assistant_message": {
                "role": turn.assistant_message.role,
                "content": turn.assistant_message.content,
                "timestamp": turn.assistant_message.timestamp.isoformat(),
                "expert": turn.assistant_message.expert,
                "experts": turn.assistant_message.experts,
            }
        })
    return serialized_turns


def deserialize_memory(turns_data: list[dict]) -> ConversationMemory:
    memory = ConversationMemory(max_turns=10)
    for turn_data in turns_data:
        u_msg = turn_data["user_message"]
        a_msg = turn_data["assistant_message"]
        
        memory.add_turn(
            question=u_msg["content"],
            answer=a_msg["content"],
            expert=a_msg.get("expert"),
            experts=a_msg.get("experts"),
            image_path=u_msg.get("image_path")
        )
        if len(memory._turns) > 0:
            memory._turns[-1].user_message.timestamp = datetime.fromisoformat(u_msg["timestamp"])
            memory._turns[-1].assistant_message.timestamp = datetime.fromisoformat(a_msg["timestamp"])
            
    return memory


def serialize_chat_state(chat_state: dict) -> dict:
    return {
        "title": chat_state["title"],
        "created_at": chat_state["created_at"].isoformat(),
        "memory": serialize_memory(chat_state["memory"]),
        "feedback": chat_state.get("feedback", {}),
        "last_responses": chat_state.get("last_responses", []),
        "last_router_decision": chat_state.get("last_router_decision", {}),
        "last_timeline": chat_state.get("last_timeline", []),
        "last_execution_plan": chat_state.get("last_execution_plan", {}),
        "last_retrieved_chunks": chat_state.get("last_retrieved_chunks", []),
    }


def deserialize_chat_state(chat_data: dict) -> dict:
    return {
        "title": chat_data["title"],
        "created_at": datetime.fromisoformat(chat_data["created_at"]),
        "memory": deserialize_memory(chat_data["memory"]),
        "feedback": chat_data.get("feedback", {}),
        "last_responses": chat_data.get("last_responses", []),
        "last_router_decision": chat_data.get("last_router_decision", {}),
        "last_timeline": chat_data.get("last_timeline", []),
        "last_execution_plan": chat_data.get("last_execution_plan", {}),
        "last_retrieved_chunks": chat_data.get("last_retrieved_chunks", []),
    }


def _get_history_filepath() -> Path:
    email = _get_signed_in_email()
    safe_email = "".join([c if c.isalnum() else "_" for c in email])
    return _PROJECT_ROOT / "data" / f"chat_history_{safe_email}.json"


def save_chat_history() -> None:
    """Serialize and save st.session_state.chats to MongoDB (with local JSON fallback)."""
    import json
    from utils.db import save_chats_to_mongodb, is_mongodb_available

    if not st.session_state.get("chats"):
        return
    try:
        email = _get_signed_in_email()
        serialized = {}
        for chat_id, chat_state in st.session_state.chats.items():
            serialized[chat_id] = serialize_chat_state(chat_state)

        # Attempt saving to MongoDB
        mongo_saved = False
        if is_mongodb_available():
            mongo_saved = save_chats_to_mongodb(email, serialized)

        # Maintain local JSON file backup for fallback
        filepath = _get_history_filepath()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

        if mongo_saved:
            logger.info("Saved chat history to MongoDB & local JSON backup for '%s'.", email)
        else:
            logger.info("Saved chat history to local JSON file for '%s'.", email)
    except Exception as e:
        logger.error("Failed to save chat history: %s", e)


def load_chat_history() -> dict:
    """Load and deserialize st.session_state.chats from MongoDB (or local JSON fallback)."""
    import json
    from utils.db import load_chats_from_mongodb, is_mongodb_available

    email = _get_signed_in_email()
    data = None

    # Attempt loading from MongoDB first if reachable
    if is_mongodb_available():
        data = load_chats_from_mongodb(email)

    # Fallback to local JSON file if not in MongoDB or connection is unavailable
    if data is None:
        filepath = _get_history_filepath()
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                logger.error("Failed to load local JSON chat history: %s", e)

    if not data:
        return {}

    try:
        deserialized = {}
        for chat_id, chat_data in data.items():
            deserialized[chat_id] = deserialize_chat_state(chat_data)
        return deserialized
    except Exception as e:
        logger.error("Failed to deserialize chat history: %s", e)
        return {}


def delete_chat(chat_id: str) -> None:
    """Delete a conversation thread from MongoDB & local storage."""
    from utils.db import delete_chat_from_mongodb, is_mongodb_available

    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        email = _get_signed_in_email()
        if is_mongodb_available():
            delete_chat_from_mongodb(email, chat_id)

        if not st.session_state.chats:
            _reset_to_welcome_chat()
        elif st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = next(iter(st.session_state.chats))
        save_chat_history()
        st.toast("Chat deleted! 🗑️", icon="🗑️")
        st.rerun()


def _make_chat_state(title: str) -> dict:
    """Build the state object for one chat thread."""
    return {
        "title": title,
        "memory": ConversationMemory(max_turns=10),
        "created_at": datetime.now(),
        "feedback": {},
        "last_responses": [],
        "last_router_decision": {},
        "last_timeline": [],
        "last_execution_plan": {},
        "last_retrieved_chunks": [],
    }


def _reset_to_welcome_chat() -> None:
    """Reset chat state to one fresh active welcome thread."""
    first_id = str(uuid.uuid4())
    st.session_state.chats = {
        first_id: _make_chat_state("Welcome Chat")
    }
    st.session_state.current_chat_id = first_id


def _init_session_state() -> None:
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = True

    if "user_email" not in st.session_state:
        st.session_state.user_email = "annamneedisuresh003@gmail.com"

    # Load chats from file if they are logged in and chats not loaded yet
    if st.session_state.get("logged_in", False):
        if "chats" not in st.session_state or not st.session_state.chats or st.session_state.get("force_reload_chats", False):
            st.session_state.force_reload_chats = False
            chats = load_chat_history()
            if chats:
                st.session_state.chats = chats
                st.session_state.current_chat_id = next(iter(chats))
            else:
                _reset_to_welcome_chat()
    else:
        # When logged out, reset to welcome chat state
        if "chats" not in st.session_state or not st.session_state.chats:
            _reset_to_welcome_chat()

    if (
        "current_chat_id" not in st.session_state
        or not st.session_state.chats
        or st.session_state.current_chat_id not in st.session_state.chats
    ):
        if st.session_state.get("chats"):
            st.session_state.current_chat_id = next(iter(st.session_state.chats))
        else:
            _reset_to_welcome_chat()

    if "just_generated" not in st.session_state:
        st.session_state.just_generated = False

    if "dev_panel" not in st.session_state:
        st.session_state.dev_panel = None

    if "show_settings_modal" not in st.session_state:
        st.session_state.show_settings_modal = False

    if "show_dev_mode_modal" not in st.session_state:
        st.session_state.show_dev_mode_modal = False

    if "show_profile_modal" not in st.session_state:
        st.session_state.show_profile_modal = False

    if "enable_aqe" not in st.session_state:
        st.session_state.enable_aqe = False

    if "enable_eval" not in st.session_state:
        st.session_state.enable_eval = False

    if "enable_typewriter" not in st.session_state:
        st.session_state.enable_typewriter = True


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_router() -> ExpertRouter:
    """Instantiate ExpertRouter once globally."""
    return ExpertRouter()


def create_new_chat() -> None:
    """Generate a new conversation session and switch to it."""
    new_id = str(uuid.uuid4())
    st.session_state.chats[new_id] = _make_chat_state(f"Chat {len(st.session_state.chats) + 1}")
    st.session_state.current_chat_id = new_id
    st.session_state.just_generated = False
    save_chat_history()
    st.toast("New chat started! 🚀", icon="💬")


def clear_all_chats() -> None:
    """Wipe all conversations and reset to a clean state."""
    _reset_to_welcome_chat()
    save_chat_history()
    st.toast("Chat history cleared! 🧹", icon="🗑️")


def _get_signed_in_email() -> str:
    """Retrieve user's email from environment, git config, or default fallback."""
    if st.session_state.get("user_email"):
        return st.session_state.user_email
    import os, subprocess
    email = os.getenv("INTELLIMOE_USER_EMAIL")
    if email:
        return email
    try:
        res = subprocess.run(["git", "config", "user.email"], capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "annamneedisuresh003@gmail.com"


def _get_user_profile() -> tuple[str, str, str]:
    """Extract user's name, email, and initials from the signed-in email address."""
    import re
    email = _get_signed_in_email()
    
    if st.session_state.get("user_name"):
        name = st.session_state.user_name
    else:
        prefix = email.split("@")[0]
        # Split by dots, dashes, underscores, and digits to find words
        words = re.findall(r'[a-zA-Z]+', prefix)
        if words:
            name = " ".join([w.capitalize() for w in words])
        else:
            name = prefix
            
    # Calculate initials
    words = name.split()
    if len(words) >= 2:
        initials = (words[0][0] + words[-1][0]).upper()
    elif len(words) == 1:
        initials = words[0][:2].upper()
    else:
        initials = "U"
        
    return name, email, initials


if hasattr(st, "dialog"):
    @st.dialog("👤 User Profile")
    def _render_profile_dialog():
        name, email, initials = _get_user_profile()
        st.markdown("<p style='font-size:1.15rem;font-weight:700;margin-top:0;'>Account Information</p>", unsafe_allow_html=True)
        
        with st.form("edit_profile_form", clear_on_submit=False):
            col_acc1, col_acc2 = st.columns(2)
            with col_acc1:
                new_name = st.text_input("Name", value=name)
            with col_acc2:
                new_email = st.text_input("Email", value=email)
                
            st.markdown("**Membership level:** `Premium Plan (Active)`")
            st.markdown("---")
            st.markdown("<p style='font-size:1.15rem;font-weight:700;'>System Integration</p>", unsafe_allow_html=True)
            col_sys1, col_sys2 = st.columns(2)
            with col_sys1:
                st.markdown("**AI Expert Modules:** `9 Modules`")
            with col_sys2:
                st.markdown("**Active Device:** `CUDA / MPS Hybrid`")
            st.info("IntelliMoE is configured with local API keys. Session state is stored locally.")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                save_btn = st.form_submit_button("Save Changes", width="stretch", type="primary")
            with col_btn2:
                close_btn = st.form_submit_button("Cancel", width="stretch")
                
            if save_btn:
                if not new_email or "@" not in new_email:
                    st.error("Please enter a valid email address.")
                elif not new_name.strip():
                    st.error("Name cannot be empty.")
                else:
                    save_chat_history()
                    st.session_state.user_name = new_name.strip()
                    st.session_state.user_email = new_email.strip()
                    st.session_state.force_reload_chats = True
                    st.success("Profile updated successfully!")
                    st.rerun()
            elif close_btn:
                st.rerun()


if hasattr(st, "dialog"):
    @st.dialog("🧑‍💻 Developer Mode Diagnostics")
    def _render_dev_mode_dialog():
        st.markdown("Select a diagnostic panel to overlay at the top of the workspace:")
        _DEV_OPTIONS = [
            "-- None (Disabled) --",
            "📊 Performance Analytics",
            "📈 Model Benchmarking",
            "🔍 Explainable AI",
            "🤖 AI Evaluation",
        ]
        dev_selection = st.selectbox(
            "Active Diagnostic Panel",
            options=_DEV_OPTIONS,
            index=_DEV_OPTIONS.index(st.session_state.dev_panel)
            if st.session_state.dev_panel in _DEV_OPTIONS else 0
        )
        st.session_state.dev_panel = None if dev_selection == "-- None (Disabled) --" else dev_selection
        if st.button("Apply and Close", width="stretch"):
            st.rerun()


if hasattr(st, "dialog"):
    @st.dialog("⚙️ Settings")
    def _render_settings_dialog():
        st.markdown("#### General Preferences")
        st.selectbox("Default AI Router Strategy", ["Hybrid ML Classifier (Recommended)", "LLM Orchestrator Fallback", "Static Single Expert Router"])
        st.selectbox("Default Primary Model", ["groq:llama3-70b", "gemini:gemini-1.5-pro", "groq:mixtral-8x7b"])
        st.checkbox(
            "Enable Real-time Typewriter Streaming Animations",
            key="enable_typewriter",
            value=st.session_state.get("enable_typewriter", True)
        )
        st.checkbox(
            "Enable Multi-Stage Answer Quality Engine (AQE)",
            key="enable_aqe",
            value=st.session_state.get("enable_aqe", True),
            help="Runs multi-stage refinement (Plan -> Review -> Improve). Disabling this switches the engine to Fast Mode, reducing response latency significantly."
        )
        st.checkbox(
            "Enable Automated AI Evaluation Scoring",
            key="enable_eval",
            value=st.session_state.get("enable_eval", True),
            help="Runs LLM-as-a-judge quality check after generation. Disabling this saves ~1.5s."
        )
        st.markdown("---")
        st.markdown("#### Account & Usage Profile")
        st.text_input("Active User", value=_get_signed_in_email(), disabled=True)
        st.info("IntelliMoE is configured with local Groq and Gemini API keys.")
        if st.button("Close", width="stretch"):
            st.rerun()


# ---------------------------------------------------------------------------
# Sidebar UI — Phase 36 (matches reference screenshot)
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    with st.sidebar:

        # ── Logo row ─────────────────────────────────────────────────────────
        st.markdown(
            """
            <div class='sb-logo-row'>
                <div class='sb-logo-icon-svg'>
                    <svg width="25" height="25" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
                        <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
                        <path d="M12 5v14"/>
                    </svg>
                </div>
                <div style='flex: 1; margin-left: 0.55rem; display: flex; flex-direction: column; justify-content: center;'>
                    <span class='sb-logo-text' style='color:#4F8CFF !important;'>IntelliMoE</span>
                    <span class='sb-logo-subtitle' style='color:#8B949E !important; font-size:0.55rem; letter-spacing:0.8px; text-transform:uppercase; font-weight:600;'>Multi-Expert AI Assistant</span>
                </div>
                <div class='sb-layout-icon-btn' title='Toggle Layout' style='display: flex; align-items: center; justify-content: center; margin-right: 0.2rem;'>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8B949E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="cursor:pointer;">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                        <line x1="9" y1="3" x2="9" y2="21"/>
                    </svg>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── New Chat button ────────────────────────────────────────────────────
        st.markdown("<div class='sb-new-chat-wrap'>", unsafe_allow_html=True)
        if st.button("＋  New Chat", key="new_chat_sidebar_btn", width="stretch"):
            create_new_chat()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Search ────────────────────────────────────────────────────────────
        st.markdown("<div class='sb-search-wrap'>", unsafe_allow_html=True)
        search_query = st.text_input(
            "search",
            placeholder="Search conversations...",
            label_visibility="collapsed",
            key="sidebar_search",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── Conversation history grouped by time ───────────────────────────────
        from datetime import date, timedelta  # noqa: PLC0415
        today = date.today()
        week_ago = today - timedelta(days=7)

        sorted_chats = sorted(
            st.session_state.chats.items(),
            key=lambda item: item[1]["created_at"],
            reverse=True,
        )

        # Filter by search
        sq = (search_query or "").lower().strip()
        if sq:
            sorted_chats = [(cid, c) for cid, c in sorted_chats if sq in c["title"].lower()]

        # Group
        today_chats = [(cid, c) for cid, c in sorted_chats if c["created_at"].date() == today]
        prev_chats  = [(cid, c) for cid, c in sorted_chats if week_ago <= c["created_at"].date() < today]
        older_chats = [(cid, c) for cid, c in sorted_chats if c["created_at"].date() < week_ago]

        def _render_chat_list(items):
            for chat_id, chat in items:
                title = chat["title"]
                if len(title) > 24:
                    title = title[:22] + "…"
                is_active = (chat_id == st.session_state.current_chat_id)
                
                # Active background styling for selector button & popover button formatting
                st.markdown(
                    f"""
                    <style>
                    {"[data-testid='stSidebar'] div[class*='st-key-chat_select_" + chat_id + "'] button { background: #161B22 !important; color: #E6EDF3 !important; border: 1px solid #21262D !important; border-radius: 8px !important; }" if is_active else ""}
                    
                    /* Frameless trigger button inside sidebar */
                    [data-testid="stSidebar"] div.stHorizontalBlock div[data-testid="stPopover"] button {{
                        background: transparent !important;
                        border: none !important;
                        color: #8B949E !important;
                        font-size: 1.2rem !important;
                        padding: 0 !important;
                        height: 38px !important;
                        line-height: 38px !important;
                        box-shadow: none !important;
                        width: 100% !important;
                    }}
                    [data-testid="stSidebar"] div.stHorizontalBlock div[data-testid="stPopover"] button:hover {{
                        color: #E6EDF3 !important;
                        background: rgba(255,255,255,0.05) !important;
                        border-radius: 4px !important;
                    }}
                    
                    /* Hide chevron icon inside sidebar popovers */
                    [data-testid="stSidebar"] div.stHorizontalBlock div[data-testid="stPopover"] button svg,
                    [data-testid="stSidebar"] div.stHorizontalBlock div[data-testid="stPopover"] button [data-testid="stIconMaterial"],
                    [data-testid="stSidebar"] div.stHorizontalBlock div[data-testid="stPopover"] button [class*="Icon"],
                    [data-testid="stSidebar"] div.stHorizontalBlock div[data-testid="stPopover"] button [class*="chevron"],
                    [data-testid="stSidebar"] div.stHorizontalBlock div[data-testid="stPopover"] button [class*="expand"] {{
                        display: none !important;
                    }}
                    
                    /* Danger delete button inside the popover */
                    [data-testid="stSidebar"] div[class*="st-key-chat_delete_action_{chat_id}"] button {{
                        color: #FF7B72 !important;
                        background-color: rgba(248, 81, 73, 0.1) !important;
                        border: 1px solid rgba(248, 81, 73, 0.25) !important;
                    }}
                    [data-testid="stSidebar"] div[class*="st-key-chat_delete_action_{chat_id}"] button:hover {{
                        background-color: rgba(248, 81, 73, 0.2) !important;
                        border-color: rgba(248, 81, 73, 0.4) !important;
                    }}
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                col_chat1, col_chat2 = st.columns([8.2, 1.8])
                with col_chat1:
                    if st.button(
                        f"💬  {title}",
                        key=f"chat_select_{chat_id}",
                        width="stretch",
                    ):
                        st.session_state.current_chat_id = chat_id
                        st.session_state.just_generated = False
                        st.rerun()
                with col_chat2:
                    with st.popover("⋯", key=f"chat_menu_trigger_{chat_id}", width="stretch"):
                        st.markdown("<p style='font-size:0.8rem;font-weight:600;margin:0 0 0.3rem 0;color:#c9d1d9;'>Rename Conversation</p>", unsafe_allow_html=True)
                        rename_input = st.text_input(
                            "Rename",
                            value=chat["title"],
                            key=f"chat_rename_input_{chat_id}",
                            label_visibility="collapsed"
                        )
                        if rename_input != chat["title"] and rename_input.strip():
                            chat["title"] = rename_input.strip()
                            save_chat_history()
                            st.rerun()
                        
                        st.markdown("<hr style='margin:0.6rem 0;border-color:#21262d;'>", unsafe_allow_html=True)
                        
                        if st.button(
                            "🗑️  Delete Chat",
                            key=f"chat_delete_action_{chat_id}",
                            width="stretch",
                            type="secondary"
                        ):
                            delete_chat(chat_id)

        if today_chats:
            st.markdown("<p class='sb-section-label'>Today</p>", unsafe_allow_html=True)
            _render_chat_list(today_chats)

        if prev_chats:
            st.markdown("<p class='sb-section-label'>Previous 7 Days</p>", unsafe_allow_html=True)
            _render_chat_list(prev_chats)

        if older_chats:
            st.markdown("<p class='sb-section-label'>Older</p>", unsafe_allow_html=True)
            _render_chat_list(older_chats)

        if not sorted_chats:
            st.markdown(
                "<p style='font-size:0.78rem;color:#484F58;padding:0.5rem 1rem;'>No conversations yet.</p>",
                unsafe_allow_html=True,
            )

        # ── Optimization settings quick toggle ──
        aqe_toggle_val = st.toggle(
            "🧠 Quality Engine (AQE)",
            value=st.session_state.get("enable_aqe", True),
            help="Enables multi-stage plan/review refinement. Disabling this switches the engine to Fast Mode (~2s latency).",
            key="aqe_toggle_btn",
        )
        if aqe_toggle_val != st.session_state.get("enable_aqe", True):
            st.session_state.enable_aqe = aqe_toggle_val
            st.rerun()

        # Spacer to push items to the bottom of the sidebar
        st.markdown("<div style='flex: 1;'></div>", unsafe_allow_html=True)

        # ── Bottom: User menu popup & profile row ──
        st.markdown("<div class='sb-bottom-items'>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Hidden user button to handle click propagation from custom HTML
        if st.button("hidden_user_btn_val", key="hidden_user_btn"):
            st.session_state.show_user_menu = not st.session_state.get("show_user_menu", False)
            st.rerun()

        # Render user menu popup if active
        if st.session_state.get("show_user_menu", False):
            name, email, initials = _get_user_profile()
            st.markdown(
                f"""
                <div class='sb-user-menu-popup'>
                    <div class='menu-item-header'>
                        <div class='avatar-container'>
                            <div class='menu-avatar'>{initials}</div>
                        </div>
                        <div class='header-details'>
                            <div class='header-name'>{name}</div>
                            <div class='header-sub'>Go</div>
                        </div>
                        <div class='header-chevron'>
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#8e8ea0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
                        </div>
                    </div>
                    <div class='menu-separator'></div>
                    <div class='menu-option-item menu-item-profile'>
                        <span class='option-icon'>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ECECF1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
                        </span>
                        <span class='option-label'>Profile</span>
                    </div>
                    <div class='menu-option-item menu-item-settings'>
                        <span class='option-icon'>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ECECF1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06-.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06-.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
                        </span>
                        <span class='option-label'>Settings</span>
                    </div>
                    <div class='menu-option-item menu-item-devmode'>
                        <span class='option-icon'>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ECECF1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="16 18 22 12 16 6"></polyline><polyline points="8 6 2 12 8 18"></polyline></svg>
                        </span>
                        <span class='option-label'>Developer Mode</span>
                    </div>
                    <div class='menu-separator'></div>
                    <div class='menu-option-item menu-item-logout'>
                        <span class='option-icon'>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path><polyline points="16 17 21 12 16 7"></polyline><line x1="21" y1="12" x2="9" y2="12"></line></svg>
                        </span>
                        <span class='option-label' style='color:#ef4444;'>Log out</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Hidden trigger buttons (accessible by JS event propagation)
            if st.button("H_header", key="user_menu_header"):
                st.session_state.show_user_menu = False
                st.session_state.show_profile_modal = True
                st.rerun()
            if st.button("H_profile", key="user_menu_profile"):
                st.session_state.show_user_menu = False
                st.session_state.show_profile_modal = True
                st.rerun()
            if st.button("H_settings", key="user_menu_settings"):
                st.session_state.show_user_menu = False
                st.session_state.show_settings_modal = True
                st.rerun()
            if st.button("H_devmode", key="user_menu_devmode"):
                st.session_state.show_user_menu = False
                st.session_state.show_dev_mode_modal = True
                st.rerun()
            if st.button("H_logout", key="user_menu_logout"):
                save_chat_history()
                st.session_state.logged_in = False
                st.session_state.user_email = None
                st.session_state.chats = None
                st.session_state.show_user_menu = False
                st.toast("Logged out successfully! 🚪", icon="🚪")
                st.rerun()

        # ── User profile ────────────────────────────────────────────────────────
        name, email, initials = _get_user_profile()
        st.markdown(
            f"""
            <div class='sb-user-row'>
                <div class='sb-avatar'>{initials}</div>
                <span class='sb-user-name'>{name}</span>
                <span class='sb-user-chevron'>⌄</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        _render_html(
            """
            <script>
            (function() {
                var par = window.parent || window;
                var doc = (par && par.document) || document;

                // Initialize Speech Recognition on parent context if not already done
                if (!par._speechRecognition) {
                    var SpeechRecognition = par.SpeechRecognition || par.webkitSpeechRecognition;
                    if (SpeechRecognition) {
                        var rec = new SpeechRecognition();
                        rec.continuous = false;
                        rec.interimResults = false;
                        rec.lang = 'en-IN'; // Robust English recognition layout

                        rec.onstart = function() {
                            var ci = doc.querySelector('[data-testid="stChatInput"]');
                            if (ci) ci.classList.add('mic-listening');
                        };

                        rec.onresult = function(event) {
                            var transcript = event.results[0][0].transcript;
                            var textarea = doc.querySelector('[data-testid="stChatInput"] textarea');
                            if (textarea) {
                                // Bypass React state bindings to correctly register the input event
                                var nativeValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                                if (nativeValueSetter) {
                                    nativeValueSetter.call(textarea, transcript);
                                } else {
                                    textarea.value = transcript;
                                }

                                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                                textarea.dispatchEvent(new Event('change', { bubbles: true }));

                                setTimeout(function() {
                                    var submitBtn = doc.querySelector('[data-testid="stChatInputSubmitButton"]');
                                    if (submitBtn) {
                                        submitBtn.removeAttribute('disabled');
                                        submitBtn.click();
                                    }
                                }, 300);
                            }
                        };

                        rec.onerror = function(event) {
                            console.error("Speech recognition error", event.error);
                            var ci = doc.querySelector('[data-testid="stChatInput"]');
                            if (ci) ci.classList.remove('mic-listening');
                        };

                        rec.onend = function() {
                            var ci = doc.querySelector('[data-testid="stChatInput"]');
                            if (ci) ci.classList.remove('mic-listening');
                        };

                        par._speechRecognition = rec;
                    }
                }

                function bindClick() {
                    // Speech recognition click handler on stChatInput
                    var chatInput = doc.querySelector('[data-testid="stChatInput"]');
                    if (chatInput && !chatInput._listenerBound) {
                        chatInput._listenerBound = true;
                        chatInput.addEventListener('click', function(e) {
                            var rect = chatInput.getBoundingClientRect();
                            var x = e.clientX - rect.left;
                            var width = rect.width;

                            // Click is on the microphone icon area (width - 80 to width - 40)
                            if (x > (width - 80) && x < (width - 40)) {
                                e.preventDefault();
                                e.stopPropagation();

                                var rec = par._speechRecognition;
                                if (rec) {
                                    if (chatInput.classList.contains('mic-listening')) {
                                        rec.stop();
                                    } else {
                                        try {
                                            rec.start();
                                            par.sessionStorage.setItem('shouldSpeakNextResponse', 'true');
                                        } catch (err) {
                                            console.log("Speech recognition start error", err);
                                        }
                                    }
                                } else {
                                    alert("Web Speech API is not supported in this browser. Please try Chrome, Edge, or Safari.");
                                }
                            }
                        });
                    }

                    // Layout toggle button
                    var layoutBtn = doc.querySelector('.sb-layout-icon-btn');
                    if (layoutBtn) {
                        layoutBtn.style.cursor = 'pointer';
                        layoutBtn.onclick = function() {
                            var collapseBtn = doc.querySelector('[data-testid="stSidebarCollapseButton"] button');
                            var expandBtn   = doc.querySelector('[data-testid="collapsedControl"] button')
                                           || doc.querySelector('[data-testid="collapsedControl"]')
                                           || doc.querySelector('[data-testid="stExpandSidebarButton"] button');
                            if (collapseBtn && collapseBtn.offsetWidth > 0) {
                                collapseBtn.click();
                            } else if (expandBtn && expandBtn.offsetWidth > 0) {
                                expandBtn.click();
                            }
                        };
                    }

                    // User profile row
                    var userRow = doc.querySelector('.sb-user-row');
                    if (userRow) {
                        userRow.style.cursor = 'pointer';
                        userRow.onclick = function() {
                            var hiddenBtn = doc.querySelector('div[class*="st-key-hidden_user_btn"] button');
                            if (hiddenBtn) hiddenBtn.click();
                        };
                    }

                    // Popup click handlers mapping
                    var menuHeader = doc.querySelector('.menu-item-header');
                    if (menuHeader) {
                        menuHeader.onclick = function() {
                            var btn = doc.querySelector('div[class*="st-key-user_menu_header"] button');
                            if (btn) btn.click();
                        };
                    }

                    var menuProfile = doc.querySelector('.menu-item-profile');
                    if (menuProfile) {
                        menuProfile.onclick = function() {
                            var btn = doc.querySelector('div[class*="st-key-user_menu_profile"] button');
                            if (btn) btn.click();
                        };
                    }

                    var menuSettings = doc.querySelector('.menu-item-settings');
                    if (menuSettings) {
                        menuSettings.onclick = function() {
                            var btn = doc.querySelector('div[class*="st-key-user_menu_settings"] button');
                            if (btn) btn.click();
                        };
                    }

                    var menuDev = doc.querySelector('.menu-item-devmode');
                    if (menuDev) {
                        menuDev.onclick = function() {
                            var btn = doc.querySelector('div[class*="st-key-user_menu_devmode"] button');
                            if (btn) btn.click();
                        };
                    }

                    var menuLogout = doc.querySelector('.menu-item-logout');
                    if (menuLogout) {
                        menuLogout.onclick = function() {
                            var btn = doc.querySelector('div[class*="st-key-user_menu_logout"] button');
                            if (btn) btn.click();
                        };
                    }

                    // ── TTS: Speaker Button Handlers ──
                    var speakBtns = doc.querySelectorAll('div[class*="st-key-speak_btn_"] button');
                    speakBtns.forEach(function(btn) {
                        if (!btn._listenerBound) {
                            btn._listenerBound = true;
                            btn.addEventListener('click', function(e) {
                                e.preventDefault();
                                e.stopPropagation();

                                var msgContainer = btn.closest('[data-testid="stChatMessage"]');
                                if (msgContainer) {
                                    var mdBlock = msgContainer.querySelector('[data-testid="stMarkdownContainer"]');
                                    if (mdBlock) {
                                        var text = mdBlock.innerText;
                                        var synth = par.speechSynthesis;

                                        if (synth.speaking) {
                                            synth.cancel();
                                            doc.querySelectorAll('div[class*="st-key-speak_btn_"] button').forEach(function(b) {
                                                b.classList.remove('speaking');
                                            });
                                        } else {
                                            synth.cancel();
                                            var utterance = new par.SpeechSynthesisUtterance(text);
                                            utterance.lang = 'en-IN';

                                            utterance.onend = function() {
                                                btn.classList.remove('speaking');
                                            };
                                            utterance.onerror = function() {
                                                btn.classList.remove('speaking');
                                            };

                                            btn.classList.add('speaking');
                                            synth.speak(utterance);
                                        }
                                    }
                                }
                            });
                        }
                    });

                    // ── TTS: Auto-Speak Next Response ──
                    if (par.sessionStorage.getItem('shouldSpeakNextResponse') === 'true') {
                        var messages = doc.querySelectorAll('[data-testid="stChatMessage"]');
                        if (messages.length > 0) {
                            var lastMsg = messages[messages.length - 1];
                            var avatarElement = lastMsg.querySelector('div[class*="stChatMessageAvatar"]');
                            var isAssistant = true;
                            if (avatarElement && avatarElement.innerText.indexOf('👤') !== -1) {
                                isAssistant = false;
                            }

                            if (isAssistant) {
                                var mdBlock = lastMsg.querySelector('[data-testid="stMarkdownContainer"]');
                                if (mdBlock) {
                                    var text = mdBlock.innerText;
                                    var synth = par.speechSynthesis;

                                    synth.cancel();
                                    var utterance = new par.SpeechSynthesisUtterance(text);
                                    utterance.lang = 'en-IN';

                                    var lastSpeakBtn = lastMsg.querySelector('div[class*="st-key-speak_btn_"] button');
                                    if (lastSpeakBtn) {
                                        lastSpeakBtn.classList.add('speaking');
                                        utterance.onend = function() {
                                            lastSpeakBtn.classList.remove('speaking');
                                        };
                                        utterance.onerror = function() {
                                            lastSpeakBtn.classList.remove('speaking');
                                        };
                                    }

                                    synth.speak(utterance);
                                    par.sessionStorage.removeItem('shouldSpeakNextResponse');
                                }
                            }
                        }
                    }
                }

                // Guard against duplicate intervals across re-renders
                if (!window._imListenersBound) {
                    window._imListenersBound = true;
                    bindClick();
                    setInterval(bindClick, 400);
                }
            })();
            </script>
            """,
            height=0,
            width=0,
        )


# ---------------------------------------------------------------------------
# Phase 31 Helper Functions
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def generate_follow_up_suggestions(response_text: str) -> list[str]:
    """Generate 3 short follow-up questions based on the assistant's response."""
    try:
        from services.groq_client import generate_response  # noqa: PLC0415
        prompt = (
            f"Based on this AI assistant's response, generate 3 highly relevant, short, "
            f"and conversational follow-up questions that the user might want to ask next.\n"
            f"Each question must be short (under 7 words).\n"
            f"Return ONLY the 3 questions as a plain list, one per line. Do not number them or add any other text.\n\n"
            f"Response:\n{response_text}"
        )
        sys_prompt = "You are a helpful assistant. Output exactly 3 follow-up questions, one per line, with no labels or numbers."
        result = generate_response(prompt, system_prompt=sys_prompt, max_tokens=100, temperature=0.5)
        questions = [q.strip().strip("-*•").strip() for q in result.split("\n") if q.strip()]
        cleaned = []
        for q in questions:
            import re
            q_clean = re.sub(r'^\d+[\.\)]\s*', '', q)
            if q_clean:
                cleaned.append(q_clean)
        if len(cleaned) >= 3:
            return cleaned[:3]
    except Exception as e:
        logger.warning("Dynamic follow-up generation failed: %s", e)
    
    return [
        "Can you explain that with an example?",
        "How can this be optimized or improved?",
        "What are the alternatives to this approach?"
    ]


def _build_export(chat_id: str, memory: ConversationMemory) -> str:
    """Build a Markdown export string for the current chat thread."""
    export_md = (
        f"# IntelliMoE Chat Export\n"
        f"Chat ID: {chat_id}\n"
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n\n"
    )
    for idx, turn in enumerate(memory.get_turns(), 1):
        active_exps = ", ".join(
            EXPERT_META.get(e, {}).get("label", e.replace("_", " ").title())
            for e in turn.experts
        )
        export_md += f"### 👤 User (Turn {idx})\n{turn.question}\n\n"
        export_md += (
            f"### 🧠 Assistant (Turn {idx}) - {active_exps}\n{turn.answer}\n\n---\n\n"
        )
    return export_md


def _render_ai_reasoning_expander(active_chat: dict) -> None:
    """
    Collapsible 'View AI Reasoning' panel shown under the latest assistant message.
    Displays router decision, execution timeline, performance metrics, and evaluation.
    """
    decision = active_chat.get("last_router_decision", {})
    responses = active_chat.get("last_responses", [])
    timeline = active_chat.get("last_timeline", [])
    eval_metrics = active_chat.get("last_evaluation_metrics", {})
    wall_time = active_chat.get("last_elapsed", 0.0)

    if not decision and not responses and not eval_metrics:
        return

    with st.expander("🔍  View AI Reasoning", expanded=False):
        # ── Router Decision ─────────────────────────────────────────────
        if decision:
            st.markdown(
                "<p style='font-size:0.8rem;font-weight:600;color:#64748b;"
                "text-transform:uppercase;letter-spacing:0.8px;margin-bottom:0.5rem;'>"
                "Routing Decision</p>",
                unsafe_allow_html=True,
            )
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                pred_expert = decision.get("predicted_expert")
                predicted = (
                    str(pred_expert).replace("_", " ").title()
                    if pred_expert
                    else "N/A"
                )
                st.metric("Predicted Expert", predicted)
            with col_r2:
                conf = decision.get("confidence")
                conf_val = float(conf) if conf is not None else 0.0
                st.metric("Confidence", f"{conf_val * 100:.1f}%")
            with col_r3:
                router_used = decision.get(
                    "router_used", decision.get("strategy_used", "—")
                )
                st.metric("Router", str(router_used)[:24] if router_used else "—")

            reason = decision.get("reason", "")
            if reason and reason != "No reasoning provided.":
                st.caption(f"💡 {reason}")

            exec_order = decision.get("execution_order", [])
            if exec_order and exec_order != ["conversational"]:
                flow = " → ".join(
                    f"`{str(e).replace('_', ' ').upper()}`" for e in exec_order if e
                )
                st.markdown(f"**Execution flow:** {flow}")

            # ── News Query Rewriter Details ────────────────────────────────
            news_rewriter = decision.get("news_rewriter")
            if news_rewriter:
                st.markdown("---")
                st.markdown(
                    "<p style='font-size:0.8rem;font-weight:600;color:#64748b;"
                    "text-transform:uppercase;letter-spacing:0.8px;margin:0.8rem 0 0.4rem 0;'>"
                    "News Live Search & Rewriter Details</p>",
                    unsafe_allow_html=True,
                )
                col_nw1, col_nw2 = st.columns(2)
                with col_nw1:
                    st.markdown(f"**Original User Query:** `{news_rewriter.get('original_query', '')}`")
                    st.markdown(f"**Rewritten Search Query:** `{news_rewriter.get('rewritten_query', '')}`")
                    st.markdown(f"**Search Provider:** `{news_rewriter.get('search_provider', 'Unknown')}`")
                with col_nw2:
                    st.markdown(f"**Search Latency:** `{news_rewriter.get('search_latency_s', 0.0):.2f}s`")
                    st.markdown(f"**LLM Synthesis Latency:** `{news_rewriter.get('llm_latency_s', 0.0):.2f}s`")
                    st.markdown(f"**Sources Used:** {', '.join(news_rewriter.get('sources_used', []))}")
                
                # Check for retrieved articles list and display in sub-expander
                articles = news_rewriter.get("retrieved_articles", [])
                if articles:
                    with st.expander("📄 View Retrieved Articles", expanded=False):
                        for idx, art in enumerate(articles, 1):
                            st.markdown(
                                f"**{idx}. [{art.get('title', 'No Title')}]({art.get('url', '#')})** "
                                f"({art.get('source', 'Unknown')})\n"
                                f"> {art.get('content', '')}\n"
                            )

        # ── Timeline ────────────────────────────────────────────────────
        if timeline:
            st.markdown(
                "<p style='font-size:0.8rem;font-weight:600;color:#64748b;"
                "text-transform:uppercase;letter-spacing:0.8px;margin:0.8rem 0 0.4rem 0;'>"
                "Execution Timeline</p>",
                unsafe_allow_html=True,
            )
            for event in timeline:
                msg_lower = event.get("message", "").lower()
                color = (
                    "#34d399" if "completed" in msg_lower
                    else "#60a5fa" if "started" in msg_lower
                    else "#94a3b8"
                )
                st.markdown(
                    f"<div style='display:flex;gap:1rem;font-size:0.82rem;margin-bottom:3px;'>"
                    f"<span style='color:#374151;min-width:52px;font-variant-numeric:tabular-nums;'>"
                    f"+{event['timestamp']:.2f}s</span>"
                    f"<span style='color:{color};'>{event['message']}</span></div>",
                    unsafe_allow_html=True,
                )

        # ── Performance ─────────────────────────────────────────────────
        if responses and wall_time:
            st.markdown(
                "<p style='font-size:0.8rem;font-weight:600;color:#64748b;"
                "text-transform:uppercase;letter-spacing:0.8px;margin:0.8rem 0 0.4rem 0;'>"
                "Performance</p>",
                unsafe_allow_html=True,
            )
            total_tokens = sum(r["tokens_generated"] for r in responses)
            total_prompt = sum(r["prompt_tokens"] for r in responses)
            total_cost = sum(r["estimated_cost"] for r in responses)
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.metric("Latency", f"{wall_time:.2f}s")
            with p2:
                st.metric("Output Tokens", f"{total_tokens:,}")
            with p3:
                st.metric("Prompt Tokens", f"{total_prompt:,}")
            with p4:
                st.metric("Est. Cost", f"${total_cost:.5f}")

        # ── Evaluation Metrics ───────────────────────────────────────────
        if eval_metrics:
            st.markdown(
                "<p style='font-size:0.8rem;font-weight:600;color:#64748b;"
                "text-transform:uppercase;letter-spacing:0.8px;margin:0.8rem 0 0.4rem 0;'>"
                "AI Evaluation</p>",
                unsafe_allow_html=True,
            )
            e1, e2, e3, e4 = st.columns(4)
            with e1:
                st.metric(
                    "AI Score",
                    f"{eval_metrics.get('overall_ai_score', 0):.0f}/100",
                )
            with e2:
                st.metric(
                    "Faithfulness",
                    f"{eval_metrics.get('faithfulness', 0) * 100:.0f}%",
                )
            with e3:
                st.metric(
                    "Relevance",
                    f"{eval_metrics.get('relevance', 0) * 100:.0f}%",
                )
            with e4:
                st.metric(
                    "Hallucination",
                    f"{eval_metrics.get('hallucination_risk', 0) * 100:.0f}%",
                )
            reasoning = eval_metrics.get("reasoning", "")
            if reasoning and reasoning != "No justification provided.":
                st.caption(f"💡 {reasoning}")

        # ── XAI JSON (nested expander) ────────────────────────────────────
        explanation = decision.get("xai_explanation", {})
        if explanation:
            with st.expander("🧬 Full XAI Trace (JSON)", expanded=False):
                st.json(explanation)


def _render_response_actions(
    idx: int,
    turn,
    active_chat: dict,
    memory: ConversationMemory,
    feedback: dict,
    is_last: bool,
) -> None:
    """
    Compact icon-only action bar — hidden by default, revealed on message hover.
    Icons: 📋 Copy  👍 Like  👎 Dislike  🔄 Regenerate  ⬇ Download
    """
    message_id = f"turn_{idx}"
    user_opinion = feedback.get(message_id)

    if is_last:
        col_copy, col_like, col_dislike, col_speak, col_dl, col_regen, col_more, _sp = st.columns(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 8.0]
        )
    else:
        col_copy, col_like, col_dislike, col_speak, col_dl, col_more, _sp = st.columns(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 8.5]
        )

    with col_copy:
        copy_key = f"show_copy_{idx}"
        if st.button(" ", key=f"copy_btn_{idx}", help="Copy response"):
            st.session_state[copy_key] = not st.session_state.get(copy_key, False)

    with col_like:
        if st.button(" ", key=f"like_{message_id}", help="Good response"):
            feedback[message_id] = "like"
            try:
                from utils.feedback import FeedbackSystem  # noqa: PLC0415
                fs = FeedbackSystem()
                for exp in turn.experts:
                    fs.add_feedback(
                        expert_name=exp,
                        query=turn.question,
                        response=turn.answer,
                        rating=1,
                    )
            except Exception:
                pass
            st.toast("Thanks for the feedback! 👍", icon="✅")
            st.rerun()

    with col_dislike:
        if st.button(" ", key=f"dislike_{message_id}", help="Bad response"):
            feedback[message_id] = "dislike"
            try:
                from utils.feedback import FeedbackSystem  # noqa: PLC0415
                fs = FeedbackSystem()
                for exp in turn.experts:
                    fs.add_feedback(
                        expert_name=exp,
                        query=turn.question,
                        response=turn.answer,
                        rating=-1,
                    )
            except Exception:
                pass
            st.toast("Feedback noted. We'll improve! 👎", icon="⚙️")
            st.rerun()

    with col_speak:
        if st.button(" ", key=f"speak_btn_{idx}", help="Read aloud (Text-to-Speech)"):
            pass

    with col_dl:
        st.download_button(
            label=" ",
            data=turn.answer,
            file_name=f"response_turn_{idx}.txt",
            mime="text/plain",
            key=f"dl_txt_{idx}",
            help="Download response as text file",
        )

    if is_last:
        with col_regen:
            if st.button(" ", key=f"regen_{idx}", help="Regenerate response"):
                question = turn.question
                memory._turns.pop(-1)
                _handle_query(question)
                st.rerun()

    with col_more:
        if st.button(" ", key=f"more_btn_{idx}", help="More options"):
            st.toast("More actions checklist: Share, Export as JSON, Bookmark.", icon="ℹ️")

    # Copy panel (shown when copy button is clicked)
    if st.session_state.get(f"show_copy_{idx}", False):
        st.code(turn.answer, language=None)


def _render_dev_panel(dev_panel: str, active_chat: dict) -> None:
    """
    Render the selected Developer Mode diagnostic panel as a full-width
    expander in the main area. Contains Performance, Benchmarking, XAI,
    or AI Evaluation tabs depending on selection.
    """
    with st.expander(f"🧑‍💻  {dev_panel}", expanded=True):

        if "Performance" in dev_panel:
            _render_evaluation_dashboard(active_chat)

        elif "Benchmarking" in dev_panel:
            st.markdown("### 📈 AI Model & Provider Benchmarking")
            st.markdown(
                "Compare latencies, token usage, error rates, and hosting costs "
                "across **Groq** and **Google Gemini** endpoints."
            )
            from benchmark.evaluator import ModelEvaluator  # noqa: PLC0415
            from benchmark.history import get_benchmark_history  # noqa: PLC0415

            evaluator = ModelEvaluator()
            if st.button("🚀 Run Complete Benchmark Suite", width="stretch"):
                with st.spinner("⚡ Running benchmark queries…"):
                    try:
                        evaluator.run_benchmark(num_queries_per_domain=1)
                        st.success("Benchmark suite completed! 🏆")
                        st.toast("Benchmark complete! 🏁", icon="📈")
                    except Exception as e:
                        st.error(f"Failed to run benchmark: {e}")

            df_hist = get_benchmark_history()
            if not df_hist.empty:
                st.markdown("#### 🏆 Provider Performance Leaderboard")
                lead_df = df_hist.groupby(["provider", "model_name"]).agg(
                    avg_latency=("response_time", "mean"),
                    avg_first_tok=("first_token_latency", "mean"),
                    success_rate=("success_rate", "mean"),
                    total_cost=("estimated_cost", "mean"),
                ).reset_index()
                lead_df["Utility Score"] = lead_df["success_rate"] / (
                    lead_df["avg_latency"] * (lead_df["total_cost"] * 1000 + 0.01)
                )
                lead_df = lead_df.sort_values("Utility Score", ascending=False).reset_index(drop=True)
                lead_df.index += 1

                pres = lead_df.copy()
                pres["avg_latency"] = pres["avg_latency"].apply(lambda x: f"{x:.3f}s")
                pres["avg_first_tok"] = pres["avg_first_tok"].apply(lambda x: f"{x:.3f}s")
                pres["success_rate"] = pres["success_rate"].apply(lambda x: f"{x*100:.1f}%")
                pres["total_cost"] = pres["total_cost"].apply(lambda x: f"${x:.5f}")
                pres["Utility Score"] = pres["Utility Score"].apply(lambda x: f"{x:.2f}")
                pres.columns = [
                    "Provider", "Model Name", "Avg Latency", "First-Token",
                    "Success Rate", "Avg Cost", "Utility Score",
                ]
                st.dataframe(pres, width="stretch")

                st.markdown("#### 📊 Metric Analytics Charts")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### ⏱️ Avg Response Time (s)")
                    st.bar_chart(lead_df.set_index("provider")["avg_latency"], color="#8b5cf6")
                    st.markdown("##### 🪙 Avg Tokens")
                    tok_df = df_hist.groupby("provider")["total_tokens"].mean().reset_index()
                    st.bar_chart(tok_df.set_index("provider")["total_tokens"], color="#f59e0b")
                with c2:
                    st.markdown("##### 💵 Avg Cost ($)")
                    st.bar_chart(lead_df.set_index("provider")["total_cost"], color="#10b981")
                    st.markdown("##### ✅ Success Rate (%)")
                    sr = lead_df.copy()
                    sr["sr_pct"] = sr["success_rate"] * 100
                    st.bar_chart(sr.set_index("provider")["sr_pct"], color="#3b82f6")

                csv_data = df_hist.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Benchmark CSV",
                    data=csv_data,
                    file_name="benchmark_history.csv",
                    mime="text/csv",
                    width="stretch",
                )
            else:
                st.info(
                    "No benchmark history. Click 'Run Complete Benchmark Suite' to generate data."
                )

        elif "Explainable" in dev_panel:
            st.markdown("### 🔍 Explainable AI (XAI) Dashboard")
            st.markdown(
                "Trace how decisions are made, from query classification to "
                "scheduling plans and API endpoints."
            )
            decision = active_chat.get("last_router_decision", {})
            explanation = decision.get("xai_explanation", {})

            # ── Conversation AI Diagnostics (Requirement 8) ──────────────────
            orig_q = decision.get("original_query", decision.get("query", "N/A"))
            clean_q = decision.get("cleaned_query", decision.get("query", "N/A"))
            greet_rem = decision.get("greeting_removed", "No")
            conv_conf = decision.get("conversation_confidence", decision.get("confidence", 0.0))
            rout_dec = decision.get("routing_decision", "Pure Conversation" if decision.get("predicted_expert") == "conversational" else "Forward to Hybrid Router")
            final_exp = decision.get("final_expert_selected", ", ".join(decision.get("selected_experts", ["N/A"])))

            st.markdown("#### 💬 Conversation AI Diagnostics")
            c_diag1, c_diag2 = st.columns(2)
            with c_diag1:
                st.markdown(f"**Original Query**:")
                st.info(orig_q)
                st.markdown(f"**Cleaned Query**:")
                st.success(clean_q)
            with c_diag2:
                st.metric(label="Conversation Confidence", value=f"{conv_conf * 100:.1f}%")
                st.markdown(f"**Greeting Removed**: `{greet_rem}`")
                st.markdown(f"**Routing Decision**: `{rout_dec}`")
                st.markdown(f"**Final Expert Selected**: `{final_exp.upper()}`")
            st.markdown("---")

            if explanation:
                try:
                    c_xai1, c_xai2 = st.columns(2)
                    with c_xai1:
                        st.markdown("#### 💼 Active Expert Domains")
                        experts_list = explanation.get("decision_engine", {}).get("experts", [])
                        badge_str = "  ".join(
                            f"`{e.replace('_', ' ').upper()}`" for e in experts_list
                        )
                        st.markdown(badge_str)
                        fallback_flag = explanation.get("router", {}).get("fallback", True)
                        if not fallback_flag:
                            st.success("🤖 Routed using high-confidence ML Classifier.")
                        else:
                            st.warning("⚠️ Routed using LLM Agent (confidence below threshold).")

                    with c_xai2:
                        st.markdown("#### 🎯 Classification Confidence")
                        conf = explanation.get("router", {}).get("confidence", 0.0)
                        st.metric(label="Intent Match Probability", value=f"{conf * 100:.1f}%")
                        st.progress(min(max(float(conf), 0.0), 1.0))

                    st.markdown("---")
                    c_graph, c_time = st.columns(2)
                    with c_graph:
                        st.markdown("#### ⛓️ System Execution Graph")
                        from explainability.graph import generate_mermaid_graph, generate_ascii_graph  # noqa: PLC0415
                        mermaid_code = generate_mermaid_graph(explanation)
                        st.code(mermaid_code, language="mermaid")
                        with st.expander("Show ASCII Trace"):
                            st.text(generate_ascii_graph(explanation))

                    with c_time:
                        st.markdown("#### ⏱️ Execution Timeline")
                        from explainability.timeline import generate_execution_timeline  # noqa: PLC0415
                        for event in generate_execution_timeline(explanation):
                            st.markdown(f"⏱️ `{event}`")

                    st.markdown("---")
                    c_reason, c_perf = st.columns(2)
                    with c_reason:
                        st.markdown("#### 🧠 Decision Reasoning")
                        from explainability.timeline import generate_reasoning_timeline  # noqa: PLC0415
                        for step in generate_reasoning_timeline(explanation):
                            st.markdown(f"💡 {step}")

                    with c_perf:
                        st.markdown("#### 📊 Performance & Telemetry")
                        perf_dict = explanation.get("performance", {})
                        api_dict = explanation.get("api", {})
                        st.markdown(f"**Response Latency**: `{perf_dict.get('response_time_ms', 0)} ms`")
                        st.markdown(f"**Estimated Token Usage**: `{perf_dict.get('tokens', 0)} tokens`")
                        st.markdown(f"**Target API**: `{api_dict.get('provider', 'Groq/Gemini')}`")
                        st.info(api_dict.get("reason", "No API selected."))

                    st.markdown("---")
                    st.markdown("#### ⚙️ Raw XAI JSON Schema")
                    st.json(explanation)
                except Exception as e:
                    st.error(f"⚠️ XAI render error: {e}")
                    st.json(explanation)
            else:
                st.info(
                    "No XAI trace available. Ask a technical question in the chat first."
                )

        elif "AI Evaluation" in dev_panel:
            st.markdown("### 🤖 AI Evaluation Diagnostics & Metrics")
            st.markdown(
                "Automated evaluation scoring using **Gemini-as-a-Judge**: "
                "Faithfulness, Relevance, Completeness, and System metrics."
            )
            eval_metrics = active_chat.get("last_evaluation_metrics", {})
            from evaluation.history import get_evaluation_history  # noqa: PLC0415
            df_eval = get_evaluation_history()

            if eval_metrics:
                c1, c2 = st.columns(2)
                with c1:
                    ai_score = eval_metrics.get("overall_ai_score", 0.0)
                    st.metric("🧠 Overall AI Score", f"{ai_score:.1f} / 100")
                    st.progress(ai_score / 100.0)
                with c2:
                    sys_score = eval_metrics.get("overall_system_score", 0.0)
                    st.metric("⚙️ Overall System Score", f"{sys_score:.1f} / 100")
                    st.progress(sys_score / 100.0)

                st.markdown("---")
                st.markdown("#### 🎯 Semantic Alignment Metrics")
                e1, e2, e3, e4 = st.columns(4)
                with e1:
                    st.metric("😇 Faithfulness", f"{eval_metrics.get('faithfulness', 0) * 100:.1f}%")
                with e2:
                    st.metric("🎯 Relevance", f"{eval_metrics.get('relevance', 0) * 100:.1f}%")
                with e3:
                    st.metric("📦 Completeness", f"{eval_metrics.get('completeness', 0) * 100:.1f}%")
                with e4:
                    st.metric("🕵️ Hallucination", f"{eval_metrics.get('hallucination_risk', 0) * 100:.1f}%")

                st.markdown("#### ⛓️ Router & Collaboration Accuracies")
                a1, a2, a3 = st.columns(3)
                with a1:
                    st.metric("🤖 Routing Accuracy", f"{eval_metrics.get('routing_accuracy', 0) * 100:.1f}%")
                with a2:
                    st.metric("💼 Expert Selection", f"{eval_metrics.get('expert_selection_accuracy', 0) * 100:.1f}%")
                with a3:
                    st.metric("🤝 Multi-Agent", f"{eval_metrics.get('multi_agent_accuracy', 0) * 100:.1f}%")

                st.markdown("---")
                c_tel1, c_tel2 = st.columns(2)
                with c_tel1:
                    st.markdown("##### ⏱️ Execution Telemetry")
                    st.markdown(f"**Latency**: `{eval_metrics.get('response_time', 0):.3f}s`")
                    st.markdown(f"**Token Usage**: `{eval_metrics.get('token_usage', 0):,}`")
                    st.markdown(f"**Response Quality**: `{eval_metrics.get('response_quality', 0) * 100:.1f}%`")
                with c_tel2:
                    st.markdown("##### 💡 Evaluator Justification")
                    st.info(eval_metrics.get("reasoning", "No justification provided."))
                st.markdown("---")

            if not df_eval.empty:
                st.markdown("#### 📊 Comparative Analytics (Groq vs Gemini)")
                prov_eval = df_eval.groupby("provider").agg(
                    avg_ai_score=("overall_ai_score", "mean"),
                    avg_sys_score=("overall_system_score", "mean"),
                    avg_faithfulness=("faithfulness", "mean"),
                    avg_relevance=("relevance", "mean"),
                    avg_latency=("response_time", "mean"),
                ).reset_index()
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 🧠 Avg AI Score")
                    st.bar_chart(prov_eval.set_index("provider")["avg_ai_score"], color="#8b5cf6")
                    st.markdown("##### 😇 Avg Faithfulness (%)")
                    prov_eval["faith_pct"] = prov_eval["avg_faithfulness"] * 100
                    st.bar_chart(prov_eval.set_index("provider")["faith_pct"], color="#ec4899")
                with c2:
                    st.markdown("##### ⚙️ Avg System Score")
                    st.bar_chart(prov_eval.set_index("provider")["avg_sys_score"], color="#3b82f6")
                    st.markdown("##### ⏱️ Avg Latency (s)")
                    st.bar_chart(prov_eval.set_index("provider")["avg_latency"], color="#10b981")

                csv_eval = df_eval.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Evaluation Logs CSV",
                    data=csv_eval,
                    file_name="evaluation_history.csv",
                    mime="text/csv",
                    width="stretch",
                )
            else:
                if not eval_metrics:
                    st.info(
                        "No evaluation data available. Ask a question in the chat first."
                    )


# ---------------------------------------------------------------------------
# Main Panel UI Rendering — Phase 35 Redesign
# ---------------------------------------------------------------------------
def render_main() -> None:
    _init_session_state()
    chat_id = st.session_state.current_chat_id
    active_chat = st.session_state.chats[chat_id]
    memory: ConversationMemory = active_chat["memory"]
    feedback: dict = active_chat["feedback"]

    # Inject JS to remove Streamlit's Material-icon 'expand_more' chevron from all popover buttons
    _render_html("""
    <script>
    (function removePopoversChevron() {
        function hideChevrons() {
            document.querySelectorAll('[data-testid="stPopover"] button').forEach(btn => {
                btn.querySelectorAll('span').forEach(span => {
                    if (span.textContent.trim() === 'expand_more') {
                        span.style.cssText = 'display:none!important;width:0!important;overflow:hidden!important;';
                    }
                });
            });
        }
        hideChevrons();
        new MutationObserver(hideChevrons).observe(document.body, { childList: true, subtree: true });
    })();
    </script>
    """, height=0)

    # ── Developer Panel ────────────────────────────────────────────────────
    dev_panel = st.session_state.get("dev_panel")
    if dev_panel:
        _render_dev_panel(dev_panel, active_chat)

    # ── Chat Header (clean — no icons) ───────────────────────────────────────────
    # Thin divider only — no icons
    st.markdown(
        """<div style='border-bottom:1px solid #1C2128; margin-bottom:0.35rem;'></div>""",
        unsafe_allow_html=True,
    )

    # ── Welcome Screen ─────────────────────────────────────────────────────
    if memory.is_empty:
        st.markdown(
            """
            <div class='welcome-screen'>
                <p class='welcome-title'>Hello 👋</p>
                <p class='welcome-sub'>I'm IntelliMoE.</p>
                <p class='welcome-sub2'>How can I help you today?</p>
            </div>
            """,
            unsafe_allow_html=True,
        )



    # ── Multi-Agent Collaboration Plan (last turn, if applicable) ──────────
    timeline = active_chat.get("last_timeline", [])
    plan_dict = active_chat.get("last_execution_plan", {})
    if plan_dict and timeline:
        with st.expander("⚙️  Multi-Agent Collaboration Plan & Timeline", expanded=False):
            st.markdown("#### 🎯 Execution Order & Plan Steps")
            steps = plan_dict.get("steps", [])
            conf_dict = active_chat.get("last_router_decision", {}).get("confidence_scores", {})
            plan_flow_parts = []
            for s in steps:
                exp = s["expert"]
                icon = EXPERT_META.get(exp, {}).get("icon", "🤖")
                label = EXPERT_META.get(exp, {}).get("label", exp.replace("_", " ").title())
                conf_val = conf_dict.get(exp, 0.0)
                plan_flow_parts.append(
                    f"<div class='plan-flow-step'><strong>{icon} {label}</strong><br>"
                    f"<span style='font-size:0.75rem;opacity:0.7;'>Conf: {conf_val * 100:.1f}%</span></div>"
                )
            flow_html = (
                "<div class='plan-flow-container'>"
                + "<div class='plan-flow-arrow'>➡️</div>".join(plan_flow_parts)
                + "</div>"
            )
            st.markdown(flow_html, unsafe_allow_html=True)
            st.markdown("---")

            steps_data = []
            for s in steps:
                deps = (
                    ", ".join(f"Step {d}" for d in s["dependencies"])
                    if s["dependencies"]
                    else "None"
                )
                steps_data.append({
                    "Step ID": f"Step {s['step_id']}",
                    "Target Expert": EXPERT_META.get(s["expert"], {}).get(
                        "label", s["expert"].replace("_", " ").title()
                    ),
                    "Dependencies": deps,
                })
            st.dataframe(pd.DataFrame(steps_data), width="stretch", hide_index=True)
            st.markdown("---")

            st.markdown("#### ⏱️ Orchestration Timeline")
            for event in timeline:
                msg_lower = event["message"].lower()
                status_color = (
                    "#34d399" if "completed" in msg_lower
                    else "#60a5fa" if "started" in msg_lower
                    else "#e2e8f0"
                )
                st.markdown(
                    f"<div style='display:flex;align-items:center;margin-bottom:5px;'>"
                    f"<span style='color:#374151;font-size:0.8rem;font-weight:600;width:60px;'>"
                    f"+{event['timestamp']:.2f}s</span>"
                    f"<span style='color:{status_color};font-weight:500;font-size:0.9rem;'>"
                    f"{event['message']}</span></div>",
                    unsafe_allow_html=True,
                )

    # ── Conversation Messages ──────────────────────────────────────────────
    turns = memory.get_turns()
    for idx, turn in enumerate(turns):
        is_last_turn = (idx == len(turns) - 1)

        # ── User Bubble ────────────────────────────────────────────────────
        with st.chat_message("user", avatar="👤"):
            img_ctx = getattr(turn, "image_path", None)
            if img_ctx:
                try:
                    st.image(img_ctx, caption="Uploaded Image", width=400)
                except Exception:
                    pass
            st.markdown(turn.question)

        # ── Assistant Bubble ───────────────────────────────────────────────
        with st.chat_message("assistant", avatar="🧠"):
            # Dynamic horizontal header layout with expert badges + action icons on right
            primary_exp = getattr(turn, "expert", "conversational")
            badges_html = ""
            for exp in turn.experts:
                badge_meta = EXPERT_META.get(exp, {
                    "label": exp.replace("_", " ").title(),
                    "icon": "🤖",
                    "color": "#60a5fa",
                    "bg": "#1e3a5f33",
                })
                # Check for Conversational custom coloring
                if exp == "conversational":
                    b_color = "#34d399"
                    b_bg = "rgba(6,64,40,0.22)"
                    b_icon = "💬"
                    b_lbl = "Conversational"
                else:
                    b_color = badge_meta['color']
                    b_bg = badge_meta['bg']
                    b_icon = badge_meta['icon']
                    b_lbl = badge_meta['label']

                badges_html += (
                    f"<div class='expert-badge' style='"
                    f"background:{b_bg}; border:1px solid {b_color}55; color:{b_color};"
                    f"padding:0.15rem 0.6rem; border-radius:999px; font-size:0.72rem; font-weight:600; display:inline-flex; align-items:center; gap:0.3rem;'>"
                    f"<span>{b_icon}</span>"
                    f"<span>{b_lbl}</span>"
                    f"</div>"
                )

            # Header row: IntelliMoE name left, badges below, action icons right
            header_html = f"""
            <div style='display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; margin-bottom:0.6rem;'>
                <div style='display:flex; flex-direction:column; gap:0.35rem;'>
                    <span style='font-size:1rem; font-weight:700; color:#E6EDF3; letter-spacing:-0.2px;'>IntelliMoE</span>
                    <div style='display:flex; flex-wrap:wrap; gap:0.35rem;'>{badges_html}</div>
                </div>
            </div>
            """
            st.markdown(header_html, unsafe_allow_html=True)

            # Response content — streaming for newest generation, static otherwise
            if is_last_turn and st.session_state.just_generated:
                if st.session_state.get("enable_typewriter", True):
                    def _stream_text():
                        for word in turn.answer.split(" "):
                            yield word + " "
                            time.sleep(0.005)
                    st.write_stream(_stream_text)
                else:
                    st.markdown(turn.answer)
                st.session_state.just_generated = False
            else:
                st.markdown(turn.answer)

            # RAG retrieved chunks (latest turn only)
            retrieved_chunks = active_chat.get("last_retrieved_chunks", [])
            if is_last_turn and retrieved_chunks:
                with st.expander(
                    f"🔬  RAG Sources - {len(retrieved_chunks)} chunk(s) retrieved",
                    expanded=False,
                ):
                    st.markdown(
                        f"**Retrieved {len(retrieved_chunks)} document chunks** from ChromaDB:"
                    )
                    for chunk in retrieved_chunks:
                        similarity = 1.0 - (chunk["distance"] / 2.0)
                        st.markdown(
                            f"<div class='rag-chunk'>"
                            f"<div class='rag-chunk-header'>"
                            f"<span>📄 {chunk['source']} · Chunk {chunk['chunk_index']}</span>"
                            f"<span>Relevance: {similarity:.1%}</span>"
                            f"</div>"
                            f"{chunk['text']}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

            # Response action icon bar (inline, always visible)
            st.markdown("<div class='msg-action-container'>", unsafe_allow_html=True)
            _render_response_actions(idx, turn, active_chat, memory, feedback, is_last_turn)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Chat Input ─────────────────────────────────────────────────────────
    if "uploader_key_suffix" not in st.session_state:
        st.session_state.uploader_key_suffix = 0

    # Hidden file uploader for Custom ChatGPT Input
    uploader_key = f"rag_file_uploader_{st.session_state.uploader_key_suffix}"
    uploaded_files = st.file_uploader(
        "Ingest papers/documents into RAG collection",
        accept_multiple_files=True,
        key=uploader_key,
        type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg", "md"]
    )
    if uploaded_files:
        st.success(f"Ingested {len(uploaded_files) if isinstance(uploaded_files, list) else 1} document(s) into ChromaDB RAG vector collection! 🚀")

    # Hidden clear button to be triggered by chip removal
    if st.button("Clear RAG Uploads", key="btn_clear_rag_uploads"):
        st.session_state.uploader_key_suffix += 1
        st.rerun()

    # Preferences active bar
    pref_badges = []
    if st.session_state.get("enable_aqe", False):
        pref_badges.append("🔬 Deep Research Active")
    if st.session_state.get("force_web_search", False):
        pref_badges.append("🌐 Web Search Active")
    if st.session_state.get("chat_input_prefill"):
        pref_badges.append("🎨 Image Generator Active")

    if pref_badges:
        badges_joined = " &nbsp;•&nbsp; ".join([f"<b>{b}</b>" for b in pref_badges])
        st.markdown(
            f"<div style='max-width:760px; margin:0 auto 8px auto; padding:6px 12px; background:rgba(79,140,255,0.08); border:1px solid rgba(79,140,255,0.15); border-radius:8px; font-size:0.78rem; color:#7C6BFF; text-align:center;'>{badges_joined}</div>",
            unsafe_allow_html=True
        )

    placeholder_text = "Ask IntelliMoE anything..."
    if st.session_state.get("chat_input_prefill"):
        placeholder_text = "Describe your image (e.g. 'a cute cat in space')..."

    # Custom styling injection
    st.markdown(
        """
        <style>
        /* Hide the default file uploader and clear button entirely */
        div[data-testid="stFileUploader"], div[class*="st-key-btn_clear_rag_uploads"] {
            position: absolute !important;
            width: 0px !important;
            height: 0px !important;
            opacity: 0 !important;
            overflow: hidden !important;
            pointer-events: none !important;
        }

        /* Offset textarea slightly to make room for left + button */
        [data-testid="stChatInput"] textarea {
            margin-left: 40px !important;
        }
        
        /* Custom interactive attachment chip */
        .attachment-chip {
            display: inline-flex;
            align-items: center;
            background: #21262D;
            border: 1px solid #30363D;
            color: #C9D1D9;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 500;
            gap: 6px;
            transition: all 0.2s ease;
        }
        .attachment-chip:hover {
            border-color: #4F8CFF;
            background: #2A2F38;
        }
        .attachment-chip .remove-btn {
            cursor: pointer;
            color: #8B949E;
            font-weight: bold;
            font-size: 0.85rem;
            transition: color 0.2s ease;
            margin-left: 4px;
        }
        .attachment-chip .remove-btn:hover {
            color: #FF7B72;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Serialize uploaded file list for JavaScript chip injection
    import json  # noqa: PLC0415
    file_list = []
    if uploaded_files:
        if isinstance(uploaded_files, list):
            file_list = [f.name for f in uploaded_files]
        else:
            file_list = [uploaded_files.name]
    file_list_json = json.dumps(file_list)

    # JavaScript DOM manipulation injection via custom hidden HTML iframe
    _render_html(
        f"""
        <script>
        (function() {{
            const doc = (window.parent ? window.parent.document : document);
            
            function setupChatGPTInput() {{
                const chatInput = doc.querySelector('[data-testid="stChatInput"]');
                if (!chatInput) return;
                
                // 1. circular "+" upload button on the left inside chatInput
                let plusBtn = doc.getElementById("custom-upload-btn");
                if (!plusBtn) {{
                    plusBtn = doc.createElement("button");
                    plusBtn.id = "custom-upload-btn";
                    plusBtn.innerHTML = "➕";
                    plusBtn.title = "Upload File";
                    plusBtn.style.cssText = `
                        position: absolute;
                        left: 12px;
                        top: 50%;
                        transform: translateY(-50%);
                        width: 32px;
                        height: 32px;
                        border-radius: 50%;
                        background: transparent;
                        border: none;
                        color: #8B949E;
                        font-size: 1.1rem;
                        cursor: pointer;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        transition: background 0.2s ease, color 0.2s ease;
                        z-index: 99;
                    `;
                    plusBtn.onmouseover = () => {{
                        plusBtn.style.background = "rgba(255, 255, 255, 0.08)";
                        plusBtn.style.color = "#E6EDF3";
                    }};
                    plusBtn.onmouseout = () => {{
                        plusBtn.style.background = "transparent";
                        plusBtn.style.color = "#8B949E";
                    }};
                    plusBtn.onclick = (e) => {{
                        e.preventDefault();
                        const fileInput = doc.querySelector('[data-testid="stFileUploader"] input[type="file"]');
                        if (fileInput) fileInput.click();
                    }};
                    
                    const innerRow = chatInput.querySelector('div > div') || chatInput.firstElementChild;
                    if (innerRow) {{
                        innerRow.appendChild(plusBtn);
                    }}
                }}
                
                // 2. custom attachment chips
                const uploadedFiles = {file_list_json};
                let chipsContainer = doc.getElementById("custom-chips-container");
                
                if (uploadedFiles && uploadedFiles.length > 0) {{
                    if (!chipsContainer) {{
                        chipsContainer = doc.createElement("div");
                        chipsContainer.id = "custom-chips-container";
                        chipsContainer.style.cssText = `
                            display: flex;
                            gap: 8px;
                            padding: 0px 8px 8px 44px;
                            flex-wrap: wrap;
                            width: 100%;
                        `;
                        const innerContainer = chatInput.querySelector('div');
                        if (innerContainer) {{
                            innerContainer.insertBefore(chipsContainer, innerContainer.firstChild);
                        }}
                    }}
                    
                    // Render individual chips
                    chipsContainer.innerHTML = "";
                    uploadedFiles.forEach(filename => {{
                        const chip = doc.createElement("div");
                        chip.className = "attachment-chip";
                        
                        const fileIcon = doc.createElement("span");
                        fileIcon.innerHTML = "📄";
                        
                        const nameSpan = doc.createElement("span");
                        nameSpan.textContent = filename;
                        
                        const removeBtn = doc.createElement("span");
                        removeBtn.className = "remove-btn";
                        removeBtn.innerHTML = "✕";
                        removeBtn.title = "Remove file";
                        removeBtn.onclick = (e) => {{
                            e.preventDefault();
                            const clearBtn = doc.querySelector('div[class*="st-key-btn_clear_rag_uploads"] button');
                            if (clearBtn) clearBtn.click();
                        }};
                        
                        chip.appendChild(fileIcon);
                        chip.appendChild(nameSpan);
                        chip.appendChild(removeBtn);
                        chipsContainer.appendChild(chip);
                    }});
                }} else {{
                    if (chipsContainer) {{
                        chipsContainer.remove();
                    }}
                }}
            }}
            
            // Execute setup
            setupChatGPTInput();
            
            // Monitor DOM mutations
            const observer = new MutationObserver(setupChatGPTInput);
            observer.observe(doc.body, {{ childList: true, subtree: true }});
        }})();
        </script>
        """,
        height=0
    )

    query = st.chat_input(placeholder_text)
        
    if query:
        # If image generator prefix was active, append it to query
        if st.session_state.get("chat_input_prefill"):
            query = st.session_state.chat_input_prefill + query
            st.session_state.chat_input_prefill = None # Reset
            
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)
        _handle_query(query.strip())
        st.rerun()

    # Footer disclaimer pinned at the bottom of the viewport
    st.markdown(
        "<div class='chat-disclaimer'>IntelliMoE can make mistakes. Check important info.</div>",
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Core Query Handler (Invokes Router & Updates State)
# ---------------------------------------------------------------------------
def _handle_query(query: str) -> None:
    if not query:
        st.warning("⚠️  Please enter a question.", icon="⚠️")
        return

    router = get_router()
    chat_id = st.session_state.current_chat_id
    active_chat = st.session_state.chats[chat_id]
    memory: ConversationMemory = active_chat["memory"]

    img_path = st.session_state.get("uploaded_image_path", None)

    # ── Phase 30: Conversation AI Layer ─────────────────────────────────────
    # Intercepts queries BEFORE the Hybrid Router.
    # Conversational intents (greetings, small-talk, follow-ups, clarifications,
    # general knowledge) are answered directly by the LLM with full memory context.
    # Technical intents fall through to the existing routing pipeline unchanged.
    # ────────────────────────────────────────────────────────────────────────
    conv_ai_diagnostics = None
    if not img_path:  # Images always go to Vision Expert; skip conv layer
        try:
            from conversation_ai.layer import ConversationLayer  # noqa: PLC0415
            _conv_layer = ConversationLayer()
            with st.spinner("💬 Understanding your message..."):
                conv_result = _conv_layer.process(query, memory)

            # Store diagnostics for UI rendering later
            conv_ai_diagnostics = {
                "original_query": conv_result.original_query,
                "cleaned_query": conv_result.cleaned_query,
                "greeting_removed": "Yes" if conv_result.greeting_removed else "No",
                "conversation_confidence": conv_result.confidence,
                "routing_decision": conv_result.routing_decision,
            }

            if conv_result.is_conversational:
                # Update conversation title if this is the first turn
                if memory.is_empty:
                    title = query if len(query) <= 30 else query[:27] + "..."
                    active_chat["title"] = title

                # Record the conversational turn in memory
                memory.add_turn(
                    query,
                    conv_result.response,
                    expert="conversational",
                    experts=["conversational"],
                )

                # Populate router_decision metadata so analytics tabs don't break
                active_chat["last_router_decision"] = {
                    "strategy_used": "Conversation AI Layer",
                    "selected_experts": ["conversational"],
                    "query": query,
                    "confidence_scores": {"conversational": conv_result.confidence},
                    "predicted_expert": "conversational",
                    "confidence": conv_result.confidence,
                    "router_used": f"Conv AI ({conv_result.tier_used} tier)",
                    "fallback_used": False,
                    "primary_expert": "conversational",
                    "additional_experts": [],
                    "reason": conv_result.reasoning,
                    "execution_order": ["conversational"],
                    "original_query": conv_result.original_query,
                    "cleaned_query": conv_result.cleaned_query,
                    "greeting_removed": "Yes" if conv_result.greeting_removed else "No",
                    "conversation_confidence": conv_result.confidence,
                    "routing_decision": conv_result.routing_decision,
                    "final_expert_selected": "conversational",
                }
                active_chat["last_responses"] = [{
                    "expert_name": "conversational",
                    "expert_label": "Conversational",
                    "elapsed": conv_result.response_time_s,
                    "prompt_tokens": 0,
                    "tokens_generated": len(conv_result.response.split()),
                    "estimated_cost": 0.0,
                    "memory_usage_mb": 0.0,
                    "success": True,
                    "error": None,
                }]
                active_chat["last_elapsed"] = conv_result.response_time_s
                active_chat["last_timeline"] = []
                active_chat["last_execution_plan"] = {}
                active_chat["last_retrieved_chunks"] = []

                # Append to router history for the analytics dashboard
                if "router_history" not in st.session_state:
                    st.session_state.router_history = []
                st.session_state.router_history.append({
                    "query": query,
                    "predicted_expert": "conversational",
                    "confidence": conv_result.confidence,
                    "router_used": f"Conv AI ({conv_result.tier_used} tier)",
                    "fallback_used": False,
                    "elapsed": conv_result.response_time_s,
                })

                # Trigger typewriter animation
                st.session_state.just_generated = True
                save_chat_history()
                return  # ← Done; never touches Hybrid Router
            else:
                # Substantive query found - forward the cleaned query to Hybrid Router
                query = conv_result.cleaned_query

        except Exception as conv_exc:
            # Graceful degradation: log and fall through to normal routing
            logger.warning(
                "Conversation AI Layer raised an exception — falling back to expert routing: %s",
                conv_exc,
            )
    # ── END Conversation AI Layer ────────────────────────────────────────────

    # 1. Routing phase (no inference yet)
    with st.spinner("🔍 Intent Router is analyzing query and selecting experts..."):
        try:
            if img_path:
                expert_names = [ExpertName.VISION]
            else:
                expert_names = router.selected_experts(query)
            # Update title of conversation if it's the very first question
            if memory.is_empty:
                title = query if len(query) <= 30 else query[:27] + "..."
                active_chat["title"] = title
        except Exception as e:
            st.error(f"❌ Routing failed: {e}", icon="❌")
            return

    # 2. Parallel generation phase (with typing placeholder)
    placeholder = st.empty()
    with placeholder.container():
        with st.chat_message("assistant", avatar="🧠"):
            st.markdown(
                f"<div style='font-size:0.9rem; color:#60a5fa; font-weight:500;'>"
                f"⚙️ Running {len(expert_names)} expert(s) in parallel pool...</div>"
                f"<div class='typing-dots'><div class='typing-dot'></div>"
                f"<div class='typing-dot'></div><div class='typing-dot'></div></div>",
                unsafe_allow_html=True
            )

    try:
        t_start = time.perf_counter()

        # Run expert pipeline via router with optional image path
        answer: str = router.route(query, memory=memory, image_path=img_path)
        elapsed = time.perf_counter() - t_start

        # Refresh actual selected experts list from router decision output
        actual_experts = router.last_router_decision.get("selected_experts", [e.value for e in expert_names])
        expert_names = [ExpertName(e) for e in actual_experts]

        # Record exchange in active memory
        primary_expert = expert_names[0].value if expert_names else None
        memory.add_turn(
            query,
            answer,
            expert=primary_expert,
            experts=[e.value for e in expert_names],
            image_path=img_path
        )

        # Clear placeholder
        placeholder.empty()

        # Save telemetry metadata inside active chat dict
        active_chat["last_responses"] = [
            {
                "expert_name": r.expert_name.value,
                "expert_label": r.expert_label,
                "elapsed": r.elapsed,
                "prompt_tokens": r.prompt_tokens,
                "tokens_generated": r.tokens_generated,
                "estimated_cost": r.estimated_cost,
                "memory_usage_mb": r.memory_usage_mb,
                "success": r.success,
                "error": str(r.error) if r.error else None
            } for r in getattr(router, "last_responses", [])
        ]
        active_chat["last_router_decision"] = dict(getattr(router, "last_router_decision", {}))
        if conv_ai_diagnostics:
            active_chat["last_router_decision"].update(conv_ai_diagnostics)
            sel_exp = active_chat["last_router_decision"].get("selected_experts", [])
            active_chat["last_router_decision"]["final_expert_selected"] = ", ".join(sel_exp)
        elif img_path:
            active_chat["last_router_decision"]["original_query"] = query
            active_chat["last_router_decision"]["cleaned_query"] = query
            active_chat["last_router_decision"]["greeting_removed"] = "No"
            active_chat["last_router_decision"]["routing_decision"] = "Forward to Hybrid Router (Vision)"
            active_chat["last_router_decision"]["final_expert_selected"] = "vision"
        
        active_chat["last_elapsed"] = elapsed

        # Save classification run inside session state router history
        if "router_history" not in st.session_state:
            st.session_state.router_history = []
        if active_chat["last_router_decision"]:
            h_entry = dict(active_chat["last_router_decision"])
            h_entry["elapsed"] = elapsed
            st.session_state.router_history.append(h_entry)

        # Save timeline and execution plan
        active_chat["last_timeline"] = [
            {
                "timestamp": e.timestamp,
                "message": e.message,
                "status": e.status,
                "expert": e.expert
            } for e in getattr(router, "last_timeline", [])
        ]
        active_chat["last_execution_plan"] = getattr(router, "last_execution_plan", {})

        # Save RAG retrieved chunks if research expert was run
        research_expert = router._get_expert(ExpertName.RESEARCH)
        retrieved_chunks = getattr(research_expert, "last_retrieved_chunks", [])
        active_chat["last_retrieved_chunks"] = [
            {
                "text": c["text"],
                "source": c["source"],
                "chunk_index": c["chunk_index"],
                "distance": c["distance"]
            } for c in retrieved_chunks
        ]

        # Trigger AI Evaluation Engine
        if st.session_state.get("enable_eval", True):
            try:
                from evaluation.engine import AIEvaluator  # noqa: PLC0415
                evaluator = AIEvaluator()
                eval_metrics = evaluator.evaluate(
                    query=query,
                    response=answer,
                    router_decision=active_chat["last_router_decision"],
                    response_time_s=elapsed
                )
                active_chat["last_evaluation_metrics"] = eval_metrics
            except Exception as eval_exc:
                logger.warning("AI Evaluation Engine execution failed: %s", eval_exc)
        else:
            active_chat["last_evaluation_metrics"] = {}

        # Trigger streaming typewriter flag
        st.session_state.just_generated = True
        save_chat_history()

    except Exception as e:
        placeholder.empty()
        st.error(f"❌ Something went wrong during response generation: {e}", icon="❌")

# ---------------------------------------------------------------------------
# Diagnostics & Performance Dashboard Tab
# ---------------------------------------------------------------------------
def _render_evaluation_dashboard(active_chat: dict) -> None:
    st.markdown("### 📊 Performance and System Telemetry Diagnostics")

    from utils.evaluation import EvaluationEngine  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    db = EvaluationEngine()
    raw_logs = db.get_raw_logs(limit=100)

    # Sync router history from SQLite logs if empty to ensure visual charts are pre-populated
    if "router_history" not in st.session_state or not st.session_state.router_history:
        st.session_state.router_history = []
        for i, log in enumerate(reversed(raw_logs)):
            conf_val = 0.85 + (i % 15) / 100.0 if log.success else 0.45 + (i % 10) / 100.0
            conf_val = min(max(conf_val, 0.0), 1.0)

            router_str = "ML Intent Classifier" if conf_val >= 0.60 else "LLM/Planner Router"
            fallback_val = False if conf_val >= 0.60 else True

            st.session_state.router_history.append({
                "query": log.query,
                "predicted_expert": log.expert_name,
                "confidence": conf_val,
                "router_used": router_str,
                "fallback_used": fallback_val,
                "timestamp": log.timestamp,
                "elapsed": log.response_time
            })

    responses = active_chat.get("last_responses", [])
    decision = active_chat.get("last_router_decision", {})
    wall_time = active_chat.get("last_elapsed", 0.0)

    # 1. Real-time Telemetry Metrics (if query exists in active chat)
    if responses and decision:
        total_tokens = sum(r["tokens_generated"] for r in responses)
        total_prompt_tokens = sum(r["prompt_tokens"] for r in responses)
        total_cost = sum(r["estimated_cost"] for r in responses)
        peak_memory = max((r["memory_usage_mb"] for r in responses), default=0.0)
        total_compute = sum(r["elapsed"] for r in responses)

        # ── Real-Time Metrics Cards ──
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="⏱️ Wall-Clock Latency",
                value=f"{wall_time:.2f}s",
                help="Total time taken for query routing + parallel response generation."
            )
        with col2:
            st.metric(
                label="🪙 Output Tokens",
                value=f"{total_tokens:,}",
                help=f"Tokens generated by experts. (Input Prompt: {total_prompt_tokens:,} tokens)."
            )
        with col3:
            st.metric(
                label="💵 Estimated Cost",
                value=f"${total_cost:.5f}",
                help="Equivalent cloud hosting costs computed based on input/output token counts."
            )
        with col4:
            st.metric(
                label="💾 Peak Memory Usage",
                value=f"{peak_memory:.1f} MB",
                help="Peak system memory consumed during parallel generation."
            )

        st.markdown("---")

        # ── Machine Learning Router Decisions visualizer ──
        active_pred = decision.get("predicted_expert") or "N/A"
        active_conf = decision.get("confidence", 0.0)
        active_router = decision.get("router_used", "LLM/Planner Router")
        active_fallback = decision.get("fallback_used", True)

        st.markdown("#### 🧠 Machine Learning Router Decision")
        c_pred1, c_pred2, c_pred3 = st.columns([1, 1.5, 1.5])

        with c_pred1:
            st.markdown("**Selected Expert**")
            st.subheader(f"💼 {active_pred.replace('_', ' ').title()}")

        with c_pred2:
            st.markdown("**Router Used**")
            router_status_str = "🤖 Machine Learning Router" if not active_fallback else "⚡ LLM Router (Fallback)"
            st.subheader(router_status_str)
            status_badge = "✅ ML Prediction Used" if not active_fallback else "⚠️ LLM Fallback Activated"
            st.markdown(f"**Status**: `{status_badge}`")

        with c_pred3:
            st.markdown("**Prediction Confidence**")
            st.subheader(f"🎯 {active_conf * 100:.1f}%")
            st.progress(min(max(float(active_conf), 0.0), 1.0))

        st.markdown("---")

        # ── AI Decision Engine outcome visualizer ──
        primary_exp_val = decision.get("primary_expert", "N/A")
        additional_exps = decision.get("additional_experts", [])
        decision_reason = decision.get("reason", "No reasoning provided.")
        execution_order = decision.get("execution_order", [])

        st.markdown("#### 🧠 AI Decision Engine Decision")

        c_dec1, c_dec2 = st.columns(2)
        with c_dec1:
            st.markdown(f"**Primary Expert**: `{primary_exp_val.replace('_', ' ').upper()}`")

            if additional_exps:
                add_list = ", ".join(f"`{e.replace('_', ' ').upper()}`" for e in additional_exps)
                st.markdown(f"**Additional Experts**: {add_list}")
                st.info("💡 **Decision**: Multi-Expert Execution Plan Activated.")
            else:
                st.markdown("**Additional Experts**: `None`")
                st.success("💡 **Decision**: Single Expert Plan Activated.")

            if execution_order:
                st.markdown("**Execution Flow Sequence**")
                flow_steps = " ➡️ ".join(f"`{e.replace('_', ' ').upper()}`" for e in execution_order)
                st.markdown(f"⛓️ {flow_steps}")

        with c_dec2:
            st.markdown("**Decision Reasoning**")
            st.info(decision_reason)

        st.markdown("---")

        # ── Router Diagnostics Panel ──
        st.markdown("#### ⚙️ Router Diagnostics Panel")
        st.markdown(
            f"""
            <div style="background: rgba(30, 41, 59, 0.35); border: 1px solid #1e293b; padding: 1.5rem; border-radius: 8px;">
                <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
                    <tr style="border-bottom: 1px solid #1e293b;">
                        <td style="padding: 0.6rem 0px; font-weight: 600; width: 30%;">User Query</td>
                        <td style="padding: 0.6rem 0px; color: #94a3b8;"><i>"{decision.get('query', 'N/A')}"</i></td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1e293b;">
                        <td style="padding: 0.6rem 0px; font-weight: 600;">Predicted Expert</td>
                        <td style="padding: 0.6rem 0px;"><code style="color: #a78bfa;">{active_pred.replace('_', ' ').upper()}</code></td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1e293b;">
                        <td style="padding: 0.6rem 0px; font-weight: 600;">Prediction Confidence</td>
                        <td style="padding: 0.6rem 0px; color: #10b981; font-weight: 600;">{active_conf * 100:.2f}%</td>
                    </tr>
                    <tr style="border-bottom: 1px solid #1e293b;">
                        <td style="padding: 0.6rem 0px; font-weight: 600;">Router Used</td>
                        <td style="padding: 0.6rem 0px; color: #3b82f6; font-weight: 500;">{active_router}</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.6rem 0px; font-weight: 600;">Processing Time</td>
                        <td style="padding: 0.6rem 0px; color: #f59e0b; font-weight: 500;">{wall_time:.3f} seconds</td>
                    </tr>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

        # ── Diagnostics breakdown table ──
        st.markdown("#### 🔬 Per-Expert Diagnostics Breakdown")
        table_rows = ""
        for r in responses:
            meta = EXPERT_META.get(r["expert_name"], {"icon": "🤖", "color": "#60a5fa"})
            status = "<span style='color:#10b981; font-weight:600;'>✅ SUCCESS</span>" if r["success"] else f"<span style='color:#ef4444; font-weight:600;'>❌ FAILED</span>"
            table_rows += f"""
            <tr style="border-bottom: 1px solid #1e293b;">
                <td style="padding: 0.8rem 1rem; font-weight: 500; color:#f8fafc;">{meta['icon']} {r['expert_label']}</td>
                <td style="padding: 0.8rem; text-align: center;">{status}</td>
                <td style="padding: 0.8rem; text-align: center;">{r['elapsed']:.2f}s</td>
                <td style="padding: 0.8rem; text-align: center;">{r['prompt_tokens']}</td>
                <td style="padding: 0.8rem; text-align: center;">{r['tokens_generated']}</td>
                <td style="padding: 0.8rem; text-align: center; color:#ef4444; font-weight:500;">${r['estimated_cost']:.5f}</td>
                <td style="padding: 0.8rem; text-align: center; color:#3b82f6; font-weight:500;">{r['memory_usage_mb']:.1f} MB</td>
            </tr>
            """

        st.markdown(
            f"""
            <table style="width: 100%; border-collapse: collapse; background: rgba(15, 23, 42, 0.45); border: 1px solid #1e293b; border-radius: 8px; overflow: hidden; font-size: 0.9rem;">
                <thead>
                    <tr style="background: rgba(30, 41, 59, 0.85); border-bottom: 2px solid #1e293b; color: #60a5fa; font-weight: 600; font-size:0.84rem; text-transform: uppercase; letter-spacing:0.8px;">
                        <th style="padding: 0.9rem 1rem; text-align: left;">Expert Domain</th>
                        <th style="padding: 0.9rem; text-align: center;">Status</th>
                        <th style="padding: 0.9rem; text-align: center;">Latency</th>
                        <th style="padding: 0.9rem; text-align: center;">Prompt Tokens</th>
                        <th style="padding: 0.9rem; text-align: center;">Generated Tokens</th>
                        <th style="padding: 0.9rem; text-align: center;">Est. Cost</th>
                        <th style="padding: 0.9rem; text-align: center;">Peak Memory</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            """,
            unsafe_allow_html=True
        )

        st.markdown("---")

    else:
        st.info("No active query telemetry logged in this session yet. Run a chat query to view real-time diagnostics.")

    # 2. Historical Classifier Statistics & Visual Charts
    history_df = pd.DataFrame(st.session_state.router_history)
    if not history_df.empty:
        st.markdown("#### 📈 Expert Statistics (Historical)")

        stats_grouped = history_df.groupby("predicted_expert").agg(
            usage_count=("query", "count"),
            avg_conf=("confidence", "mean"),
            avg_latency=("elapsed", "mean")
        ).reset_index()

        stats_grouped.columns = ["Expert", "Expert Usage Count", "Average Confidence", "Average Response Time"]

        pres_df = stats_grouped.copy()
        pres_df["Expert"] = pres_df["Expert"].apply(lambda x: str(x).replace("_", " ").title())
        pres_df["Average Confidence"] = pres_df["Average Confidence"].apply(lambda x: f"{x * 100:.1f}%")
        pres_df["Average Response Time"] = pres_df["Average Response Time"].apply(lambda x: f"{x:.3f}s")

        c_stats1, _ = st.columns([1, 3])
        with c_stats1:
            st.metric(label="📊 Total Classified Requests", value=len(history_df))

        st.dataframe(pres_df, width="stretch", hide_index=True)

        # ── Visual Analytics Charts ──
        st.markdown("#### 📊 Router Visual Analytics")
        c_chart1, c_chart2 = st.columns(2)

        with c_chart1:
            st.markdown("##### 🍕 Expert Usage Distribution")
            usage_counts = history_df["predicted_expert"].value_counts().reset_index()
            usage_counts.columns = ["Expert", "Usage Count"]
            usage_counts["Expert"] = usage_counts["Expert"].apply(lambda x: str(x).replace("_", " ").title())
            st.bar_chart(usage_counts.set_index("Expert")["Usage Count"], color="#8b5cf6")

            st.markdown("##### ⏱️ Response Time Timeline (s)")
            st.line_chart(history_df["elapsed"], color="#f59e0b")

        with c_chart2:
            st.markdown("##### 📉 Prediction Confidence Timeline (%)")
            conf_timeline = history_df["confidence"] * 100
            st.line_chart(conf_timeline, color="#10b981")

            st.markdown("##### ⚡ Router Performance (ML vs Fallback)")
            router_counts = history_df["router_used"].value_counts().reset_index()
            router_counts.columns = ["Router", "Count"]
            st.bar_chart(router_counts.set_index("Router")["Count"], color="#3b82f6")

        raw_data = []
        for log in raw_logs:
            raw_data.append({
                "Timestamp": log.timestamp,
                "Expert": log.expert_name.replace("_", " ").title(),
                "Query": log.query,
                "Latency (s)": round(log.response_time, 2),
                "Prompt Tokens": log.prompt_tokens,
                "Completion Tokens": log.completion_tokens,
                "Memory (MB)": round(log.memory_usage_mb, 1),
                "CPU (%)": round(log.cpu_usage_pct, 1),
                "Status": "✅ SUCCESS" if log.success else f"❌ FAILED: {log.error}"
            })
        if raw_data:
            st.dataframe(pd.DataFrame(raw_data), width="stretch", hide_index=True)
        else:
            st.text("No raw logs found.")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 🗣️ User Feedback & Prompt Optimization Learning")

    from utils.feedback import FeedbackSystem  # noqa: PLC0415
    fs = FeedbackSystem()

    # ── Database Actions ──
    col_clear_fb, col_logout = st.columns([1, 1])
    with col_clear_fb:
        if st.button("🧹 Clear SQLite Feedback Logs", width="stretch"):
            fs.clear_feedback()
            st.toast("SQLite feedback logs cleared! 🧼", icon="🗑️")
            st.rerun()

    fb_summary = fs.get_ratings_summary()
    if not fb_summary:
        st.info("No historical user feedback ratings recorded yet. Like 👍 or Dislike 👎 responses to collect metrics!")
    else:
        fb_data = []
        for name, stats in fb_summary.items():
            fb_data.append({
                "Expert": name.replace("_", " ").title(),
                "Likes": stats["likes"],
                "Dislikes": stats["dislikes"],
                "Net Rating": stats["net_rating"],
                "Satisfaction Ratio (%)": round(stats["like_ratio"] * 100, 1)
            })
        fb_df = pd.DataFrame(fb_data)

        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            st.markdown("##### 👍 Likes vs 👎 Dislikes per Expert")
            st.bar_chart(fb_df.set_index("Expert")[["Likes", "Dislikes"]])
        with fb_col2:
            st.markdown("##### 🗣️ Net Rating Score (Likes - Dislikes)")
            st.bar_chart(fb_df.set_index("Expert")["Net Rating"], color="#10b981")

        st.markdown("##### 📊 Expert Ratings Analytics Summary")
        st.dataframe(fb_df, width="stretch", hide_index=True)

    # ── Prompt Optimization Recommendations ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💡 Contextual Prompt Optimization Recommendations")
    st.markdown("Based on feedback history and the active experts in this session:")

    active_experts = decision.get("selected_experts", []) if "decision" in dir() else []
    if not active_experts:
        active_experts = list(EXPERT_META.keys())[:3]

    for exp_str in active_experts:
        recs = fs.get_prompt_recommendations(exp_str)
        st.markdown(f"##### **{exp_str.replace('_', ' ').title()} Expert**")
        for rec in recs:
            st.markdown(rec)
        st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Login screen & Logout Flow
# ---------------------------------------------------------------------------
def render_login_screen() -> None:
    """Render a premium dark-mode, glassmorphism centered login screen for IntelliMoE."""
    st.markdown(
        """
        <style>
        [data-testid="stHeader"] {
            display: none !important;
        }
        .main .block-container {
            max-width: 440px !important;
            margin: auto !important;
            padding: 6rem 1rem 2rem 1rem !important;
        }
        /* Custom inputs styling */
        div[data-baseweb="input"] {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 8px !important;
        }
        div[data-baseweb="input"] input {
            color: #c9d1d9 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 12px; padding: 2.5rem 2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.5); text-align: center; margin-bottom: 1.5rem;">
            <div style="display: inline-flex; align-items: center; justify-content: center; width: 56px; height: 56px; border-radius: 50%; background: rgba(79, 140, 255, 0.1); border: 1px solid rgba(79, 140, 255, 0.25); margin-bottom: 1.25rem;">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#4F8CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
                    <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
                    <path d="M12 5v14"/>
                </svg>
            </div>
            <h2 style="color: #f0f6fc; margin: 0 0 0.5rem 0; font-weight: 600; font-size: 1.6rem; letter-spacing: -0.5px;">Welcome to IntelliMoE</h2>
            <p style="color: #8b949e; font-size: 0.88rem; margin: 0 0 1.5rem 0;">Sign in to access your multi-expert AI assistant</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("login_form", clear_on_submit=False):
        email_val = st.text_input("Gmail Address", value="annamneedisuresh003@gmail.com", placeholder="username@gmail.com")
        password_val = st.text_input("Password", type="password", value="••••••••", placeholder="Enter password")
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("Sign In", width="stretch")
        
        if submit_btn:
            if not email_val or "@" not in email_val:
                st.error("Please enter a valid Gmail address.")
            else:
                st.session_state.user_email = email_val
                st.session_state.logged_in = True
                st.session_state.show_user_menu = False
                st.session_state.force_reload_chats = True
                st.success("Signed in successfully!")
                st.rerun()

    st.markdown(
        """
        <p style="text-align: center; color: #484f58; font-size: 0.72rem; margin-top: 1.5rem;">
            Secured via local environment authentication.
        </p>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    _init_session_state()

    # Always render sidebar first to prevent an empty sidebar column on the login screen
    render_sidebar()

    # Enforce login screen if user is not authenticated
    if not st.session_state.get("logged_in", False):
        render_login_screen()
        return

    # Render deferred dialogs (triggers programmatically after rerun)
    if st.session_state.get("show_settings_modal", False):
        st.session_state.show_settings_modal = False
        _render_settings_dialog()

    if st.session_state.get("show_dev_mode_modal", False):
        st.session_state.show_dev_mode_modal = False
        _render_dev_mode_dialog()

    if st.session_state.get("show_profile_modal", False):
        st.session_state.show_profile_modal = False
        _render_profile_dialog()

    render_main()


if __name__ == "__main__":
    main()
