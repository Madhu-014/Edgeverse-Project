import streamlit as st

st.set_page_config(
    page_title="Percieva™ Auto-Annotation Studio",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import io
import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data_augmentation import (
    ensure_dirs,
    extract_frames_every,
    augment_images_in_dir,
)

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR
VIDEOS_DIR = BASE_DIR / "videos"

ensure_dirs(str(BASE_DIR))

# Initialize session state for directory configuration
if "frames_dir" not in st.session_state:
    st.session_state["frames_dir"] = str(BASE_DIR / "output_frames")
if "annot_dir" not in st.session_state:
    st.session_state["annot_dir"] = str(BASE_DIR / "output_annotation")

# Use session state values
FRAMES_DIR = Path(st.session_state["frames_dir"])
ANNOT_DIR = Path(st.session_state["annot_dir"])

# Automatically create directories if they don't exist
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES_TXT = ANNOT_DIR / "classes.txt"

# Modern State-of-the-Art Design Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&display=swap');

/* ===== GLOBAL RESET & BASE ===== */
*, *::before, *::after {
    font-family: 'Poppins', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}

/* ===== MODERN BACKGROUND ===== */
.stApp {
    background: 
        radial-gradient(ellipse 100% 60% at 50% -20%, rgba(6, 182, 212, 0.15) 0%, transparent 50%),
        radial-gradient(ellipse 80% 50% at 100% 50%, rgba(168, 85, 247, 0.1) 0%, transparent 50%),
        radial-gradient(ellipse 80% 50% at 0% 100%, rgba(59, 130, 246, 0.08) 0%, transparent 50%),
        linear-gradient(135deg, #0f172a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ===== MAIN CONTAINER ===== */
.block-container {
    max-width: 1280px;
    padding: 2.5rem 1.5rem 4rem 1.5rem;
    margin-top: 0;
}

/* ===== MODERN HERO HEADER ===== */
.hero-header {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 50%, #ec4899 100%);
    padding: 3rem 2.5rem;
    border-radius: 24px;
    text-align: center;
    margin-bottom: 2.5rem;
    margin-top: 0.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 
        0 0 60px rgba(6, 182, 212, 0.3),
        0 20px 40px -10px rgba(139, 92, 246, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    border-radius: 50%;
    z-index: 0;
}

.hero-header > * {
    position: relative;
    z-index: 1;
}

.hero-title {
    color: #ffffff !important;
    font-size: 2.5rem;
    font-weight: 900;
    margin: 0;
    letter-spacing: -0.03em;
    text-shadow: 0 4px 15px rgba(0,0,0,0.3);
    background: linear-gradient(135deg, #ffffff 0%, #e0e7ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-subtitle {
    color: rgba(255,255,255,0.95) !important;
    font-size: 1.05rem;
    margin-top: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.01em;
}

.hero-badges {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    margin-top: 1.75rem;
    flex-wrap: wrap;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(15px);
    padding: 0.5rem 1.1rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 700;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.4);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    transition: all 0.3s ease;
}

.hero-badge:hover {
    background: rgba(255,255,255,0.25);
    transform: translateY(-2px);
    box-shadow: 0 10px 30px rgba(255,255,255,0.1);
}

/* ===== MODERN SECTION CARDS ===== */
.section-card {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 
        0 10px 40px rgba(0, 0, 0, 0.3),
        0 0 40px rgba(6, 182, 212, 0.05),
        inset 0 1px 0 rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(10px);
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
}

.section-card:hover {
    border-color: rgba(148, 163, 184, 0.3);
    box-shadow: 
        0 20px 60px rgba(0, 0, 0, 0.4),
        0 0 80px rgba(6, 182, 212, 0.15),
        inset 0 1px 0 rgba(255, 255, 255, 0.12);
    transform: translateY(-2px);
}

.section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 2px solid rgba(148, 163, 184, 0.1);
}

.section-icon {
    width: 56px;
    height: 56px;
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.4);
    flex-shrink: 0;
}

.section-title {
    color: #f8fafc !important;
    font-size: 1.25rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.02em;
}

.section-desc {
    color: #94a3b8 !important;
    font-size: 0.9rem;
    margin: 0;
    font-weight: 400;
}

/* ===== METRIC CARDS ===== */
.stMetric {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.08) 0%, rgba(139, 92, 246, 0.05) 100%);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 16px;
    padding: 1.25rem;
    backdrop-filter: blur(5px);
    transition: all 0.3s ease;
}

.stMetric:hover {
    border-color: rgba(6, 182, 212, 0.3);
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(139, 92, 246, 0.08) 100%);
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.15);
    transform: translateY(-2px);
}

/* ===== DIVIDER ===== */
.stDivider {
    background: linear-gradient(90deg, transparent 0%, rgba(148, 163, 184, 0.3) 50%, transparent 100%) !important;
}

/* ===== MODERN TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(15, 23, 42, 0.6);
    padding: 8px;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 0 24px;
    background: transparent;
    border-radius: 12px;
    color: #cbd5e1 !important;
    font-weight: 700;
    font-size: 0.9rem;
    border: none;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.stTabs [data-baseweb="tab"]:hover {
    color: #f8fafc !important;
    background: rgba(255, 255, 255, 0.05);
}

.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    font-weight: 800;
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.4);
}

.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span {
    color: #ffffff !important;
}

.stTabs [data-baseweb="tab-panel"] {
    padding-top: 2rem;
}

/* ===== MODERN BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
    color: #ffffff !important;
    border: none;
    border-radius: 14px;
    padding: 0.9rem 2.2rem;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    box-shadow: 
        0 0 0 1px rgba(255, 255, 255, 0.1),
        0 8px 24px rgba(6, 182, 212, 0.35);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    cursor: pointer;
}

.stButton > button:hover {
    transform: translateY(-3px);
    color: #ffffff !important;
    box-shadow: 
        0 0 0 1px rgba(255, 255, 255, 0.2),
        0 12px 32px rgba(6, 182, 212, 0.45);
}

.stButton > button:active {
    transform: translateY(-1px);
    color: #ffffff !important;
}

.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #ffffff !important;
    font-weight: 700;
}

/* ===== MODERN INPUTS ===== */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1.5px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    padding: 0.85rem 1.1rem;
    font-size: 0.95rem;
    transition: all 0.25s ease;
    backdrop-filter: blur(5px);
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: rgba(6, 182, 212, 0.5) !important;
    box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.15) !important;
    background: rgba(15, 23, 42, 0.95) !important;
}

/* ===== BUTTON STYLING ===== */
.stButton > button {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px;
    padding: 0.75rem 1.5rem !important;
    font-weight: 700;
    font-size: 0.95rem;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 8px 24px rgba(6, 182, 212, 0.35);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(6, 182, 212, 0.45) !important;
}

.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 4px 16px rgba(6, 182, 212, 0.3) !important;
}

/* ===== INPUT STYLING ===== */
.stTextInput > div > div > input {
    background: rgba(30, 41, 59, 0.6) !important;
    border: 1.5px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.1) !important;
}

/* ===== EXPANDER STYLING ===== */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%) !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
    transition: all 0.3s ease !important;
}

.streamlit-expanderHeader:hover {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.15) 0%, rgba(139, 92, 246, 0.1) 100%) !important;
    border-color: rgba(6, 182, 212, 0.3) !important;
}

.stTextInput label, .stNumberInput label, .stSelectbox label {
    color: #cbd5e1 !important;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.6rem;
}

/* ===== SELECT BOX ===== */
.stSelectbox > div > div {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1.5px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(5px);
}

/* ===== TYPOGRAPHY ===== */
.stMarkdown, .stText, p, span, li {
    color: #cbd5e1 !important;
    line-height: 1.7;
    font-size: 0.95rem;
}

h1, h2 {
    color: #f8fafc !important;
    font-weight: 800;
    letter-spacing: -0.02em;
}

h3 {
    color: #e2e8f0 !important;
    font-weight: 700;
    font-size: 1.3rem;
    margin-bottom: 1rem;
}

h4 {
    color: #cbd5e1 !important;
    font-weight: 700;
    font-size: 1.05rem;
}

/* ===== FILE UPLOADER ===== */
.stFileUploader {
    background: rgba(15, 23, 42, 0.4);
    border: 2px dashed rgba(6, 182, 212, 0.3);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    backdrop-filter: blur(5px);
}

.stFileUploader:hover {
    border-color: rgba(6, 182, 212, 0.6);
    background: rgba(15, 23, 42, 0.6);
}

.stFileUploader > div > button {
    background: rgba(6, 182, 212, 0.2) !important;
    border: 1.5px solid rgba(6, 182, 212, 0.4) !important;
    color: #22d3ee !important;
    border-radius: 10px !important;
    font-weight: 600;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: rgba(15, 23, 42, 0.6);
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    color: #cbd5e1 !important;
    font-weight: 600;
    font-size: 0.95rem;
    padding: 1rem 1.2rem;
    transition: all 0.25s ease;
    backdrop-filter: blur(5px);
}

.streamlit-expanderHeader:hover {
    color: #f8fafc !important;
    background: rgba(15, 23, 42, 0.8);
    border-color: rgba(148, 163, 184, 0.2);
}

.streamlit-expanderContent {
    background: rgba(15, 23, 42, 0.4);
    border-radius: 0 0 14px 14px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-top: none;
    padding: 1.5rem;
    backdrop-filter: blur(5px);
}

/* ===== ALERTS ===== */
.stSuccess {
    background: rgba(34, 197, 94, 0.12) !important;
    border: 1.5px solid rgba(34, 197, 94, 0.25);
    border-left: 4px solid #22c55e;
    border-radius: 12px;
    color: #86efac !important;
    padding: 1.2rem !important;
}

.stError {
    background: rgba(239, 68, 68, 0.12) !important;
    border: 1.5px solid rgba(239, 68, 68, 0.25);
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 1.2rem !important;
}

.stInfo {
    background: rgba(6, 182, 212, 0.12) !important;
    border: 1.5px solid rgba(6, 182, 212, 0.25);
    border-left: 4px solid #06b6d4;
    border-radius: 12px;
    padding: 1.2rem !important;
}

.stWarning {
    background: rgba(245, 158, 11, 0.12) !important;
    border: 1.5px solid rgba(245, 158, 11, 0.25);
    border-left: 4px solid #f59e0b;
    border-radius: 12px;
    padding: 1.2rem !important;
}

/* ===== SLIDER ===== */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #06b6d4 0%, #8b5cf6 100%) !important;
    height: 7px;
    border-radius: 4px;
}

.stSlider > div > div > div > div > div {
    background: #ffffff !important;
    border: 3px solid #06b6d4;
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.4);
    width: 22px;
    height: 22px;
}

.stSlider label {
    color: #f8fafc !important;
    font-weight: 600;
    font-size: 0.9rem;
}

/* ===== SELECT BOX IMPROVED ===== */
.stSelectbox [data-baseweb="select"] > div {
    background: rgba(15, 23, 42, 0.7) !important;
    border: 1.5px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 12px !important;
    color: #f8fafc !important;
    min-height: 46px;
    backdrop-filter: blur(5px);
}

.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: rgba(6, 182, 212, 0.4) !important;
}

.stSelectbox [data-baseweb="select"] > div > div {
    color: #f8fafc !important;
}

.stSelectbox svg {
    fill: #94a3b8 !important;
}

[data-baseweb="popover"] {
    background: rgba(15, 23, 42, 0.98) !important;
    border: 1.5px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 12px !important;
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5) !important;
    backdrop-filter: blur(20px);
}

[data-baseweb="menu"] li {
    color: #cbd5e1 !important;
    background: transparent !important;
}

[data-baseweb="menu"] li:hover {
    background: rgba(6, 182, 212, 0.15) !important;
    color: #f8fafc !important;
}

[data-baseweb="menu"] li[aria-selected="true"] {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
}

/* ===== CHECKBOX & RADIO ===== */
.stCheckbox label, .stRadio label {
    color: #cbd5e1 !important;
    font-weight: 600;
    font-size: 0.95rem;
}

.stRadio > div > label {
    background: rgba(15, 23, 42, 0.5);
    padding: 0.7rem 1.2rem;
    border-radius: 10px;
    border: 1.5px solid rgba(148, 163, 184, 0.1);
    transition: all 0.25s ease;
}

.stRadio > div > label:hover {
    background: rgba(15, 23, 42, 0.8);
    border-color: rgba(6, 182, 212, 0.3);
}

/* ===== METRIC CARDS ===== */
.stMetric {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.6) 100%);
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(10px);
}

.stMetric label {
    color: #94a3b8 !important;
    font-weight: 700;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
}

.stMetric [data-testid="stMetricValue"] {
    color: #06b6d4 !important;
    font-weight: 800;
    font-size: 2rem;
}

/* ===== FOLDER INFO ===== */
.folder-info {
    background: rgba(15, 23, 42, 0.6);
    padding: 0.75rem 1.2rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    font-family: 'SF Mono', 'Monaco', monospace;
    font-size: 0.85rem;
    color: #94a3b8;
    border: 1px solid rgba(148, 163, 184, 0.1);
    transition: all 0.25s ease;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    backdrop-filter: blur(5px);
}

.folder-info:hover {
    background: rgba(15, 23, 42, 0.9);
    border-color: rgba(6, 182, 212, 0.3);
    color: #cbd5e1;
}

/* ===== WORKFLOW STEP ===== */
.workflow-step {
    display: flex;
    align-items: flex-start;
    gap: 1.2rem;
    padding: 1.2rem;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.1);
    margin: 0.75rem 0;
    transition: all 0.25s ease;
    backdrop-filter: blur(5px);
}

.workflow-step:hover {
    border-color: rgba(6, 182, 212, 0.2);
    background: rgba(15, 23, 42, 0.8);
}

.step-number {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.9rem;
    color: white;
    flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
}

.step-title {
    color: #f8fafc;
    font-weight: 700;
    font-size: 1rem;
    margin-bottom: 0.35rem;
}

.step-desc {
    color: #94a3b8;
    font-size: 0.85rem;
    line-height: 1.6;
}

/* ===== DIVIDER ===== */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.1), transparent);
    margin: 2rem 0;
}

/* ===== IMAGES ===== */
.stImage {
    border-radius: 14px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(6, 182, 212, 0.2);
    border: 1px solid rgba(148, 163, 184, 0.1);
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.5);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.3);
}

/* ===== CODE BLOCKS ===== */
.stCodeBlock {
    background: rgba(15, 23, 42, 0.95) !important;
    border: 1px solid rgba(148, 163, 184, 0.1);
    border-radius: 12px;
    backdrop-filter: blur(10px);
}

code {
    background: rgba(6, 182, 212, 0.15) !important;
    color: #22d3ee !important;
    padding: 0.2rem 0.5rem;
    border-radius: 5px;
    font-size: 0.9rem;
    font-weight: 500;
}

/* ===== METRIC DISPLAY ===== */
.metric-value {
    color: #06b6d4 !important;
    font-weight: 800;
    font-size: 1.5rem;
}

/* ===== FOOTER ===== */
.app-footer {
    text-align: center;
    padding: 3rem 1.5rem 1.5rem 1.5rem;
    margin-top: 3rem;
    border-top: 1px solid rgba(148, 163, 184, 0.1);
}

.footer-brand {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
}

.footer-logo {
    width: 36px;
    height: 36px;
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3);
}

.footer-name {
    color: #f8fafc;
    font-weight: 800;
    font-size: 1rem;
    letter-spacing: -0.01em;
}

.footer-tagline {
    color: #64748b;
    font-size: 0.8rem;
    margin-bottom: 1rem;
}

.footer-tech {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
}

.tech-badge {
    color: #64748b;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.25s ease;
}

.tech-badge:hover {
    color: #06b6d4;
}

/* ===== EMPTY STATES ===== */
.empty-state {
    text-align: center;
    padding: 4rem 2.5rem;
    background: rgba(15, 23, 42, 0.4);
    border-radius: 18px;
    border: 2px dashed rgba(148, 163, 184, 0.1);
}

.empty-icon {
    font-size: 3rem;
    margin-bottom: 1.5rem;
    opacity: 0.6;
}

.empty-title {
    color: #cbd5e1;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}

.empty-desc {
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.6;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.section-card {
    animation: fadeIn 0.5s ease-out;
}

/* ===== TOPBAR & NAV ===== */
.nav-row {
    margin: 0.75rem 0 1.5rem 0;
}

div[data-testid="stHorizontalBlock"]:has(input[name="nav_page"]) {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 18px;
    padding: 0.55rem 0.8rem;
    box-shadow: 0 14px 35px rgba(0, 0, 0, 0.28);
    backdrop-filter: blur(12px);
    align-items: center;
}

.topbar-title {
    color: #e2e8f0;
    font-weight: 800;
    letter-spacing: -0.02em;
    font-size: 1.05rem;
    text-align: right;
}

.menu-btn .stButton > button {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%) !important;
    color: #ffffff !important;
    border: 0 !important;
    border-radius: 12px !important;
    padding: 0.45rem 0.8rem !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 20px rgba(6, 182, 212, 0.25) !important;
}

div[data-testid="stRadio"]:has(input[name="nav_page"]) > div {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 14px;
    padding: 0.25rem 0.4rem;
    gap: 0.4rem;
}

div[data-testid="stRadio"]:has(input[name="nav_page"]) label {
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 10px;
    padding: 0.4rem 0.9rem;
    color: #e2e8f0 !important;
    font-weight: 700;
    transition: all 0.2s ease;
}

div[data-testid="stRadio"]:has(input[name="nav_page"]) label:hover {
    border-color: rgba(6, 182, 212, 0.35);
    color: #ffffff !important;
}

div[data-testid="stRadio"]:has(input[name="nav_page"]) input:checked + div {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%);
    border-color: rgba(6, 182, 212, 0.55);
    color: #ffffff !important;
    box-shadow: 0 10px 25px rgba(6, 182, 212, 0.3);
}

/* ===== DRAWER ===== */
.drawer {
    position: fixed;
    top: 0;
    left: -500px;
    width: 480px;
    height: 100vh;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.99) 0%, rgba(30, 41, 59, 0.99) 100%);
    border-right: 1.5px solid rgba(148, 163, 184, 0.15);
    box-shadow: 25px 0 60px rgba(0, 0, 0, 0.45);
    padding: 1.5rem 1.5rem;
    transition: left 0.35s cubic-bezier(0.34, 1.56, 0.64, 1);
    z-index: 9999;
    overflow-y: auto;
    backdrop-filter: blur(8px);
}

.drawer.open {
    left: 0;
}

.drawer.open ~ [data-testid="stAppViewContainer"] {
    filter: brightness(0.5);
    pointer-events: none;
}

.drawer::-webkit-scrollbar {
    width: 6px;
}

.drawer::-webkit-scrollbar-track {
    background: rgba(15, 23, 42, 0.4);
}

.drawer::-webkit-scrollbar-thumb {
    background: rgba(6, 182, 212, 0.4);
    border-radius: 3px;
}

.drawer::-webkit-scrollbar-thumb:hover {
    background: rgba(6, 182, 212, 0.6);
}

.drawer-overlay {
    position: fixed;
    inset: 0;
    background: rgba(2, 6, 23, 0.65);
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.35s ease;
    z-index: 9998;
}

.drawer-overlay.show {
    opacity: 1;
    pointer-events: auto;
    cursor: pointer;
}

.drawer-title {
    color: #f8fafc;
    font-weight: 800;
    font-size: 0.95rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

.drawer-list {
    list-style: none;
    padding-left: 0;
    margin: 0.75rem 0 1rem 0;
}

.drawer-list li {
    color: #e2e8f0;
    padding: 0.5rem 0.65rem;
    border-radius: 10px;
    margin-bottom: 0.4rem;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.08);
}

.panel-blue {
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    border-radius: 20px;
    padding: 2rem;
    color: #ffffff;
    box-shadow: 0 15px 40px rgba(14, 165, 233, 0.35);
    margin: 1rem 0 2rem 0;
}

.panel-blue h2 {
    margin: 0 0 0.35rem 0;
    font-size: 1.6rem;
    font-weight: 800;
}

.panel-blue p {
    margin: 0;
    opacity: 0.95;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2rem;
    }
    
    .block-container {
        padding: 1.5rem 1rem 2rem 1rem;
    }
    
    .hero-header {
        padding: 2rem 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ===== PAGE STATE =====
if "nav_page" not in st.session_state:
    st.session_state["nav_page"] = "Annotate"

# ============================================================================
# PAGE NAVIGATION - Using Query Params
# ============================================================================
page_param = st.query_params.get("page", "annotate")
current_page = "Annotate" if page_param == "annotate" else "Model Compare" if page_param == "model" else "Analytics"

# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <style>
    [data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; padding-top: 1rem;">
        <div style="font-size: 1.1rem; font-weight: 700; background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">
            Percieva™
        </div>
        <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">Auto-Annotation Studio</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation Buttons - Vertical Stack with Icons
    st.markdown("""
    <style>
    .nav-divider {
        background: linear-gradient(90deg, rgba(139, 92, 246, 0.2) 0%, transparent 50%, rgba(139, 92, 246, 0.2) 100%);
        height: 1px;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
    }
    .stButton > button svg {
        width: 18px;
        height: 18px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div style="margin-bottom: 1rem;"></div>', unsafe_allow_html=True)
    
    # Annotate Button
    annotate_active = page_param == "annotate"
    col1, col2 = st.columns([0.15, 0.85])
    with col1:
        st.markdown(f"""
        <svg viewBox="0 0 24 24" fill="none" stroke="{'#06b6d4' if annotate_active else 'currentColor'}" stroke-width="2" style="width: 20px; height: 20px; margin-top: 8px; transition: all 0.3s ease;">
            <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
            <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
        </svg>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("Annotate", key="nav_annotate", use_container_width=True):
            st.query_params["page"] = "annotate"
            st.rerun()
        if annotate_active:
            st.markdown('<div style="height: 2px; background: linear-gradient(90deg, #06b6d4, #8b5cf6); margin-top: -8px; border-radius: 1px;"></div>', unsafe_allow_html=True)
    
    # Model Compare Button
    model_active = page_param == "model"
    col1, col2 = st.columns([0.15, 0.85])
    with col1:
        st.markdown(f"""
        <svg viewBox="0 0 24 24" fill="none" stroke="{'#06b6d4' if model_active else 'currentColor'}" stroke-width="2" style="width: 20px; height: 20px; margin-top: 8px; transition: all 0.3s ease;">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M12 1v6m0 6v6m5.2-13.2l-4.2 4.2m0 6l4.2 4.2M23 12h-6m-6 0H5m13.2 5.2l-4.2-4.2m0-6l4.2-4.2"></path>
        </svg>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("Model Compare", key="nav_model", use_container_width=True):
            st.query_params["page"] = "model"
            st.rerun()
        if model_active:
            st.markdown('<div style="height: 2px; background: linear-gradient(90deg, #06b6d4, #8b5cf6); margin-top: -8px; border-radius: 1px;"></div>', unsafe_allow_html=True)
    
    # Analytics Button
    analytics_active = page_param == "analytics"
    col1, col2 = st.columns([0.15, 0.85])
    with col1:
        st.markdown(f"""
        <svg viewBox="0 0 24 24" fill="none" stroke="{'#06b6d4' if analytics_active else 'currentColor'}" stroke-width="2" style="width: 20px; height: 20px; margin-top: 8px; transition: all 0.3s ease;">
            <line x1="18" y1="20" x2="18" y2="10"></line>
            <line x1="12" y1="20" x2="12" y2="4"></line>
            <line x1="6" y1="20" x2="6" y2="14"></line>
        </svg>
        """, unsafe_allow_html=True)
    with col2:
        if st.button("Analytics", key="nav_analytics", use_container_width=True):
            st.query_params["page"] = "analytics"
            st.rerun()
        if analytics_active:
            st.markdown('<div style="height: 2px; background: linear-gradient(90deg, #06b6d4, #8b5cf6); margin-top: -8px; border-radius: 1px;"></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="nav-divider"></div>', unsafe_allow_html=True)
    
    # Optional: Settings section
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    with st.expander("⚙ Settings", expanded=False):
        st.markdown("#### Directory Configuration")
        new_frames_dir = st.text_input(
            "Frames Directory",
            value=st.session_state["frames_dir"],
            key="config_frames",
        )
        if new_frames_dir != st.session_state["frames_dir"]:
            st.session_state["frames_dir"] = new_frames_dir
            Path(new_frames_dir).mkdir(parents=True, exist_ok=True)
        
        new_annot_dir = st.text_input(
            "Annotation Directory",
            value=st.session_state["annot_dir"],
            key="config_annot",
        )
        if new_annot_dir != st.session_state["annot_dir"]:
            st.session_state["annot_dir"] = new_annot_dir
            Path(new_annot_dir).mkdir(parents=True, exist_ok=True)

# ============================================================================
# PAGE CONTENT
# ============================================================================

# ANNOTATE PAGE
if current_page == "Annotate":
    # Hero Header
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:inline;vertical-align:middle;margin-right:8px;">
                <polygon points="23 7 16 12 23 17 23 7"></polygon>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
            </svg>
            Percieva<sup>TM</sup> Auto-Annotation Studio
        </h1>
        <p class="hero-subtitle">End-to-end video annotation pipeline powered by AI</p>
        <div class="hero-badges">
            <span class="hero-badge">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
                Smart Capture
            </span>
            <span class="hero-badge">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="3"></circle></svg>
                Auto Detect
            </span>
            <span class="hero-badge">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
                Export Ready
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different sections
    tabs = st.tabs(["Upload", "Augment", "Image Gallery", "Auto-Annotate", "Annotated Gallery"])
    
    # TAB 1: UPLOAD
    with tabs[0]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <div class="section-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                </div>
                <div>
                    <h3 class="section-title">Upload & Extract Frames</h3>
                    <p class="section-desc">Import video files or image datasets</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Supported Videos", "MP4, AVI, MOV")
        with col_info2:
            st.metric("Image Formats", "JPG, PNG, BMP")
        with col_info3:
            st.metric("Max Size", "200 MB")
        
        st.warning("""
        **⚠ Video Size Limit:** Videos must be under 200MB for upload.
        
        **If your video is larger:** Use the `segment_video.py` tool to split it into 200MB chunks:
        ```
        python segment_video.py large_video.mp4 ./segmented_videos 200
        ```
        Then upload each segment separately. All segments will be extracted to the same output folder.
        """)
        
        st.info(f"ℹ **Current destination:** `{FRAMES_DIR}` — Edit in **Settings** to use a different directory")
        
        src_type = st.radio("Select source", ["Video (.mp4)", "Images (ZIP/Files)"], horizontal=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            dest_base = st.text_input("Destination folder", value=str(FRAMES_DIR), label_visibility="collapsed")
        with col2:
            create_subfolder = st.checkbox("Auto subfolder", value=True)
        
        if src_type == "Video (.mp4)":
            up_video = st.file_uploader("Select MP4 video", type=["mp4"], key="video_upl")
            interval = st.number_input("Frame interval (seconds)", min_value=1, max_value=30, value=3, step=1)
            
            if up_video is not None:
                st.success(f"✓ Video loaded: {up_video.name} ({up_video.size / 1024 / 1024:.2f} MB)")
                if st.button("Extract Frames", type="primary"):
                    with st.spinner("Extracting frames..."):
                        saved = Path(VIDEOS_DIR) / up_video.name
                        saved.parent.mkdir(parents=True, exist_ok=True)
                        with open(saved, "wb") as f:
                            f.write(up_video.getbuffer())
                        vid_name = Path(up_video.name).stem if create_subfolder else ""
                        out_dir = Path(dest_base) / vid_name if vid_name else Path(dest_base)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        # Clear existing frames
                        for item in out_dir.iterdir():
                            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                item.unlink()
                        count = extract_frames_every(str(saved), str(out_dir), interval_seconds=int(interval))
                    st.success(f"✓ Cleared previous frames and extracted {count} new frames to {out_dir}")
                    st.balloons()
        else:
            up_zip = st.file_uploader("Images ZIP", type=["zip"], key="zip_upl")
            up_imgs = st.file_uploader("Or image files", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True, key="imgs_upl")
            
            if up_zip is not None or up_imgs:
                if st.button("Save Images", type="primary"):
                    with st.spinner("Saving images..."):
                        out_dir = Path(dest_base)
                        out_dir.mkdir(parents=True, exist_ok=True)
                        # Clear existing images
                        for item in out_dir.iterdir():
                            if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                item.unlink()
                        if up_zip:
                            with zipfile.ZipFile(io.BytesIO(up_zip.getvalue())) as zf:
                                zf.extractall(out_dir)
                        if up_imgs:
                            for f in up_imgs:
                                with open(out_dir / f.name, "wb") as fo:
                                    fo.write(f.getbuffer())
                    st.success(f"✓ Cleared previous images and saved {len(up_imgs) if up_imgs else 'ZIP'} new images to {out_dir}")
                    st.balloons()
    
    # TAB 2: AUGMENT
    with tabs[1]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <div class="section-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                        <circle cx="13.5" cy="6.5" r="0.5" fill="white"></circle>
                        <circle cx="17.5" cy="10.5" r="0.5" fill="white"></circle>
                        <circle cx="8.5" cy="7.5" r="0.5" fill="white"></circle>
                        <circle cx="6.5" cy="12.5" r="0.5" fill="white"></circle>
                        <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c.926 0 1.648-.746 1.648-1.688 0-.437-.18-.835-.437-1.125-.29-.289-.438-.652-.438-1.125a1.64 1.64 0 0 1 1.668-1.668h1.996c3.051 0 5.555-2.503 5.555-5.555C21.965 6.012 17.461 2 12 2z"></path>
                    </svg>
                </div>
                <div>
                    <h3 class="section-title">Data Augmentation</h3>
                    <p class="section-desc">Expand your dataset with intelligent transformations</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Techniques", "6+")
        with col_m2:
            st.metric("Variants", "0-6 per image")
        with col_m3:
            st.metric("Output", "JPEG/PNG")
        
        st.divider()
        
        aug_target = st.text_input("Source folder", value=str(FRAMES_DIR), key="aug_folder")
        variants = st.slider("Variants per image", 0, 6, 2)
        
        st.markdown("#### Augmentation Techniques")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            use_noise = st.checkbox("Gaussian noise", True)
            use_blur = st.checkbox("Gaussian blur", True)
        with col_b:
            use_motion = st.checkbox("Motion blur", True)
            use_brightness = st.checkbox("Brightness/Contrast", True)
        with col_c:
            use_rotate = st.checkbox("Small rotation", True)
            use_fog = st.checkbox("Light fog", False)
        
        st.divider()
        
        if st.button("Run Augmentation", type="primary", use_container_width=True):
            with st.spinner("Augmenting..."):
                # Save to both source and output_frames directories
                written = augment_images_in_dir(
                    aug_target,
                    output_dir=aug_target,
                    variants_per_image=variants,
                    use_gaussian_noise=use_noise,
                    use_salt_pepper=False,
                    use_small_rotate=use_rotate,
                    use_brightness_contrast=use_brightness,
                    use_gaussian_blur=use_blur,
                    use_motion_blur=use_motion,
                    use_fog=use_fog,
                    use_color_shift=False,
                )
                # Also copy augmented images to output_frames
                output_frames_dir = Path(st.session_state.get("frames_dir", str(FRAMES_DIR))) / "augmented"
                output_frames_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                for img_file in Path(aug_target).glob("*.jpg") + Path(aug_target).glob("*.png"):
                    if "aug_" in img_file.name or img_file.name.count('_') > 1:
                        shutil.copy2(img_file, output_frames_dir / img_file.name)
            st.success(f"✓ Created {written} augmented images and saved to output frames")
            st.balloons()
    
    # TAB 3: IMAGE GALLERY (BEFORE ANNOTATION)
    with tabs[2]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <div class="section-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                        <circle cx="8.5" cy="8.5" r="1.5"></circle>
                        <polyline points="21 15 16 10 5 21"></polyline>
                    </svg>
                </div>
                <div>
                    <h3 class="section-title">Image Gallery</h3>
                    <p class="section-desc">Preview your dataset before annotation</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        gallery_dir = Path(st.session_state.get("frames_dir", str(FRAMES_DIR)))
        IMAGES_PER_PAGE = 12
        
        def get_all_images(dir_path):
            imgs = []
            for root, _, files in os.walk(dir_path):
                for f in sorted(files):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        imgs.append(Path(root) / f)
            return sorted(imgs)
        
        def load_image(path):
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(path)
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        cols_per_row = st.selectbox("Grid columns", [2, 3, 4, 5], 1, key="gallery_cols_per_row")
        
        all_imgs = get_all_images(gallery_dir)
        total_pages = (len(all_imgs) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE
        
        if "gallery_page" not in st.session_state:
            st.session_state["gallery_page"] = 1
        
        if all_imgs:
            st.success(f"Total images: {len(all_imgs)}")
            
            # Pagination controls
            col_prev, col_page, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.button("← Previous", key="gallery_prev", use_container_width=True):
                    if st.session_state["gallery_page"] > 1:
                        st.session_state["gallery_page"] -= 1
                        st.rerun()
            
            with col_page:
                pages = list(range(1, total_pages + 1))
                current_page = st.selectbox(
                    "Page",
                    pages,
                    index=st.session_state["gallery_page"] - 1,
                    key="gallery_page_select"
                )
                st.session_state["gallery_page"] = current_page
                st.write(f"Page {st.session_state['gallery_page']} of {total_pages}")
            
            with col_next:
                if st.button("Next →", key="gallery_next", use_container_width=True):
                    if st.session_state["gallery_page"] < total_pages:
                        st.session_state["gallery_page"] += 1
                        st.rerun()
            
            st.divider()
            
            # Display images for current page
            start_idx = (st.session_state["gallery_page"] - 1) * IMAGES_PER_PAGE
            end_idx = start_idx + IMAGES_PER_PAGE
            page_imgs = all_imgs[start_idx:end_idx]
            
            st.write(f"Displaying images {start_idx + 1}-{min(end_idx, len(all_imgs))} in {cols_per_row} columns")
            cols = st.columns(cols_per_row)
            
            for i, p in enumerate(page_imgs):
                try:
                    pil = load_image(p)
                    with cols[i % cols_per_row]:
                        st.image(pil, caption=p.name, use_column_width=True)
                        if st.button("🗑 Delete", key=f"del_gallery_{p.name}", use_container_width=True):
                            try:
                                p.unlink()
                                st.success(f"Deleted {p.name}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete: {e}")
                except Exception as e:
                    with cols[i % cols_per_row]:
                        st.error(f"Failed to load {p.name}")
        else:
            st.info("No images found. Start by uploading images or extracting frames from a video.")
    
    # TAB 4: AUTO-ANNOTATE
    with tabs[3]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <div class="section-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                        <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                        <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                    </svg>
                </div>
                <div>
                    <h3 class="section-title">AI Auto-Annotation</h3>
                    <p class="section-desc">YOLO-powered object detection</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Model", "YOLO v11")
        with col_m2:
            st.metric("Format", "YOLO TXT")
        with col_m3:
            st.metric("GPU", "Enabled")
        
        annot_dir_input = st.text_input("Annotation directory", value=str(ANNOT_DIR), key="annot_input")
        
        # Class Management
        st.markdown("#### Class Management")
        CLASSES_TXT = Path(annot_dir_input) / "classes.txt"
        
        def load_classes():
            if CLASSES_TXT.exists():
                with open(CLASSES_TXT) as f:
                    return [line.strip() for line in f if line.strip()]
            return []
        
        def save_classes(classes_list):
            CLASSES_TXT.parent.mkdir(parents=True, exist_ok=True)
            with open(CLASSES_TXT, "w") as f:
                f.write("\n".join(classes_list) + "\n" if classes_list else "")
        
        current_classes = load_classes()
        
        if current_classes:
            st.write("Current classes:", ", ".join(current_classes))
        
        col_add, col_remove = st.columns(2)
        with col_add:
            new_class = st.text_input("Add class", placeholder="class name")
            if st.button("Add", key="add_class"):
                if new_class.strip() and new_class.lower() not in [c.lower() for c in current_classes]:
                    current_classes.append(new_class.lower())
                    save_classes(current_classes)
                    st.rerun()
        
        with col_remove:
            if current_classes:
                cls_remove = st.selectbox("Remove class", current_classes, key="remove_class_sel")
                if st.button("Remove", key="remove_class"):
                    current_classes.remove(cls_remove)
                    save_classes(current_classes)
                    st.rerun()
        
        st.divider()
        
        if st.button("Run Auto-Annotation", type="primary"):
            frames_dir = Path(st.session_state.get("frames_dir", str(FRAMES_DIR)))
            annot_dir = Path(annot_dir_input)
            
            def list_images(dir_path, limit=30):
                imgs = []
                for root, _, files in os.walk(dir_path):
                    for f in files:
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            imgs.append(Path(root) / f)
                            if len(imgs) >= limit:
                                return imgs
                return imgs
            
            frames_list = list_images(frames_dir)
            if not frames_list:
                st.error(f"❌ No frames found in {frames_dir}")
            else:
                st.info(f"Found {len(frames_list)} frames. Starting annotation...")
                with st.spinner("Running YOLO inference..."):
                    proc = subprocess.run(
                        [sys.executable, "auto_annotation.py", "--frames-dir", str(frames_dir), "--annot-dir", str(annot_dir)],
                        cwd=str(BASE_DIR),
                        capture_output=True,
                        text=True,
                    )
                
                if proc.returncode == 0:
                    st.success("✓ Auto-annotation completed!")
                    st.balloons()
                else:
                    st.error(f"❌ Failed with code {proc.returncode}")
    
    # TAB 5: ANNOTATED GALLERY (AFTER ANNOTATION)
    with tabs[4]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <div class="section-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                        <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                        <circle cx="12" cy="12" r="3"></circle>
                    </svg>
                </div>
                <div>
                    <h3 class="section-title">Annotated Gallery</h3>
                    <p class="section-desc">Review annotated images with bounding boxes</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        preview_dir = Path(st.session_state.get("annot_dir", str(ANNOT_DIR)))
        ANNOTATED_IMAGES_PER_PAGE = 12
        
        def get_all_annotated_images(dir_path):
            imgs = []
            for root, _, files in os.walk(dir_path):
                for f in sorted(files):
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        imgs.append(Path(root) / f)
            return sorted(imgs)
        
        def load_image(path):
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(path)
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        def draw_boxes(img_pil, txt_path):
            img = np.array(img_pil).copy()
            h, w = img.shape[:2]
            if not txt_path.exists():
                return Image.fromarray(img)
            with open(txt_path) as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x, y = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
                x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
                cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
            return Image.fromarray(img)
        
        cols_per_row = st.selectbox("Grid columns", [2, 3, 4, 5], 1, key="annotated_cols_per_row")
        
        all_annotated_imgs = get_all_annotated_images(preview_dir)
        total_annotated_pages = (len(all_annotated_imgs) + ANNOTATED_IMAGES_PER_PAGE - 1) // ANNOTATED_IMAGES_PER_PAGE
        
        if "annotated_page" not in st.session_state:
            st.session_state["annotated_page"] = 1
        
        if all_annotated_imgs:
            st.success(f"Total annotated images: {len(all_annotated_imgs)}")
            
            # Pagination controls
            col_prev, col_page, col_next = st.columns([1, 3, 1])
            with col_prev:
                if st.button("← Previous", key="annotated_prev", use_container_width=True):
                    if st.session_state["annotated_page"] > 1:
                        st.session_state["annotated_page"] -= 1
                        st.rerun()
            
            with col_page:
                pages = list(range(1, total_annotated_pages + 1))
                current_page = st.selectbox(
                    "Page",
                    pages,
                    index=st.session_state["annotated_page"] - 1,
                    key="annotated_page_select"
                )
                st.session_state["annotated_page"] = current_page
                st.write(f"Page {st.session_state['annotated_page']} of {total_annotated_pages}")
            
            with col_next:
                if st.button("Next →", key="annotated_next", use_container_width=True):
                    if st.session_state["annotated_page"] < total_annotated_pages:
                        st.session_state["annotated_page"] += 1
                        st.rerun()
            
            st.divider()
            
            # Display images for current page
            start_idx = (st.session_state["annotated_page"] - 1) * ANNOTATED_IMAGES_PER_PAGE
            end_idx = start_idx + ANNOTATED_IMAGES_PER_PAGE
            page_annotated_imgs = all_annotated_imgs[start_idx:end_idx]
            
            st.write(f"Displaying images {start_idx + 1}-{min(end_idx, len(all_annotated_imgs))} in {cols_per_row} columns")
            cols = st.columns(cols_per_row)
            
            for i, p in enumerate(page_annotated_imgs):
                try:
                    pil = load_image(p)
                    drawn = draw_boxes(pil, p.with_suffix(".txt"))
                    with cols[i % cols_per_row]:
                        st.image(drawn, caption=p.name, use_column_width=True)
                        if st.button("🗑 Delete", key=f"del_annotated_{p.name}", use_container_width=True):
                            try:
                                txt_file = p.with_suffix(".txt")
                                p.unlink()
                                if txt_file.exists():
                                    txt_file.unlink()
                                st.success(f"Deleted {p.name} and annotation")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to delete: {e}")
                except Exception as e:
                    with cols[i % cols_per_row]:
                        st.error(f"Failed to load {p.name}")
        else:
            st.info("No annotated images yet")

# MODEL COMPARE PAGE
elif current_page == "Model Compare":
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2">
            <circle cx="12" cy="12" r="3"></circle>
            <path d="M12 1v6m0 6v6m5.2-13.2l-4.2 4.2m0 6l4.2 4.2M23 12h-6m-6 0H5m13.2 5.2l-4.2-4.2m0-6l4.2-4.2"></path>
        </svg>
        <h1 style="color: #f8fafc; margin: 0; font-weight: 800;">Model Compare</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="panel-blue">
        <h2>Compare YOLO Models</h2>
        <p>Analyze and compare different model versions side-by-side.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model 1", "YOLO v12s")
    with col2:
        st.metric("Model 2", "YOLO v11n")
    
    st.info("Coming soon: Compare inference speed, accuracy, and resource usage")

# ANALYTICS PAGE
elif current_page == "Analytics":
    st.title("� Analytics")
    st.markdown("""
    <div class="panel-blue">
        <h2>Annotation Analytics</h2>
        <p>Monitor annotation progress and model performance metrics.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Frames", "0")
    with col2:
        st.metric("Annotated", "0%")
    with col3:
        st.metric("Classes", "0")
    
    st.info("Coming soon: Detailed analytics dashboard with charts and statistics")

# Footer
st.markdown("---")
st.markdown("""
<div class="app-footer">
    <div class="footer-brand">
        <div class="footer-logo">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                <polygon points="23 7 16 12 23 17 23 7"></polygon>
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
            </svg>
        </div>
        <span class="footer-name">Percieva<sup>TM</sup> Auto-Annotation Studio</span>
    </div>
    <p class="footer-tagline">Streamlined annotation workflow for computer vision teams</p>
    <div class="footer-tech">
        <span class="tech-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle></svg>
            <span>Python</span>
        </span>
        <span class="tech-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="3"></circle></svg>
            <span>Ultralytics YOLO</span>
        </span>
        <span class="tech-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
            <span>Streamlit</span>
        </span>
        <span class="tech-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
            <span>OpenCV</span>
        </span>
    </div>
</div>
""", unsafe_allow_html=True)
