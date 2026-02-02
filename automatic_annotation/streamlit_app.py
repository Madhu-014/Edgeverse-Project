import streamlit as st

st.set_page_config(
    page_title="ARAS Auto-Annotation Studio",
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
    transition: all 0.3s ease;
}

.section-card:hover {
    border-color: rgba(148, 163, 184, 0.2);
    box-shadow: 
        0 15px 50px rgba(0, 0, 0, 0.4),
        0 0 60px rgba(6, 182, 212, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
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

# Hero header with production design
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:inline;vertical-align:middle;margin-right:8px;">
            <polygon points="23 7 16 12 23 17 23 7"></polygon>
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
        </svg>
        ARAS Auto-Annotation Studio
    </h1>
    <p class="hero-subtitle">End-to-end video annotation pipeline powered by AI</p>
    <div class="hero-badges">
        <span class="hero-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
            YOLO v11
        </span>
        <span class="hero-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="3"></circle></svg>
            Auto-Detect
        </span>
        <span class="hero-badge">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
            Export Ready
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# Folders info and configuration
with st.expander("Project Configuration", expanded=False):
    st.markdown("""
    <style>
    .config-section {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.75rem 0;
    }
    .config-title {
        color: #e2e8f0;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Directory Configuration")
    st.info("Customize your working directories to match your project structure. Changes take effect immediately.")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        st.markdown('<div class="config-title">📁 Output Frames Directory</div>', unsafe_allow_html=True)
        new_frames_dir = st.text_input(
            "Frames directory path",
            value=st.session_state["frames_dir"],
            key="config_frames_input",
            label_visibility="collapsed",
            placeholder="e.g., ./output_frames or ./frames_extracted"
        )
        if new_frames_dir != st.session_state["frames_dir"]:
            st.session_state["frames_dir"] = new_frames_dir
            Path(new_frames_dir).mkdir(parents=True, exist_ok=True)
            st.success(f"✓ Frames directory created/updated: {new_frames_dir}")
    
    with col_config2:
        st.markdown('<div class="config-title">🏷️ Output Annotation Directory</div>', unsafe_allow_html=True)
        new_annot_dir = st.text_input(
            "Annotations directory path",
            value=st.session_state["annot_dir"],
            key="config_annot_input",
            label_visibility="collapsed",
            placeholder="e.g., ./output_annotation or ./labels"
        )
        if new_annot_dir != st.session_state["annot_dir"]:
            st.session_state["annot_dir"] = new_annot_dir
            Path(new_annot_dir).mkdir(parents=True, exist_ok=True)
            st.success(f"✓ Annotations directory created/updated: {new_annot_dir}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Current Project Paths")
    
    col_display1, col_display2, col_display3 = st.columns(3)
    
    with col_display1:
        st.markdown(f'''
        <div class="folder-info">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"></polygon><rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect></svg>
            <span>Videos</span>
        </div>
        <div style="background: rgba(17, 24, 39, 0.4); padding: 0.5rem; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); font-family: monospace; font-size: 0.75rem; color: #64748b; word-break: break-all;">
            {VIDEOS_DIR}
        </div>
        ''', unsafe_allow_html=True)
    
    with col_display2:
        st.markdown(f'''
        <div class="folder-info">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle cx="8.5" cy="8.5" r="1.5"></circle><polyline points="21 15 16 10 5 21"></polyline></svg>
            <span>Frames</span>
        </div>
        <div style="background: rgba(17, 24, 39, 0.4); padding: 0.5rem; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); font-family: monospace; font-size: 0.75rem; color: #64748b; word-break: break-all;">
            {FRAMES_DIR}
        </div>
        ''', unsafe_allow_html=True)
    
    with col_display3:
        st.markdown(f'''
        <div class="folder-info">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="2"><path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path><line x1="7" y1="7" x2="7.01" y2="7"></line></svg>
            <span>Annotations</span>
        </div>
        <div style="background: rgba(17, 24, 39, 0.4); padding: 0.5rem; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05); font-family: monospace; font-size: 0.75rem; color: #64748b; word-break: break-all;">
            {ANNOT_DIR}
        </div>
        ''', unsafe_allow_html=True)

def unzip_to_dir(zipped_bytes: bytes, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zipped_bytes)) as zf:
        zf.extractall(dest_dir)

def clear_directory(dir_path: Path):
    """Delete all files in a directory (does not delete subdirectories)"""
    if dir_path.exists():
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()

def save_uploaded_file(uploaded_file, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path

def run_auto_annotation() -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "auto_annotation.py"],
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )

def list_images(dir_path: Path, limit: int = 30):
    imgs = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                imgs.append(Path(root) / f)
                if len(imgs) >= limit:
                    return imgs
    return imgs

def load_image(path: Path) -> Image.Image:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def draw_yolo_boxes_on_image(img_pil: Image.Image, txt_path: Path, color=(0, 255, 0)) -> Image.Image:
    img = np.array(img_pil).copy()
    h, w = img.shape[:2]
    if not txt_path.exists():
        return Image.fromarray(img)
    try:
        with open(txt_path, "r") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except Exception:
        lines = []
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        x = int((cx - bw / 2) * w)
        y = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x, y), (x2, y2), color, 2)
    return Image.fromarray(img)

# Tabs
tabs = st.tabs(["Upload", "Augment", "Auto-Annotate", "Preview", "LabelImg"])

# Tab 1: Upload
with tabs[0]:
    st.markdown('''
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
                <p class="section-desc">Import your video files or image datasets to begin the annotation pipeline</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("Supported Videos", "MP4, AVI, MOV")
    with col_info2:
        st.metric("Image Formats", "JPG, PNG, BMP")
    with col_info3:
        st.metric("Max Upload Size", "200 MB")
    
    st.warning("""
    **📹 Video Size Limit:** Videos must be under 200MB for upload.
    
    **If your video is larger:** Use the `segment_video.py` tool to split it into 200MB chunks:
    ```
    python segment_video.py large_video.mp4 ./segmented_videos 200
    ```
    Then upload each segment separately. All segments will be extracted to the same output folder.
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"📂 **Current destination:** `{FRAMES_DIR}` — Edit in **Project Configuration** to use a different directory")
    
    src_type = st.radio("Select source type", ["Video (.mp4)", "Images (ZIP or files)"], horizontal=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        dest_base = st.text_input("Destination folder", value=str(FRAMES_DIR), label_visibility="collapsed")
    with col2:
        create_subfolder = st.checkbox("Auto subfolder", value=True)

    if src_type == "Video (.mp4)":
        up_video = st.file_uploader("Select MP4 video", type=["mp4"], key="video_upl")
        interval = st.number_input("Frame interval (seconds)", min_value=1, max_value=30, value=3, step=1)
        
        if up_video is not None:
            st.success(f"Video loaded: {up_video.name} ({up_video.size / 1024 / 1024:.2f} MB)")
            saved = save_uploaded_file(up_video, VIDEOS_DIR / up_video.name)
            vid_name = Path(up_video.name).stem if create_subfolder else ""
            out_dir = Path(dest_base) / vid_name if vid_name else Path(dest_base)
            
            if st.button("Extract Frames", type="primary"):
                with st.spinner("Extracting frames..."):
                    out_dir.mkdir(parents=True, exist_ok=True)
                    # Clear existing frames
                    clear_directory(out_dir)
                    count = extract_frames_every(str(saved), str(out_dir), interval_seconds=int(interval))
                st.success(f"✓ Cleared previous frames and extracted {count} new frames to {out_dir}")
                st.balloons()
    else:
        col_zip, col_files = st.columns(2)
        with col_zip:
            up_zip = st.file_uploader("Images ZIP", type=["zip"], key="zip_upl")
        with col_files:
            up_imgs = st.file_uploader("Or select image files", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True, key="imgs_upl")

        if up_zip is not None or up_imgs:
            if up_zip:
                st.success(f"ZIP loaded: {up_zip.name}")
            if up_imgs:
                st.success(f"Files selected: {len(up_imgs)} images")
            
            sub_name = "images" if up_zip else (up_imgs[0].name if up_imgs else "")
            folder_name = Path(sub_name).stem if create_subfolder and sub_name else ""
            out_dir = Path(dest_base) / folder_name if folder_name else Path(dest_base)
            
            if st.button("Save Images", type="primary"):
                with st.spinner("Saving images..."):
                    out_dir.mkdir(parents=True, exist_ok=True)
                    # Clear existing images
                    clear_directory(out_dir)
                    if up_zip:
                        unzip_to_dir(up_zip.getvalue(), out_dir)
                    if up_imgs:
                        for f in up_imgs:
                            if f:
                                with open(out_dir / Path(f.name).name, "wb") as fo:
                                    fo.write(f.getbuffer())
                st.success(f"✓ Cleared previous images and saved {len(up_imgs) if up_imgs else 'ZIP'} new images to {out_dir}")
                st.balloons()

# Tab 2: Augment
with tabs[1]:
    st.markdown('''
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
                <p class="section-desc">Expand your dataset with intelligent transformations to improve model robustness</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.info("**ARAS-Optimized:** No horizontal flips are applied to preserve left/right arm semantics for accurate surgical annotations.")
    
    aug_target = st.text_input("Source folder", value=str(FRAMES_DIR), key="aug_folder")
    
    st.markdown("""
    <style>
    .variant-selector {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1rem 0 1.5rem 0;
    }
    .variant-label {
        color: #94a3b8;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }
    .variant-options {
        display: flex;
        gap: 0.35rem;
    }
    .variant-btn {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: rgba(17, 24, 39, 0.8);
        color: #94a3b8;
        font-weight: 600;
        font-size: 0.95rem;
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .variant-btn:hover {
        background: rgba(99, 102, 241, 0.2);
        border-color: rgba(99, 102, 241, 0.4);
        color: #e2e8f0;
    }
    .variant-btn.active {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        border-color: transparent;
        color: #ffffff;
        box-shadow: 0 4px 12px rgba(14, 165, 233, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    col_var1, col_var2 = st.columns([1, 3])
    with col_var1:
        st.markdown('<span class="variant-label">Variants per image:</span>', unsafe_allow_html=True)
    with col_var2:
        variants_per_image = st.selectbox(
            "Select number",
            options=[0, 1, 2, 3, 4, 5, 6],
            index=2,
            key="variants_select",
            label_visibility="collapsed"
        )
    
    st.markdown("#### Augmentation Options")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        use_gaussian_noise = st.checkbox("Gaussian noise", value=True)
        use_gaussian_blur = st.checkbox("Gaussian blur", value=True)
    with col_b:
        use_motion_blur = st.checkbox("Motion blur", value=True)
        use_brightness_contrast = st.checkbox("Brightness/Contrast", value=True)
    with col_c:
        use_small_rotate = st.checkbox("Small rotation", value=True)
        use_fog = st.checkbox("Light fog/haze", value=False)

    if st.button("Run Augmentation", type="primary"):
        with st.spinner("Augmenting images..."):
            written = augment_images_in_dir(
                str(aug_target),
                output_dir=str(aug_target),
                variants_per_image=int(variants_per_image),
                use_gaussian_noise=use_gaussian_noise,
                use_salt_pepper=False,
                use_small_rotate=use_small_rotate,
                use_brightness_contrast=use_brightness_contrast,
                use_gaussian_blur=use_gaussian_blur,
                use_motion_blur=use_motion_blur,
                use_fog=use_fog,
                use_color_shift=False,
            )
        st.success(f"Created {written} augmented images")
        st.balloons()

# Tab 3: Auto-Annotate
with tabs[2]:
    st.markdown('''
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"></rect>
                    <path d="M7 11V7a5 5 0 0 1 10 0v4"></path>
                    <circle cx="12" cy="16" r="1"></circle>
                </svg>
            </div>
            <div>
                <h3 class="section-title">AI Auto-Annotation</h3>
                <p class="section-desc">Leverage YOLO deep learning to automatically detect and label objects in your frames</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Model", "YOLO v11")
    with col_m2:
        st.metric("Output Format", "YOLO TXT")
    with col_m3:
        st.metric("GPU Accelerated", "Yes")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"📂 **Current annotation directory:** `{ANNOT_DIR}` — Edit in **Project Configuration** to use a different directory")
    
    annot_dir_input = st.text_input(
        "Annotation output directory",
        value=st.session_state.get("annot_dir", str(ANNOT_DIR)),
        key="annot_input",
        label_visibility="collapsed"
    )
    st.session_state["annot_dir"] = annot_dir_input
    chosen_annot_dir = Path(annot_dir_input)
    chosen_annot_dir.mkdir(parents=True, exist_ok=True)
    
    # ===== CLASS MANAGEMENT SECTION =====
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('''
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                    <path d="M20.59 13.41l-7.17 7.17a2 2 0 0 1-2.83 0L2 12V2h10l8.59 8.59a2 2 0 0 1 0 2.82z"></path>
                    <line x1="7" y1="7" x2="7.01" y2="7"></line>
                </svg>
            </div>
            <div>
                <h3 class="section-title">Class Management</h3>
                <p class="section-desc">Add or remove object classes for annotation</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Path to predefined_classes.txt in labelImg package
    LABELIMG_CLASSES_PATH = Path("/Users/madhusudhan/Documents/edgeverse_project/venv/lib/python3.10/site-packages/labelImg/data/predefined_classes.txt")
    # Path to new_classes.txt in project
    NEW_CLASSES_PATH = BASE_DIR / "class" / "new_classes.txt"
    
    def load_classes():
        """Load classes from classes.txt"""
        classes = []
        if CLASSES_TXT.exists():
            with open(CLASSES_TXT, "r") as f:
                classes = [line.strip() for line in f if line.strip()]
        return classes
    
    def save_classes(classes_list):
        """Save classes to classes.txt, predefined_classes.txt, and new_classes.txt"""
        # Save to classes.txt
        CLASSES_TXT.parent.mkdir(parents=True, exist_ok=True)
        with open(CLASSES_TXT, "w") as f:
            f.write("\n".join(classes_list) + "\n" if classes_list else "")
        
        # Save to predefined_classes.txt (LabelImg)
        if LABELIMG_CLASSES_PATH.exists():
            with open(LABELIMG_CLASSES_PATH, "w") as f:
                f.write("\n".join(classes_list) + "\n" if classes_list else "")
        
        # Save to new_classes.txt
        NEW_CLASSES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(NEW_CLASSES_PATH, "w") as f:
            f.write("\n".join(classes_list) + "\n" if classes_list else "")
    
    # Load current classes
    current_classes = load_classes()
    
    # Display current classes
    st.markdown("#### Current Classes")
    if current_classes:
        # Create a styled display of classes
        classes_html = '<div style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 0.75rem 0 1.5rem 0;">'
        for idx, cls in enumerate(current_classes):
            classes_html += f'''
            <span style="
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                background: rgba(99, 102, 241, 0.15);
                border: 1px solid rgba(99, 102, 241, 0.3);
                padding: 0.4rem 0.75rem;
                border-radius: 8px;
                font-size: 0.85rem;
                color: #a5b4fc;
            ">
                <span style="color: #64748b; font-size: 0.7rem; font-weight: 600;">{idx}</span>
                {cls}
            </span>'''
        classes_html += '</div>'
        st.markdown(classes_html, unsafe_allow_html=True)
    else:
        st.warning("No classes defined yet. Add classes below.")
    
    # Add/Remove class controls
    col_add, col_remove = st.columns(2)
    
    with col_add:
        st.markdown("##### Add New Class")
        new_class = st.text_input("Class name", key="new_class_input", placeholder="Enter class name...")
        if st.button("➕ Add Class", key="add_class_btn"):
            if new_class.strip():
                new_class_clean = new_class.strip().lower()
                if new_class_clean in [c.lower() for c in current_classes]:
                    st.warning(f"Class '{new_class_clean}' already exists!")
                else:
                    current_classes.append(new_class_clean)
                    save_classes(current_classes)
                    st.success(f"Added class: '{new_class_clean}'")
                    st.rerun()
            else:
                st.warning("Please enter a class name.")
    
    with col_remove:
        st.markdown("##### Remove Class")
        if current_classes:
            class_to_remove = st.selectbox(
                "Select class to remove",
                options=current_classes,
                key="remove_class_select"
            )
            if st.button("🗑️ Remove Class", key="remove_class_btn"):
                if class_to_remove in current_classes:
                    current_classes.remove(class_to_remove)
                    save_classes(current_classes)
                    st.success(f"Removed class: '{class_to_remove}'")
                    st.rerun()
        else:
            st.info("No classes to remove.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ===== END CLASS MANAGEMENT SECTION =====
    
    st.markdown('''
    <div class="workflow-step">
        <div class="step-number">1</div>
        <div class="step-content">
            <div class="step-title">Read Frames</div>
            <div class="step-desc">Loads images from <code>output_frames</code> directory</div>
        </div>
    </div>
    <div class="workflow-step">
        <div class="step-number">2</div>
        <div class="step-content">
            <div class="step-title">YOLO Inference</div>
            <div class="step-desc">Runs object detection on each frame using the trained model</div>
        </div>
    </div>
    <div class="workflow-step">
        <div class="step-number">3</div>
        <div class="step-content">
            <div class="step-title">Export Labels</div>
            <div class="step-desc">Saves bounding boxes in YOLO format to <code>output_annotation</code></div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Run Auto-Annotation", type="primary"):
        with st.spinner("Running YOLO inference..."):
            proc = run_auto_annotation()
        
        if proc.returncode == 0:
            # Clear existing annotations in the default directory
            clear_directory(ANNOT_DIR)
            
            if chosen_annot_dir.resolve() != ANNOT_DIR.resolve():
                # Clear the custom directory as well
                clear_directory(chosen_annot_dir)
                for src_path in ANNOT_DIR.rglob("*"):
                    if src_path.is_dir():
                        continue
                    rel = src_path.relative_to(ANNOT_DIR)
                    dest_path = chosen_annot_dir / rel
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dest_path)
            st.success(f"✓ Cleared previous annotations and saved new results to {chosen_annot_dir}")
            st.balloons()
        else:
            st.error(f"Failed with exit code {proc.returncode}")
        
        with st.expander("Logs"):
            st.code(proc.stdout if proc.stdout else "No output")
            st.code(proc.stderr if proc.stderr else "No errors")

# Tab 4: Preview
with tabs[3]:
    st.markdown('''
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                    <circle cx="12" cy="12" r="3"></circle>
                </svg>
            </div>
            <div>
                <h3 class="section-title">Preview Annotations</h3>
                <p class="section-desc">Visualize bounding boxes overlaid on your images to verify annotation quality</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    annot_dir = Path(st.session_state.get("annot_dir", str(ANNOT_DIR)))
    
    st.markdown("""
    <style>
    .preview-controls {
        background: rgba(17, 24, 39, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin: 1rem 0 1.5rem 0;
    }
    .control-label {
        color: #94a3b8;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col_prev1, col_prev2, col_prev3 = st.columns([2, 1, 1])
    with col_prev1:
        st.markdown('<p class="control-label">Source Directory</p>', unsafe_allow_html=True)
        st.code(str(annot_dir), language=None)
    with col_prev2:
        sample_count = st.number_input(
            "Images to display",
            min_value=3,
            max_value=100,
            value=12,
            step=3,
            key="preview_count"
        )
    with col_prev3:
        cols_per_row = st.selectbox("Grid columns", [2, 3, 4, 5], index=1, key="cols_per_row")
    
    imgs = list_images(annot_dir, limit=sample_count)
    
    if imgs:
        st.success(f"Displaying {len(imgs)} annotated images in {cols_per_row} columns")
        cols = st.columns(cols_per_row)
        for i, p in enumerate(imgs):
            try:
                pil = load_image(p)
                txt = p.with_suffix(".txt")
                drawn = draw_yolo_boxes_on_image(pil, txt)
                with cols[i % cols_per_row]:
                    st.image(drawn, caption=p.name)
            except Exception:
                with cols[i % cols_per_row]:
                    st.error(f"Failed to load {p.name}")
    else:
        st.markdown('''
        <div class="empty-state">
            <div class="empty-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="1.5">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
            </div>
            <div class="empty-title">No Annotated Images Found</div>
            <div class="empty-desc">Run the Auto-Annotation step first to generate labels, then return here to preview them.</div>
        </div>
        ''', unsafe_allow_html=True)

# Tab 5: LabelImg
with tabs[4]:
    st.markdown('''
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                    <path d="M12 19l7-7 3 3-7 7-3-3z"></path>
                    <path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"></path>
                    <path d="M2 2l7.586 7.586"></path>
                    <circle cx="11" cy="11" r="2"></circle>
                </svg>
            </div>
            <div>
                <h3 class="section-title">Manual Refinement with LabelImg</h3>
                <p class="section-desc">Fine-tune auto-generated annotations using the industry-standard LabelImg tool</p>
            </div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    col_l1, col_l2, col_l3 = st.columns(3)
    with col_l1:
        st.metric("Tool", "LabelImg")
    with col_l2:
        st.metric("Format", "YOLO/PascalVOC")
    with col_l3:
        st.metric("Keyboard", "Shortcuts")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Quick Reference")
    st.markdown("""
    | Shortcut | Action |
    |----------|--------|
    | `W` | Create new bounding box |
    | `D` | Next image |
    | `A` | Previous image |
    | `Ctrl+S` | Save annotation |
    """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    annot_dir = Path(st.session_state.get("annot_dir", str(ANNOT_DIR)))
    classes_txt = annot_dir / "classes.txt"
    img_dir = st.text_input("Images directory", value=str(annot_dir), key="labelimg_dir")
    
    if st.button("Launch LabelImg", type="primary"):
        predef = str(classes_txt) if classes_txt.exists() else ""
        
        def try_cmd(cmd: str) -> bool:
            try:
                subprocess.Popen([cmd, img_dir, predef] if predef else [cmd, img_dir])
                return True
            except FileNotFoundError:
                return False
            except Exception as e:
                st.error(f"Failed: {e}")
                return False
        
        if try_cmd("labelImg") or try_cmd("labelimg"):
            st.success("LabelImg launched successfully. Check for a new window.")
        else:
            st.error("LabelImg not found. Install with: pip install labelImg")

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
        <span class="footer-name">ARAS Auto-Annotation Studio</span>
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
