import streamlit as st

st.set_page_config(
    page_title="Percieva™ Auto-Annotation Studio",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import io
import json
import os
import sys
import shutil
import zipfile
import subprocess
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
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

/* ===== GALLERY STYLING ===== */
.gallery-container {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.gallery-image-wrapper {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1.5px solid rgba(148, 163, 184, 0.2);
    border-radius: 18px;
    padding: 1rem;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
}

.gallery-image-wrapper:hover {
    border-color: rgba(6, 182, 212, 0.5);
    box-shadow: 0 12px 40px rgba(6, 182, 212, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    transform: translateY(-2px);
}

.gallery-image-wrapper img {
    border-radius: 14px;
    width: 100%;
    display: block;
}

.gallery-delete-btn {
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: rgba(239, 68, 68, 0.9) !important;
    border: none !important;
    width: 36px !important;
    height: 36px !important;
    padding: 0 !important;
    border-radius: 10px !important;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4) !important;
    z-index: 10;
}

.gallery-delete-btn:hover {
    background: rgba(220, 38, 38, 0.95) !important;
    transform: scale(1.1);
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6) !important;
}

.gallery-delete-btn:active {
    transform: scale(0.95);
}

.gallery-delete-btn span,
.gallery-delete-btn div,
.gallery-delete-btn p {
    color: #ffffff !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
}

.gallery-info-text {
    color: #cbd5e1 !important;
    font-size: 0.9rem;
    font-weight: 500;
    margin: 0.5rem 0 1rem 0;
    text-align: center;
}

.gallery-pagination {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(148, 163, 184, 0.15);
    border-radius: 14px;
    padding: 0.75rem 1rem;
    margin: 1.5rem 0;
    backdrop-filter: blur(10px);
}

.gallery-pagination button {
    background: linear-gradient(135deg, #06b6d4 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.3) !important;
}

.gallery-pagination button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.4) !important;
}

/* Gallery Delete Button Styling */
[data-testid="baseButton-secondary"] {
    background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(6, 182, 212, 0.05) 100%) !important;
    border: 1.5px solid rgba(139, 92, 246, 0.3) !important;
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    padding: 0.6rem 1.2rem !important;
}

[data-testid="baseButton-secondary"]:hover {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.05) 100%) !important;
    border-color: rgba(239, 68, 68, 0.5) !important;
    color: #ef4444 !important;
    box-shadow: 0 6px 20px rgba(239, 68, 68, 0.2) !important;
    transform: translateY(-2px) !important;
}

.gallery-image-container {
    position: relative;
    width: 100%;
    border-radius: 14px;
    overflow: hidden;
}

.gallery-image-container:hover .gallery-image-actions {
    opacity: 1;
    pointer-events: auto;
}

.gallery-image-actions {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
    display: flex;
    gap: 0.5rem;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s ease;
    z-index: 10;
}

.gallery-btn {
    background: rgba(30, 41, 59, 0.9) !important;
    border: 1px solid rgba(6, 182, 212, 0.4) !important;
    width: 32px !important;
    height: 32px !important;
    padding: 0 !important;
    border-radius: 8px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3) !important;
}

.gallery-btn:hover {
    background: rgba(6, 182, 212, 0.9) !important;
    border-color: rgba(139, 92, 246, 0.6) !important;
    transform: scale(1.1) !important;
    box-shadow: 0 4px 12px rgba(6, 182, 212, 0.4) !important;
}

.gallery-delete-btn-hover {
    background: rgba(239, 68, 68, 0.9) !important;
    border: 1px solid rgba(239, 68, 68, 0.6) !important;
}

.gallery-delete-btn-hover:hover {
    background: rgba(220, 38, 38, 0.95) !important;
    border-color: rgba(239, 68, 68, 0.8) !important;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.5) !important;
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
if page_param == "annotate":
    current_page = "Annotate"
elif page_param == "model":
    current_page = "Model Comparison"
elif page_param in ["insights", "analytics"]:
    current_page = "Insights"
else:
    current_page = "Annotate"

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
    
    # Model Comparison Button
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
        if st.button("Model Comparison", key="nav_model", use_container_width=True):
            st.query_params["page"] = "model"
            st.rerun()
        if model_active:
            st.markdown('<div style="height: 2px; background: linear-gradient(90deg, #06b6d4, #8b5cf6); margin-top: -8px; border-radius: 1px;"></div>', unsafe_allow_html=True)
    
    # Insights Button
    analytics_active = page_param in ["insights", "analytics"]
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
        if st.button("Insights", key="nav_analytics", use_container_width=True):
            st.query_params["page"] = "insights"
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
    
    compare_base_dir = BASE_DIR / "Model_Compare"
    filter_new_model_dir = compare_base_dir / "new_model"
    filter_yolo_model_dir = compare_base_dir / "yolo model"
    filter_runner_script = BASE_DIR.parent / "performance_testing" / "filter_frames_by_model_gap.py"

    def latest_model_file(model_dir: Path):
        """Return the most recently modified `.pt` model file in a directory."""
        model_dir.mkdir(parents=True, exist_ok=True)
        model_files = sorted(model_dir.glob("*.pt"), key=lambda path: path.stat().st_mtime)
        return model_files[-1] if model_files else None

    def list_yolo_models(model_dir: Path):
        """List YOLO baseline models, preferring filenames containing `yolo`."""
        model_dir.mkdir(parents=True, exist_ok=True)
        all_models = sorted(model_dir.glob("*.pt"))
        yolo_models = [path for path in all_models if "yolo" in path.stem.lower()]
        return yolo_models if yolo_models else all_models

    # Tabs for different sections
    tabs = st.tabs(["Upload", "Filter", "Augment", "Image Gallery", "Auto-Annotate", "Annotated Gallery"])
    
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
    
    # TAB 2: FILTER
    with tabs[1]:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">
                <div class="section-icon">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                        <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"></polygon>
                    </svg>
                </div>
                <div>
                    <h3 class="section-title">Model-Based Frame Filter</h3>
                    <p class="section-desc">Keep only frames where latest model performs worse than YOLO</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        latest_new_model = latest_model_file(filter_new_model_dir)
        yolo_candidates = list_yolo_models(filter_yolo_model_dir)

        source_filter_dir = st.text_input("Source frames folder", value=str(FRAMES_DIR), key="filter_source_folder")
        destination_filter_dir = st.text_input("Destination folder", value=str(ANNOT_DIR), key="filter_destination_folder")

        if latest_new_model:
            st.info(f"Latest new model: `{latest_new_model.name}`")
        else:
            st.warning(f"No `.pt` model found in `{filter_new_model_dir}`")

        if yolo_candidates:
            yolo_model_choice = st.selectbox(
                "YOLO model",
                options=yolo_candidates,
                format_func=lambda path: path.name,
                key="filter_yolo_model_select",
            )
        else:
            yolo_model_choice = None
            st.warning(f"No `.pt` model found in `{filter_yolo_model_dir}`")

        st.caption("Threshold guidance: adjust these to control how strict the filter is when deciding whether your model is worse than YOLO.")
        conf_threshold_filter = st.slider(
            "Confidence threshold",
            0.05,
            0.95,
            0.25,
            0.05,
            key="filter_conf_threshold",
            help="Only detections with confidence above this value are considered. Higher values keep fewer boxes (stricter), lower values keep more boxes (looser).",
        )
        iou_threshold_filter = st.slider(
            "IoU threshold",
            0.10,
            0.95,
            0.40,
            0.05,
            key="filter_iou_threshold",
            help="Minimum overlap needed to treat boxes as matching. Higher values require tighter box alignment, lower values allow looser matches.",
        )
        st.info(
            "Higher confidence or higher IoU generally marks more frames as poor (more strict). "
            "Lower confidence or lower IoU generally marks fewer frames as poor (more lenient)."
        )

        if st.button("Run Filter", type="primary", use_container_width=True):
            if latest_new_model is None:
                st.error("No latest model found in new_model folder.")
            elif yolo_model_choice is None:
                st.error("No YOLO model found in model folder.")
            elif not filter_runner_script.exists():
                st.error(f"Filter runner not found: {filter_runner_script}")
            else:
                src_path = Path(source_filter_dir)
                if not src_path.exists():
                    st.error(f"Source folder does not exist: {src_path}")
                else:
                    with st.spinner("Filtering frames where latest model is worse than YOLO..."):
                        command = [
                            sys.executable,
                            str(filter_runner_script),
                            "--mode",
                            "filter",
                            "--new-model",
                            str(latest_new_model),
                            "--yolo-model",
                            str(yolo_model_choice),
                            "--source-dir",
                            str(src_path),
                            "--destination-dir",
                            str(destination_filter_dir),
                            "--conf-thresh",
                            str(conf_threshold_filter),
                            "--iou-thresh",
                            str(iou_threshold_filter),
                            "--clear-destination",
                        ]

                        proc = subprocess.run(
                            command,
                            cwd=str(BASE_DIR.parent),
                            capture_output=True,
                            text=True,
                        )

                    if proc.returncode != 0:
                        st.error("Frame filtering failed.")
                        if proc.stderr:
                            st.code(proc.stderr)
                    else:
                        summary = None
                        output_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
                        for line in reversed(output_lines):
                            try:
                                summary = json.loads(line)
                                break
                            except Exception:
                                continue

                        if summary is None:
                            st.warning("Filter completed, but summary was not found.")
                            if proc.stdout:
                                st.code(proc.stdout)
                        else:
                            st.success("✓ Filtering completed. Destination now contains only poor-performing frames.")
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            with metric_col1:
                                st.metric("Total Images", int(summary.get("total_images", 0)))
                            with metric_col2:
                                st.metric("Selected (Poor)", int(summary.get("selected_images", 0)))
                            with metric_col3:
                                st.metric("Ignored", int(summary.get("ignored_images", 0)))
                            st.caption(f"Saved to: {summary.get('destination_dir', destination_filter_dir)}")

    # TAB 3: AUGMENT
    with tabs[2]:
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
    
    # TAB 4: IMAGE GALLERY (BEFORE ANNOTATION)
    with tabs[3]:
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
                    <p class="section-desc">Preview and manage your dataset before annotation</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        gallery_dir = Path(st.session_state.get("frames_dir", str(FRAMES_DIR)))
        IMAGES_PER_PAGE = 12
        
        def natural_sort_key(path):
            """Sort file names naturally (e.g., frame2 before frame10)."""
            import re
            name = path.name
            return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', name)]
        
        def get_all_images_gallery(dir_path):
            """Collect all gallery images recursively from the selected frame directory."""
            imgs = []
            if dir_path.exists():
                for root, _, files in os.walk(dir_path):
                    for f in sorted(files):
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            imgs.append(Path(root) / f)
            return sorted(imgs, key=natural_sort_key)
        
        def load_image_gallery(path):
            """Load one image as RGB PIL for Streamlit display."""
            img = cv2.imread(str(path))
            if img is None:
                return None
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        cols_per_row = st.selectbox("Grid columns", [2, 3, 4, 5], 1, key="gallery_cols_per_row")
        
        all_imgs = get_all_images_gallery(gallery_dir)
        total_pages = max(1, (len(all_imgs) + IMAGES_PER_PAGE - 1) // IMAGES_PER_PAGE)
        
        if "gallery_page" not in st.session_state:
            st.session_state["gallery_page"] = 1
        
        if all_imgs:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; backdrop-filter: blur(5px);">
                <p style="color: #cbd5e1; margin: 0; font-weight: 600;">📊 Total images: <span style="color: #06b6d4; font-weight: 800;">{len(all_imgs)}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Pagination controls with buttons that work immediately
            col_prev, col_page, col_next = st.columns([0.8, 2, 0.8])
            with col_prev:
                def on_prev_click():
                    """Move image gallery pagination to previous page."""
                    if st.session_state["gallery_page"] > 1:
                        st.session_state["gallery_page"] -= 1
                st.button("◀ Previous", key="gallery_prev_btn", use_container_width=True, on_click=on_prev_click)
            
            with col_page:
                pages = list(range(1, total_pages + 1))
                def on_page_change():
                    """Sync image gallery page selector with session state."""
                    st.session_state["gallery_page"] = st.session_state["gallery_page_select"]
                current_page = st.selectbox(
                    "Page",
                    pages,
                    index=min(st.session_state["gallery_page"] - 1, len(pages) - 1),
                    key="gallery_page_select",
                    label_visibility="collapsed",
                    on_change=on_page_change
                )
                st.markdown(f"<p style='text-align: center; color: #94a3b8; margin: 0.5rem 0; font-size: 0.9rem;'>Page <span style='color: #06b6d4; font-weight: 800;'>{st.session_state['gallery_page']}</span> of <span style='color: #8b5cf6; font-weight: 800;'>{total_pages}</span></p>", unsafe_allow_html=True)
            
            with col_next:
                def on_next_click():
                    """Move image gallery pagination to next page."""
                    if st.session_state["gallery_page"] < total_pages:
                        st.session_state["gallery_page"] += 1
                st.button("Next ▶", key="gallery_next_btn", use_container_width=True, on_click=on_next_click)
            
            st.divider()
            
            # Display images for current page
            start_idx = (st.session_state["gallery_page"] - 1) * IMAGES_PER_PAGE
            end_idx = min(start_idx + IMAGES_PER_PAGE, len(all_imgs))
            page_imgs = all_imgs[start_idx:end_idx]
            
            st.markdown(f"<p class='gallery-info-text'>Displaying images {start_idx + 1}-{end_idx} of {len(all_imgs)} • {cols_per_row} columns</p>", unsafe_allow_html=True)
            cols = st.columns(cols_per_row)
            
            for i, p in enumerate(page_imgs):
                col_idx = i % cols_per_row
                with cols[col_idx]:
                    pil = load_image_gallery(p)
                    if pil:
                        st.image(pil, use_container_width=True)
                        
                        # Delete button below image - neatly placed and centered
                        st.markdown('<div style="text-align: center; margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
                        col_spacer1, col_del, col_spacer2 = st.columns([1, 2, 1])
                        with col_del:
                            if st.button("🗑️ Delete", key=f"del_gallery_{i}_{start_idx}", use_container_width=True, help=f"Delete {p.name}"):
                                try:
                                    p.unlink()
                                    st.success(f"✓ Deleted {p.name}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)[:50]}")
                    else:
                        st.error(f"Failed to load {p.name}")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: rgba(15, 23, 42, 0.4); border: 2px dashed rgba(148, 163, 184, 0.2); border-radius: 16px;">
                <p style="color: #94a3b8; font-size: 1.1rem; margin: 0;">📷 No images found</p>
                <p style="color: #64748b; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Start by uploading images or extracting frames from a video</p>
            </div>
            """, unsafe_allow_html=True)
    
    # TAB 5: AUTO-ANNOTATE
    with tabs[4]:
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
            """Load currently configured class list from classes.txt."""
            if CLASSES_TXT.exists():
                with open(CLASSES_TXT) as f:
                    return [line.strip() for line in f if line.strip()]
            return []
        
        def save_classes(classes_list):
            """Persist class list to classes.txt used by auto-annotation."""
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
                """Quickly sample image files to validate frame availability."""
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
    
    # TAB 6: ANNOTATED GALLERY (AFTER ANNOTATION)
    with tabs[5]:
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
        
        def get_all_annotated_images_ann(dir_path):
            """Collect all annotated images recursively for review gallery."""
            imgs = []
            if dir_path.exists():
                for root, _, files in os.walk(dir_path):
                    for f in sorted(files):
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                            imgs.append(Path(root) / f)
            return sorted(imgs, key=natural_sort_key)
        
        def load_image_ann(path):
            """Load one annotated image as RGB PIL."""
            img = cv2.imread(str(path))
            if img is None:
                return None
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        def draw_boxes_ann(img_pil, txt_path):
            """Render YOLO txt boxes on top of a preview image."""
            img = np.array(img_pil).copy()
            h, w = img.shape[:2]
            if txt_path.exists():
                try:
                    with open(txt_path) as f:
                        lines = [ln.strip() for ln in f if ln.strip()]
                    for ln in lines:
                        parts = ln.split()
                        if len(parts) == 5:
                            _, cx, cy, bw, bh = map(float, parts)
                            x = int((cx - bw / 2) * w)
                            y = int((cy - bh / 2) * h)
                            x2 = int((cx + bw / 2) * w)
                            y2 = int((cy + bh / 2) * h)
                            cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
                except:
                    pass
            return Image.fromarray(img)
        
        cols_per_row = st.selectbox("Grid columns", [2, 3, 4, 5], 1, key="annotated_cols_per_row")
        
        all_annotated_imgs = get_all_annotated_images_ann(preview_dir)
        total_annotated_pages = max(1, (len(all_annotated_imgs) + ANNOTATED_IMAGES_PER_PAGE - 1) // ANNOTATED_IMAGES_PER_PAGE)
        
        if "annotated_page" not in st.session_state:
            st.session_state["annotated_page"] = 1
        
        if all_annotated_imgs:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(139, 92, 246, 0.05) 100%); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 12px; padding: 1rem; margin-bottom: 1.5rem; backdrop-filter: blur(5px);">
                <p style="color: #cbd5e1; margin: 0; font-weight: 600;">🏷️ Total annotated images: <span style="color: #06b6d4; font-weight: 800;">{len(all_annotated_imgs)}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Pagination controls with callbacks
            col_prev, col_page, col_next = st.columns([0.8, 2, 0.8])
            with col_prev:
                def on_prev_click_ann():
                    """Move annotated gallery pagination to previous page."""
                    if st.session_state["annotated_page"] > 1:
                        st.session_state["annotated_page"] -= 1
                st.button("◀ Previous", key="annotated_prev_btn", use_container_width=True, on_click=on_prev_click_ann)
            
            with col_page:
                pages = list(range(1, total_annotated_pages + 1))
                def on_page_change_ann():
                    """Sync annotated gallery page selector with session state."""
                    st.session_state["annotated_page"] = st.session_state["annotated_page_select"]
                current_page = st.selectbox(
                    "Page",
                    pages,
                    index=min(st.session_state["annotated_page"] - 1, len(pages) - 1),
                    key="annotated_page_select",
                    label_visibility="collapsed",
                    on_change=on_page_change_ann
                )
                st.markdown(f"<p style='text-align: center; color: #94a3b8; margin: 0.5rem 0; font-size: 0.9rem;'>Page <span style='color: #06b6d4; font-weight: 800;'>{st.session_state['annotated_page']}</span> of <span style='color: #8b5cf6; font-weight: 800;'>{total_annotated_pages}</span></p>", unsafe_allow_html=True)
            
            with col_next:
                def on_next_click_ann():
                    """Move annotated gallery pagination to next page."""
                    if st.session_state["annotated_page"] < total_annotated_pages:
                        st.session_state["annotated_page"] += 1
                st.button("Next ▶", key="annotated_next_btn", use_container_width=True, on_click=on_next_click_ann)
            
            st.divider()
            
            # Display images for current page
            start_idx = (st.session_state["annotated_page"] - 1) * ANNOTATED_IMAGES_PER_PAGE
            end_idx = min(start_idx + ANNOTATED_IMAGES_PER_PAGE, len(all_annotated_imgs))
            page_annotated_imgs = all_annotated_imgs[start_idx:end_idx]
            
            st.markdown(f"<p class='gallery-info-text'>Displaying images {start_idx + 1}-{end_idx} of {len(all_annotated_imgs)} • {cols_per_row} columns</p>", unsafe_allow_html=True)
            cols = st.columns(cols_per_row)
            
            for i, p in enumerate(page_annotated_imgs):
                col_idx = i % cols_per_row
                with cols[col_idx]:
                    pil = load_image_ann(p)
                    if pil:
                        drawn = draw_boxes_ann(pil, p.with_suffix(".txt"))
                        st.image(drawn, use_container_width=True)
                        
                        # Delete button below image - neatly placed and centered
                        st.markdown('<div style="text-align: center; margin-top: 0.5rem;"></div>', unsafe_allow_html=True)
                        col_spacer1, col_del, col_spacer2 = st.columns([1, 2, 1])
                        with col_del:
                            if st.button("🗑️ Delete", key=f"del_annotated_{i}_{start_idx}", use_container_width=True, help="Delete image and annotation"):
                                try:
                                    txt_file = p.with_suffix(".txt")
                                    p.unlink()
                                    if txt_file.exists():
                                        txt_file.unlink()
                                    st.success(f"✓ Deleted {p.name}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)[:50]}")
                    else:
                        st.error(f"Failed to load {p.name}")
        else:
            st.info("No annotated images yet")

# MODEL COMPARISON PAGE
elif current_page == "Model Comparison":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:inline;vertical-align:middle;margin-right:8px;">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M12 1v6m0 6v6m5.2-13.2l-4.2 4.2m0 6l4.2 4.2M23 12h-6m-6 0H5m13.2 5.2l-4.2-4.2m0-6l4.2-4.2"></path>
            </svg>
            Comparison
        </h1>
        <p class="hero-subtitle">Benchmark YOLO models against ground truth with clear per-class insights</p>
        <div class="hero-badges">
            <span class="hero-badge">Ground Truth Eval</span>
            <span class="hero-badge">Per-Class Matrix</span>
            <span class="hero-badge">Metrics Timeline</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Paths - Comparison folder structure
    COMPARE_BASE_DIR = Path(__file__).resolve().parent / "Model_Compare"
    MODELS_DIR = COMPARE_BASE_DIR / "model"
    NEW_MODEL_DIR = COMPARE_BASE_DIR / "new_model"
    GROUND_TRUTH_DIR = COMPARE_BASE_DIR / "ground_truth"
    COMPARE_OUTPUT_DIR = COMPARE_BASE_DIR / "output"
    METRICS_CSV = COMPARE_BASE_DIR / "metrics.csv"
    
    # Create directories if they don't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    NEW_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    COMPARE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Helper functions
    def get_available_models():
        """Return model names available for evaluation in comparison page."""
        models = []
        # Get models from model folder (looking for .pt files)
        if MODELS_DIR.exists():
            pt_files = sorted(MODELS_DIR.glob("*.pt"))
            for pt_file in pt_files:
                model_name = pt_file.stem  # Get filename without extension (v1, v2, etc.)
                models.append(model_name)
        # Add new_model if it has .pt files
        if NEW_MODEL_DIR.exists():
            new_pt_files = list(NEW_MODEL_DIR.glob("*.pt"))
            if new_pt_files:
                models.insert(0, "Latest (new_model)")
        return models
    
    def get_model_output_frames(model_name):
        """Get previewable comparison output frames for selected model."""
        # Handle "Latest (new_model)" special case
        if model_name == "Latest (new_model)":
            model_output_dir = COMPARE_OUTPUT_DIR / "new_model"
        else:
            model_output_dir = COMPARE_OUTPUT_DIR / model_name
        
        if model_output_dir.exists():
            frames = sorted(model_output_dir.glob("*.jpg")) + sorted(model_output_dir.glob("*.png"))
            return frames[:6]  # Return first 6 frames for preview
        return []

    def get_latest_new_model_info():
        """Return metadata for the newest model in `new_model` directory."""
        if not NEW_MODEL_DIR.exists():
            return None
        latest_models = sorted(NEW_MODEL_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
        if not latest_models:
            return None
        latest = latest_models[-1]
        return {
            "name": latest.stem,
            "path": latest,
            "updated": datetime.fromtimestamp(latest.stat().st_mtime)
        }

    def get_class_label_map():
        """Load class id -> label map from comparison ground-truth classes file."""
        class_file = GROUND_TRUTH_DIR / "class" / "classes.txt"
        class_map = {}
        if class_file.exists():
            try:
                with open(class_file, "r") as f:
                    for idx, line in enumerate(f):
                        label = line.strip()
                        if label:
                            class_map[idx] = label
            except Exception:
                pass
        return class_map

    def metric_safe_label(label):
        """Normalize a class label into a metrics-friendly key suffix."""
        return str(label).strip().lower().replace(" ", "_").replace("-", "_")
    
    def parse_yolo_annotation(txt_path):
        """Parse YOLO txt format: class_id x_center y_center width height."""
        boxes = []
        if not txt_path.exists():
            return boxes
        try:
            with open(txt_path) as f:
                for line in f:
                    if line.strip():
                        parts = [float(x) for x in line.strip().split()]
                        if len(parts) >= 5:
                            boxes.append((
                                int(parts[0]),
                                parts[1],
                                parts[2],
                                parts[3],
                                parts[4]
                            ))
        except:
            pass
        return boxes
    
    def compare_annotations(gt_txt, pred_txt, img_shape):
        """Compare GT and predictions and compute match/FP/FN statistics."""
        gt_boxes = parse_yolo_annotation(gt_txt)
        pred_boxes = parse_yolo_annotation(pred_txt)
        
        # Convert normalized coordinates to pixel coordinates
        w, h = img_shape
        gt_pixel_boxes = []
        for box in gt_boxes:
            class_id, x_c, y_c, bw, bh = box
            x1 = max(0, (x_c - bw/2) * w)
            y1 = max(0, (y_c - bh/2) * h)
            x2 = min(w, (x_c + bw/2) * w)
            y2 = min(h, (y_c + bh/2) * h)
            gt_pixel_boxes.append((class_id, x1, y1, x2, y2))
        
        pred_pixel_boxes = []
        for box in pred_boxes:
            class_id, x_c, y_c, bw, bh = box
            x1 = max(0, (x_c - bw/2) * w)
            y1 = max(0, (y_c - bh/2) * h)
            x2 = min(w, (x_c + bw/2) * w)
            y2 = min(h, (y_c + bh/2) * h)
            pred_pixel_boxes.append((class_id, x1, y1, x2, y2))
        
        # Calculate IoU
        def calculate_iou(box1, box2):
            """Calculate IoU between two class-aware pixel-coordinate boxes."""
            _, x1_1, y1_1, x2_1, y2_1 = box1
            _, x1_2, y1_2, x2_2, y2_2 = box2
            
            # Intersection
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            
            # Union
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0
        
        # Match boxes with IoU threshold 0.5
        iou_threshold = 0.5
        matched_pairs = []
        used_pred = set()
        
        for gt_box in gt_pixel_boxes:
            best_iou = 0
            best_pred_idx = -1
            
            for pred_idx, pred_box in enumerate(pred_pixel_boxes):
                if pred_idx in used_pred:
                    continue
                
                # Only match boxes of same class
                if gt_box[0] != pred_box[0]:
                    continue
                
                iou = calculate_iou(gt_box, pred_box)
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            if best_pred_idx >= 0:
                matched_pairs.append((gt_box, pred_pixel_boxes[best_pred_idx]))
                used_pred.add(best_pred_idx)
        
        per_class = {}
        for cls, *_ in gt_pixel_boxes:
            per_class.setdefault(cls, {'tp': 0, 'fp': 0, 'fn': 0})
        for cls, *_ in pred_pixel_boxes:
            per_class.setdefault(cls, {'tp': 0, 'fp': 0, 'fn': 0})

        matched_pred_indices = set()
        matched_gt_indices = set()
        for gt_box, pred_box in matched_pairs:
            gt_idx = gt_pixel_boxes.index(gt_box)
            pred_idx = pred_pixel_boxes.index(pred_box)
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(pred_idx)
            cls_id = gt_box[0]
            per_class.setdefault(cls_id, {'tp': 0, 'fp': 0, 'fn': 0})
            per_class[cls_id]['tp'] += 1

        for pred_idx, pred_box in enumerate(pred_pixel_boxes):
            if pred_idx not in matched_pred_indices:
                cls_id = pred_box[0]
                per_class.setdefault(cls_id, {'tp': 0, 'fp': 0, 'fn': 0})
                per_class[cls_id]['fp'] += 1

        for gt_idx, gt_box in enumerate(gt_pixel_boxes):
            if gt_idx not in matched_gt_indices:
                cls_id = gt_box[0]
                per_class.setdefault(cls_id, {'tp': 0, 'fp': 0, 'fn': 0})
                per_class[cls_id]['fn'] += 1

        metrics = {
            'gt_count': len(gt_boxes),
            'pred_count': len(pred_boxes),
            'matches': len(matched_pairs),
            'false_positives': len(pred_boxes) - len(matched_pairs),
            'false_negatives': len(gt_boxes) - len(matched_pairs),
            'per_class': per_class
        }
        return metrics
    
    def run_model_comparison(model_name):
        """Run full model evaluation against comparison ground-truth dataset."""
        from ultralytics import YOLO
        
        results = {
            'model': model_name,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'total_frames': 0,
            'matched_boxes': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get model path
        if model_name == "Latest (new_model)":
            latest_models = sorted(NEW_MODEL_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
            if not latest_models:
                return results
            model_path = latest_models[-1]
            output_model_name = "new_model"
        else:
            model_path = MODELS_DIR / f"{model_name}.pt"
            output_model_name = model_name
        
        if not model_path.exists():
            return results
        
        # Load model
        try:
            model = YOLO(str(model_path))
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return results
        
        # Get ground truth images
        gt_imgs = sorted(list(GROUND_TRUTH_DIR.glob("*.jpg")) + list(GROUND_TRUTH_DIR.glob("*.png")))
        
        # Create output directory for predictions
        pred_output_dir = COMPARE_OUTPUT_DIR / output_model_name
        pred_output_dir.mkdir(parents=True, exist_ok=True)
        
        total_matches = 0
        total_fp = 0
        total_fn = 0
        class_label_map = get_class_label_map()
        per_class_totals = {}
        frame_errors = 0
        
        for gt_img in gt_imgs:
            gt_txt = gt_img.with_suffix(".txt")
            
            if not gt_txt.exists():
                continue
            
            # Run inference on image
            try:
                results_yolo = model(str(gt_img), conf=0.5, verbose=False)
                
                # Save predictions to txt file
                pred_txt = pred_output_dir / f"{gt_img.stem}.txt"
                pred_txt.parent.mkdir(parents=True, exist_ok=True)
                
                # Extract predictions from YOLO results
                predictions = []
                if len(results_yolo) > 0 and results_yolo[0].boxes is not None:
                    boxes = results_yolo[0].boxes
                    h, w = results_yolo[0].orig_shape
                    
                    for box in boxes:
                        # Convert to YOLO format (normalized center coordinates)
                        x_center = (box.xywh[0][0] / w).item()
                        y_center = (box.xywh[0][1] / h).item()
                        box_width = (box.xywh[0][2] / w).item()
                        box_height = (box.xywh[0][3] / h).item()
                        class_id = int(box.cls[0].item())
                        predictions.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
                
                # Write predictions
                with open(pred_txt, 'w') as f:
                    f.writelines(predictions)
                
                # Compare annotations
                h, w = results_yolo[0].orig_shape if len(results_yolo) > 0 else (0, 0)
                metrics = compare_annotations(gt_txt, pred_txt, (w, h))
                total_matches += metrics['matches']
                total_fp += metrics['false_positives']
                total_fn += metrics['false_negatives']
                results['total_frames'] += 1

                for class_id, class_stats in metrics.get('per_class', {}).items():
                    per_class_totals.setdefault(class_id, {'tp': 0, 'fp': 0, 'fn': 0})
                    per_class_totals[class_id]['tp'] += class_stats.get('tp', 0)
                    per_class_totals[class_id]['fp'] += class_stats.get('fp', 0)
                    per_class_totals[class_id]['fn'] += class_stats.get('fn', 0)

                preview_img = cv2.imread(str(gt_img))
                if preview_img is not None:
                    gt_boxes = parse_yolo_annotation(gt_txt)
                    for class_id, x_c, y_c, bw, bh in gt_boxes:
                        x1 = int(max(0, (x_c - bw / 2) * w))
                        y1 = int(max(0, (y_c - bh / 2) * h))
                        x2 = int(min(w, (x_c + bw / 2) * w))
                        y2 = int(min(h, (y_c + bh / 2) * h))
                        label = class_label_map.get(class_id, f"class_{class_id}")
                        cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(preview_img, f"GT:{label}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                    if len(results_yolo) > 0 and results_yolo[0].boxes is not None:
                        for box in results_yolo[0].boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            class_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            label = class_label_map.get(class_id, f"class_{class_id}")
                            cv2.rectangle(preview_img, (int(x1), int(y1)), (int(x2), int(y2)), (32, 64, 255), 2)
                            cv2.putText(preview_img, f"P:{label} {conf:.2f}", (int(x1), min(h - 8, int(y2) + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (32, 64, 255), 1)

                    cv2.imwrite(str(pred_output_dir / gt_img.name), preview_img)
                
            except Exception as e:
                frame_errors += 1
                continue
        
        # Calculate metrics
        results['matched_boxes'] = total_matches
        results['false_positives'] = total_fp
        results['false_negatives'] = total_fn
        results['frame_errors'] = frame_errors
        
        if total_matches + total_fp > 0:
            results['precision'] = total_matches / (total_matches + total_fp)
        if total_matches + total_fn > 0:
            results['recall'] = total_matches / (total_matches + total_fn)
        if results['precision'] + results['recall'] > 0:
            results['f1_score'] = 2 * (results['precision'] * results['recall']) / (results['precision'] + results['recall'])

        for class_id, stats in per_class_totals.items():
            tp = stats.get('tp', 0)
            fp = stats.get('fp', 0)
            fn = stats.get('fn', 0)
            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            class_name = metric_safe_label(class_label_map.get(class_id, f"class_{class_id}"))
            results[f'precision_{class_name}'] = class_precision
            results[f'recall_{class_name}'] = class_recall
        
        return results
    
    def save_metrics_to_csv(metrics_dict):
        """Append one metrics row to the comparison metrics CSV."""
        df_new = pd.DataFrame([metrics_dict])
        if METRICS_CSV.exists():
            df_existing = pd.read_csv(METRICS_CSV)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new
        df_combined.to_csv(METRICS_CSV, index=False)

    def resolve_metric_columns(df):
        """Support both legacy and current metric column naming in dashboards."""
        model_col = 'model_name' if 'model_name' in df.columns else 'model'
        date_col = 'date' if 'date' in df.columns else 'timestamp'
        prec_col = 'overall_precision' if 'overall_precision' in df.columns else 'precision'
        rec_col = 'overall_recall' if 'overall_recall' in df.columns else 'recall'
        return model_col, date_col, prec_col, rec_col

    def build_per_class_table(metrics_row):
        """Build display-ready per-class precision/recall/F1 table from one run."""
        class_rows = []
        for col in metrics_row.index:
            if not col.startswith('precision_'):
                continue
            cls = col.replace('precision_', '')
            rec_col = f"recall_{cls}"
            if rec_col not in metrics_row.index:
                continue

            precision_val = metrics_row.get(col, np.nan)
            recall_val = metrics_row.get(rec_col, np.nan)

            if pd.isna(precision_val) and pd.isna(recall_val):
                continue

            precision_num = float(precision_val) if pd.notna(precision_val) else 0.0
            recall_num = float(recall_val) if pd.notna(recall_val) else 0.0
            f1_num = (2 * precision_num * recall_num / (precision_num + recall_num)) if (precision_num + recall_num) > 0 else 0.0

            class_rows.append({
                'Class': cls.replace('_', ' ').title(),
                'Precision': precision_num,
                'Recall': recall_num,
                'F1': f1_num
            })

        if not class_rows:
            return pd.DataFrame()

        return pd.DataFrame(class_rows).sort_values('F1', ascending=False).reset_index(drop=True)
    
    # Get available models
    available_models = get_available_models()
    latest_new_model = get_latest_new_model_info()
    
    if not available_models:
        st.warning("No models found in models folder")

    st.markdown("""
    <div class="section-card" style="margin-top: 0.5rem; padding: 1.25rem 1.5rem;">
        <div style="display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;">
            <div>
                <h3 style="margin: 0; color: #f8fafc; font-size: 1.1rem; font-weight: 700;">Run Model Evaluation</h3>
                <p style="margin: 0.35rem 0 0; color: #cbd5e1; font-size: 0.92rem;">Evaluates the selected model on ground truth and appends metrics to the CSV.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if available_models:
        col_eval_1, col_eval_2 = st.columns([2, 1])
        with col_eval_1:
            selected_model = st.selectbox("Model to evaluate", available_models, key="model_eval_select")
        with col_eval_2:
            if st.button("Run Evaluation", type="primary", use_container_width=True):
                if selected_model:
                    with st.spinner(f"Comparing {selected_model} with ground truth..."):
                        results = run_model_comparison(selected_model)
                        total_activity = results['matched_boxes'] + results['false_positives'] + results['false_negatives']
                        if results.get('total_frames', 0) == 0:
                            st.error("No frames were processed. Metrics entry was not saved.")
                        elif total_activity == 0:
                            st.warning("Evaluation found no GT/prediction boxes. Metrics entry was not saved.")
                        else:
                            save_metrics_to_csv(results)
                            st.success(f"✓ Comparison completed!\nPrecision: {results['precision']:.2%} | Recall: {results['recall']:.2%} | F1: {results['f1_score']:.2%}")
                else:
                    st.error("Please select a model first")

    st.divider()
    
    # Load metrics once for this page
    metrics_df = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else None

    # Display selected model with preview frames and metrics
    if available_models:
        st.markdown("### Model Insights")

        view_model = st.selectbox(
            "Choose version to inspect",
            available_models,
            index=0,
            key="model_compare_view_select",
            format_func=lambda name: f"Latest (new_model) • {latest_new_model['name']}" if (name == "Latest (new_model)" and latest_new_model) else name
        )

        latest_by_model = {}
        if metrics_df is not None and len(metrics_df) > 0:
            model_col, date_col, prec_col, rec_col = resolve_metric_columns(metrics_df)
            metrics_df_sorted = metrics_df.copy()
            metrics_df_sorted[date_col] = pd.to_datetime(metrics_df_sorted[date_col], errors='coerce')
            latest_rows = metrics_df_sorted.sort_values(date_col).groupby(model_col, as_index=False).tail(1)
            for _, row in latest_rows.iterrows():
                latest_by_model[row[model_col]] = row

        with st.container(border=True):
            col_title, col_metrics = st.columns([1, 3])

            with col_title:
                if view_model == "Latest (new_model)":
                    label = "Latest (new_model)"
                    if latest_new_model:
                        st.markdown(f"**{label}**")
                        st.caption(f"Active file: {latest_new_model['name']}.pt")
                        st.caption(f"Updated: {latest_new_model['updated'].strftime('%Y-%m-%d %H:%M:%S')}")
                    else:
                        st.markdown(f"**{label}**")
                        st.caption("No model file found in new_model folder.")
                else:
                    st.markdown(f"**{view_model}**")

                if view_model in latest_by_model:
                    latest = latest_by_model[view_model]
                    overall_precision = float(latest.get(prec_col, 0.0))
                    overall_recall = float(latest.get(rec_col, 0.0))
                    overall_f1 = (2 * overall_precision * overall_recall / (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0.0

                    st.metric("Overall Precision", f"{overall_precision:.2%}")
                    st.metric("Overall Recall", f"{overall_recall:.2%}")
                    st.metric("Overall F1", f"{overall_f1:.2%}")
                else:
                    st.info("No metrics run yet")

            with col_metrics:
                metric_tab, frames_tab = st.tabs(["Per-Class Metrics", "Comparison Frames"])

                with metric_tab:
                    if view_model in latest_by_model:
                        per_class_df = build_per_class_table(latest_by_model[view_model])
                        if not per_class_df.empty:
                            chart_df = per_class_df.set_index('Class')[['Precision', 'Recall', 'F1']]
                            st.bar_chart(chart_df)
                            st.dataframe(
                                per_class_df,
                                use_container_width=True,
                                hide_index=True,
                                height=min(420, 70 + (len(per_class_df) * 36)),
                                column_config={
                                    "Class": st.column_config.TextColumn("Class", width="medium"),
                                    "Precision": st.column_config.ProgressColumn("Precision", format="%.1f%%", min_value=0.0, max_value=1.0),
                                    "Recall": st.column_config.ProgressColumn("Recall", format="%.1f%%", min_value=0.0, max_value=1.0),
                                    "F1": st.column_config.ProgressColumn("F1", format="%.1f%%", min_value=0.0, max_value=1.0),
                                }
                            )
                        else:
                            st.info("No per-class metrics available for this model.")
                    else:
                        st.info("Run comparison to generate per-class metrics.")

                with frames_tab:
                    st.markdown("**Comparison Frames Preview**")
                    frames = get_model_output_frames(view_model)

                    if frames:
                        cols = st.columns(3)
                        for idx, frame_path in enumerate(frames):
                            with cols[idx % 3]:
                                try:
                                    img = cv2.imread(str(frame_path))
                                    if img is not None:
                                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        st.image(img_rgb, caption=frame_path.name, use_container_width=True)
                                except:
                                    st.write(f"Could not load {frame_path.name}")
                        st.caption("Showing up to 6 recent output frames.")
                    else:
                        st.info("No comparison frames yet. Run evaluation to generate.")
    
    st.divider()
    
    # Display metrics history
    if metrics_df is not None and len(metrics_df) > 0:
        st.markdown("### Metrics History")
        model_col, date_col, prec_col, rec_col = resolve_metric_columns(metrics_df)
        metrics_df_work = metrics_df.copy()
        metrics_df_work[date_col] = pd.to_datetime(metrics_df_work[date_col], errors='coerce')
        metrics_df_work = metrics_df_work.sort_values(date_col, ascending=False)

        filter_models = sorted(metrics_df_work[model_col].dropna().unique().tolist())
        selected_models = st.multiselect(
            "Filter models",
            options=filter_models,
            default=filter_models,
            key="history_model_filter"
        )

        if selected_models:
            metrics_df_work = metrics_df_work[metrics_df_work[model_col].isin(selected_models)]
        else:
            st.info("Select at least one model to view history.")
            metrics_df_work = metrics_df_work.iloc[0:0]
        
        if len(metrics_df_work) > 0:
            trend_df = metrics_df_work[[date_col, model_col, prec_col, rec_col]].copy()
            trend_df = trend_df.dropna(subset=[date_col]).sort_values(date_col)
            if len(trend_df) > 0:
                st.markdown("#### Precision / Recall Trend")
                trend_metric = st.segmented_control(
                    "Trend metric",
                    ["Precision", "Recall"],
                    key="history_trend_metric",
                    default="Precision"
                )
                value_col = prec_col if trend_metric == "Precision" else rec_col
                pivot_data = trend_df.pivot_table(values=value_col, index=date_col, columns=model_col)
                st.line_chart(pivot_data)

            st.markdown("#### Detailed Runs")
            st.dataframe(
                metrics_df_work,
                use_container_width=True,
                hide_index=True,
                height=360,
                column_config={
                    model_col: st.column_config.TextColumn("Model", width="small"),
                    date_col: st.column_config.DatetimeColumn("Run Time", format="YYYY-MM-DD HH:mm:ss"),
                    prec_col: st.column_config.ProgressColumn("Overall Precision", format="%.1f%%", min_value=0.0, max_value=1.0),
                    rec_col: st.column_config.ProgressColumn("Overall Recall", format="%.1f%%", min_value=0.0, max_value=1.0),
                }
            )

        csv_data = metrics_df.to_csv(index=False)
        st.download_button("Download Metrics CSV", csv_data, "metrics.csv", "text/csv")

        st.divider()
        st.markdown("### Advanced Analytics")

        analytics_tab1, analytics_tab2, analytics_tab3, analytics_tab4 = st.tabs([
            "Model Performance",
            "Per-Class Metrics",
            "Metrics History",
            "Model Comparison",
        ])

        with analytics_tab1:
            latest_metrics = metrics_df.copy()
            latest_metrics[date_col] = pd.to_datetime(latest_metrics[date_col], errors='coerce')
            latest_metrics = latest_metrics.sort_values(date_col).groupby(model_col, as_index=False).tail(1)

            if len(latest_metrics) > 0:
                latest_metrics = latest_metrics.copy()
                latest_metrics['F1-Score'] = latest_metrics.apply(
                    lambda x: x.get('f1_score', 2 * (x.get(prec_col, 0) * x.get(rec_col, 0)) / (x.get(prec_col, 0) + x.get(rec_col, 0)) if (x.get(prec_col, 0) + x.get(rec_col, 0)) > 0 else 0),
                    axis=1
                )

                kpi_df = latest_metrics[[model_col, prec_col, rec_col, 'F1-Score']].copy()
                kpi_df.rename(columns={model_col: 'Model', prec_col: 'Precision', rec_col: 'Recall'}, inplace=True)
                kpi_df = kpi_df.sort_values('F1-Score', ascending=False)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Best F1 Model", str(kpi_df.iloc[0]['Model']))
                with c2:
                    st.metric("Top F1", f"{float(kpi_df.iloc[0]['F1-Score']):.2%}")
                with c3:
                    st.metric("Models Tracked", f"{kpi_df['Model'].nunique()}")

                st.bar_chart(kpi_df.set_index('Model')[['Precision', 'Recall', 'F1-Score']])
                st.dataframe(
                    kpi_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Model": st.column_config.TextColumn("Model", width="small"),
                        "Precision": st.column_config.ProgressColumn("Precision", format="%.1f%%", min_value=0.0, max_value=1.0),
                        "Recall": st.column_config.ProgressColumn("Recall", format="%.1f%%", min_value=0.0, max_value=1.0),
                        "F1-Score": st.column_config.ProgressColumn("F1", format="%.1f%%", min_value=0.0, max_value=1.0),
                    }
                )
            else:
                st.info("No metrics available yet. Run model comparison to generate metrics.")

        with analytics_tab2:
            per_class_cols = [col for col in metrics_df.columns if col.startswith('precision_') or col.startswith('recall_')]

            if per_class_cols:
                work_df = metrics_df.copy()
                work_df[date_col] = pd.to_datetime(work_df[date_col], errors='coerce')

                available_models_pc = sorted(work_df[model_col].dropna().unique().tolist())
                selected_model_pc = st.selectbox("Select Model", available_models_pc, key="model_compare_per_class_model_select")

                if selected_model_pc:
                    model_latest = work_df[work_df[model_col] == selected_model_pc].sort_values(date_col).tail(1)
                    if len(model_latest) > 0:
                        row = model_latest.iloc[0]
                        class_metrics = []

                        classes = set()
                        for col in per_class_cols:
                            if col.startswith('precision_'):
                                classes.add(col.replace('precision_', ''))
                            elif col.startswith('recall_'):
                                classes.add(col.replace('recall_', ''))

                        for cls in sorted(classes):
                            p_col = f'precision_{cls}'
                            r_col = f'recall_{cls}'
                            if p_col not in row.index or r_col not in row.index:
                                continue
                            p_val = float(row.get(p_col, 0)) if pd.notna(row.get(p_col, np.nan)) else 0.0
                            r_val = float(row.get(r_col, 0)) if pd.notna(row.get(r_col, np.nan)) else 0.0
                            f_val = (2 * p_val * r_val / (p_val + r_val)) if (p_val + r_val) > 0 else 0.0
                            class_metrics.append({
                                'Class': cls.replace('_', ' ').title(),
                                'Precision': p_val,
                                'Recall': r_val,
                                'F1': f_val
                            })

                        if class_metrics:
                            class_df = pd.DataFrame(class_metrics).sort_values('F1', ascending=False)
                            st.bar_chart(class_df.set_index('Class')[['Precision', 'Recall', 'F1']])
                            st.dataframe(
                                class_df,
                                use_container_width=True,
                                hide_index=True,
                                height=min(500, 72 + (len(class_df) * 36)),
                                column_config={
                                    "Class": st.column_config.TextColumn("Class", width="medium"),
                                    "Precision": st.column_config.ProgressColumn("Precision", format="%.1f%%", min_value=0.0, max_value=1.0),
                                    "Recall": st.column_config.ProgressColumn("Recall", format="%.1f%%", min_value=0.0, max_value=1.0),
                                    "F1": st.column_config.ProgressColumn("F1", format="%.1f%%", min_value=0.0, max_value=1.0),
                                }
                            )
                        else:
                            st.info("No per-class metrics available for this model.")
            else:
                st.info("No per-class metrics found in the data. Run model comparison to generate per-class metrics.")

        with analytics_tab3:
            history_df = metrics_df.copy()
            history_df[date_col] = pd.to_datetime(history_df[date_col], errors='coerce')
            history_df = history_df.dropna(subset=[date_col]).sort_values(date_col)

            model_opts = sorted(history_df[model_col].dropna().unique().tolist())
            selected_history_models = st.multiselect(
                "Models",
                options=model_opts,
                default=model_opts,
                key="model_compare_history_models"
            )

            if selected_history_models:
                history_df = history_df[history_df[model_col].isin(selected_history_models)]

            metric_choice = st.segmented_control(
                "Metric",
                ["Precision", "Recall"],
                key="model_compare_history_metric",
                default="Precision"
            )

            metric_col = prec_col if metric_choice == "Precision" else rec_col
            chart_df = history_df.pivot_table(values=metric_col, index=date_col, columns=model_col)
            if len(chart_df) > 0:
                st.line_chart(chart_df)
                st.area_chart(chart_df)

            st.dataframe(
                history_df.sort_values(date_col, ascending=False),
                use_container_width=True,
                hide_index=True,
                height=360,
                column_config={
                    model_col: st.column_config.TextColumn("Model", width="small"),
                    date_col: st.column_config.DatetimeColumn("Run Time", format="YYYY-MM-DD HH:mm:ss"),
                    prec_col: st.column_config.ProgressColumn("Overall Precision", format="%.1f%%", min_value=0.0, max_value=1.0),
                    rec_col: st.column_config.ProgressColumn("Overall Recall", format="%.1f%%", min_value=0.0, max_value=1.0),
                }
            )

        with analytics_tab4:
            latest_metrics = metrics_df.copy()
            latest_metrics[date_col] = pd.to_datetime(latest_metrics[date_col], errors='coerce')
            latest_metrics = latest_metrics.sort_values(date_col).groupby(model_col, as_index=False).tail(1)

            if len(latest_metrics) > 0:
                latest_metrics = latest_metrics.copy()
                latest_metrics['F1-Score'] = latest_metrics.apply(
                    lambda x: x.get('f1_score', 2 * (x.get(prec_col, 0) * x.get(rec_col, 0)) / (x.get(prec_col, 0) + x.get(rec_col, 0)) if (x.get(prec_col, 0) + x.get(rec_col, 0)) > 0 else 0),
                    axis=1
                )

                overview = latest_metrics[[model_col, prec_col, rec_col, 'F1-Score']].set_index(model_col)
                overview.columns = ['Precision', 'Recall', 'F1-Score']
                st.bar_chart(overview)

                per_class_cols = [col for col in latest_metrics.columns if col.startswith('precision_') or col.startswith('recall_')]
                if per_class_cols:
                    classes = set()
                    for col in per_class_cols:
                        if col.startswith('precision_'):
                            classes.add(col.replace('precision_', ''))
                        elif col.startswith('recall_'):
                            classes.add(col.replace('recall_', ''))

                    classes = sorted(classes)
                    if classes:
                        selected_class = st.selectbox("Select Class", classes, key="model_compare_class_comparison_select")
                        metric_type = st.segmented_control(
                            "Compare",
                            ["Precision", "Recall"],
                            key="model_compare_class_comp_metric",
                            default="Precision"
                        )

                        selected_col = f"precision_{selected_class}" if metric_type == "Precision" else f"recall_{selected_class}"
                        if selected_col in latest_metrics.columns:
                            class_comp_df = latest_metrics[[model_col, selected_col]].copy()
                            class_comp_df.rename(columns={selected_col: metric_type, model_col: 'Model'}, inplace=True)
                            st.bar_chart(class_comp_df.set_index('Model'))
                            st.dataframe(
                                class_comp_df,
                                use_container_width=True,
                                hide_index=True,
                                column_config={
                                    "Model": st.column_config.TextColumn("Model", width="small"),
                                    metric_type: st.column_config.ProgressColumn(metric_type, format="%.1f%%", min_value=0.0, max_value=1.0)
                                }
                            )

                display_cols = [model_col, prec_col, rec_col, 'F1-Score']
                for col in ['matched_boxes', 'false_positives', 'false_negatives', 'total_frames']:
                    if col in latest_metrics.columns:
                        display_cols.append(col)

                comparison_table = latest_metrics[display_cols].copy()
                comparison_table.rename(columns={
                    model_col: 'Model',
                    prec_col: 'Overall Precision',
                    rec_col: 'Overall Recall'
                }, inplace=True)

                st.dataframe(
                    comparison_table.sort_values('F1-Score', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Model": st.column_config.TextColumn("Model", width="small"),
                        "Overall Precision": st.column_config.ProgressColumn("Overall Precision", format="%.1f%%", min_value=0.0, max_value=1.0),
                        "Overall Recall": st.column_config.ProgressColumn("Overall Recall", format="%.1f%%", min_value=0.0, max_value=1.0),
                        "F1-Score": st.column_config.ProgressColumn("F1", format="%.1f%%", min_value=0.0, max_value=1.0),
                    }
                )
            else:
                st.info("No models compared yet.")

# INSIGHTS PAGE
elif current_page == "Insights":
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:inline;vertical-align:middle;margin-right:8px;">
                <path d="M3 3v18h18"></path>
                <path d="M7 14l3-3 3 2 4-5"></path>
                <circle cx="10" cy="11" r="1"></circle>
                <circle cx="13" cy="13" r="1"></circle>
                <circle cx="17" cy="8" r="1"></circle>
            </svg>
            Insights
        </h1>
        <p class="hero-subtitle">Quick summary of your annotation and model comparison workspace</p>
        <div class="hero-badges">
            <span class="hero-badge">Run Summary</span>
            <span class="hero-badge">Data Health</span>
            <span class="hero-badge">Next Actions</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    compare_base_dir = Path(__file__).resolve().parent / "Model_Compare"
    metrics_csv = compare_base_dir / "metrics.csv"
    frames_dir = Path(st.session_state.get("frames_dir", str(FRAMES_DIR)))
    annot_dir = Path(st.session_state.get("annot_dir", str(ANNOT_DIR)))

    def count_images(folder_path):
        """Count all image files recursively in a folder for Insights summary."""
        if not folder_path.exists():
            return 0
        return len([
            path for path in folder_path.rglob("*")
            if path.is_file() and path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

    frame_count = count_images(frames_dir)
    annot_image_count = count_images(annot_dir)
    label_count = len(list(annot_dir.rglob("*.txt"))) if annot_dir.exists() else 0

    metrics_df = pd.read_csv(metrics_csv) if metrics_csv.exists() else pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Frames", frame_count)
    with col2:
        st.metric("Annotated Images", annot_image_count)
    with col3:
        st.metric("Label Files", label_count)
    with col4:
        st.metric("Model Runs", len(metrics_df) if len(metrics_df) > 0 else 0)

    st.divider()

    if len(metrics_df) > 0:
        model_col = 'model_name' if 'model_name' in metrics_df.columns else 'model'
        date_col = 'date' if 'date' in metrics_df.columns else 'timestamp'
        prec_col = 'overall_precision' if 'overall_precision' in metrics_df.columns else 'precision'
        rec_col = 'overall_recall' if 'overall_recall' in metrics_df.columns else 'recall'

        work_df = metrics_df.copy()
        work_df[date_col] = pd.to_datetime(work_df[date_col], errors='coerce')
        work_df = work_df.sort_values(date_col)
        latest = work_df.tail(1).iloc[0]

        st.markdown("### Latest Run")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Model", str(latest.get(model_col, "-")))
        with c2:
            st.metric("Precision", f"{float(latest.get(prec_col, 0.0)):.2%}")
        with c3:
            st.metric("Recall", f"{float(latest.get(rec_col, 0.0)):.2%}")

        st.caption("Detailed analytics, comparisons, and trends are available in the Model Comparison page.")
    else:
        st.info("No model metrics found yet. Run at least one evaluation in Model Comparison.")




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
