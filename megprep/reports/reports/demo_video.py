# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import streamlit as st

# --- Professional CSS Styling ---
st.markdown("""
    <style>
    /* Global Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1E293B;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
    }

    /* Header Styling */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 1rem;
    }

    .header-icon {
        color: #3B82F6; /* Blue-500 */
    }



    .desc-text {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #475569;
    }

    .section-label {
        text-transform: uppercase;
        font-size: 0.75rem;
        font-weight: 700;
        color: #94A3B8;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Badge/Tag Styling */
    .highlight-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-right: 8px;
        margin-bottom: 8px;
        background-color: #EFF6FF; /* Blue-50 */
        color: #2563EB; /* Blue-600 */
        border: 1px solid #BFDBFE;
    }
    
    .highlight-badge:hover {
        background-color: #DBEAFE;
    }
    
    </style>
    """, unsafe_allow_html=True)
current_dir = os.path.dirname(os.path.abspath(__file__))

# --- 3. Content Data (Enriched with Icons) ---
CONTENT_MAP = {
    "Preprocessing": {
        "icon": "",
        "title": "Preprocessing",
        "video": os.path.join(current_dir, 'assets', '1.preproc.mp4'),
        "desc": "Interactive raw data inspection with real-time filtering capabilities. Dynamically adjust channel density and time windows to view specific segments, while editing artifact annotations.",
        "highlights": ["Real-time Filtering", "Waveform Inspection", "Artifact Annotation"]
    },
    "Artifacts - QuickCheck": {
        "icon": "",
        "title": "Artifact Detection & QuickCheck",
        "video": os.path.join(current_dir, 'assets', '2.artifact.mp4'),
        "desc": "A streamlined interface designed for rapid identification and marking of bad channels and segments, significantly accelerating the artifact annotation and recording process.",
        "highlights": ["Rapid Marking", "Edit bad channels and segments"]
    },
    "ICA": {
        "icon": "",
        "title": "Independent Component Analysis",
        "video": os.path.join(current_dir, 'assets', '3.ica.mp4'),
        "desc": "Interactive management of Independent Components (ICs). Review automatically flagged artifact components and manually select or deselect specific ICs for exclusion from the data.",
        "highlights": ["Component Review", "Artefact IC Auto-labeling"]
    },
    "Source Localization": {
        "icon": "🧠",
        "title": "Source Estimation",
        "video": os.path.join(current_dir, 'assets', '4.source.mp4'),
        "desc": "Interactive exploration of source estimation results. Visualize brain activity across specific timepoints and anatomical regions, with granular control over plotting parameters and thresholds.",
        "highlights": ["Timing Navigation", "Hemisphere layout selection", "Plot Controls"]
    },
}

# --- 4. Sidebar Logic ---
with st.sidebar:
    st.markdown("### 🧭 User Guide")
    st.info("Select a module below to watch the interactive walkthrough.")

    # Custom CSS styled radio button
    selected_module = st.radio(
        "Modules",
        options=CONTENT_MAP.keys(),
        label_visibility="collapsed"
    )

    st.markdown("---")

# --- 5. Main Content Renderer ---

data = CONTENT_MAP[selected_module]

# 5.1 Header Section
st.markdown(f"""
    <div class="main-header">
        <span class="header-icon">{data['icon']}</span>
        {data['title']}
    </div>
""", unsafe_allow_html=True)

# 5.2 Video Section (Cinema Mode)
try:
    st.video(data["video"])
except Exception as e:
    st.error(f"Video source unavailable: {data['video']}")
st.markdown('</div>', unsafe_allow_html=True)

# Info Section
st.markdown('<span class="section-label">Description</span>', unsafe_allow_html=True)
st.markdown(f'<div class="desc-text">{data["desc"]}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<span class="section-label">Key Features</span>', unsafe_allow_html=True)

# Render badges
badges_html = ""
for highlight in data["highlights"]:
    badges_html += f'<span class="highlight-badge">{highlight}</span>'

st.markdown(f'<div>{badges_html}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- 6. Subtle Footer ---
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style="text-align: center; color: #94A3B8; font-size: 0.8rem;">
        MegPrep Analytics Platform • Interactive Documentation
    </div>
""", unsafe_allow_html=True)
