from __future__ import annotations

import base64
import html
import json
import os
import re
import time
from datetime import datetime
from io import BytesIO
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageDraw

# Superb AI SDK는 현재 공식 문서 기준 Python 3.12+가 필요합니다.
# 패키지가 없거나 API Key가 설정되지 않은 경우에도 앱 자체가 즉시 종료되지 않고
# 화면에서 연결 준비 상태를 안내하도록 optional import로 처리합니다.
try:
    from superb_ai import Client as SuperbAIClient
except ImportError:
    SuperbAIClient = None

# ---------------------------------------------------------
# 기본 설정 및 모던 커스텀 CSS
# ---------------------------------------------------------

API_BASE_URL = "http://127.0.0.1:8000"

# V1은 Superb AI에 배포된 모델을 직접 호출합니다.
# API Key는 코드에 저장하지 않고 환경변수 SUPERB_AI_API_KEY에서 읽습니다.
SUPERB_TENANT = os.getenv("SUPERB_AI_TENANT", "koala")
V1_DEPLOYMENT_ID = os.getenv(
    "SUPERB_V1_DEPLOYMENT_ID",
    "91344fb9-12bd-403e-addf-03af4b12a7b0",
)

# Superb AI 서버에서는 낮은 기준으로 후보를 받아오고,
# 실제 RELEASE/HOLD/REVIEW 판정 기준은 사이드바 confidence_threshold를 로컬에서 적용합니다.
# 이렇게 해야 0.70 미만 예측도 화면에 클래스/신뢰도/BBox로 표시해 원인을 진단할 수 있습니다.
V1_RETRIEVAL_CONFIDENCE = float(os.getenv("SUPERB_V1_RETRIEVAL_CONFIDENCE", "0.05"))

STATUS_LABELS = {
    "RELEASE": "생산 진행 권고",
    "HOLD": "생산 보류 권고",
    "REVIEW": "전문가 검토 필요",
}

# 주요 결함 한글 표기
CLASS_LABELS = {
    "good": "정상",
    "no_good": "부품 위치 이상",
    "exc_solder": "과납",
    "poor_solder": "미납·부족 납땜",
    "spike": "돌기",
}

VIEW_LABELS = {
    "V1": "V1 실장 상태",
    "V2": "V2 좌측 납땜",
    "V2.1": "V2.1 우측 납땜",
}

VIEW_DESCRIPTIONS = {
    "V1": "상단 이미지 · 부품 위치/실장 상태 확인",
    "V2": "좌측 45° 이미지 · 좌측 납땜 상태 확인",
    "V2.1": "우측 45° 이미지 · 우측 납땜 상태 확인",
}

SOLDER_DEFECT_CLASSES = {"exc_solder", "poor_solder", "spike"}

st.set_page_config(
    page_title="LineGate AOI AI 관제탑",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

        /*
         * Pretendard는 일반 UI 텍스트에만 적용합니다.
         * Streamlit 내부 아이콘(Material Symbols)까지 * 선택자로 덮어쓰면
         * `keyboard_double_arrow_left`, `_arrow_right`, `upload` 같은 아이콘 이름이
         * 글자로 노출되므로 전역 * font-family 지정은 사용하지 않습니다.
         */
        html, body, .stApp, button, input, textarea, select, label {
            font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif;
        }

        .stApp {
            background-color: #0f172a;
            color: #f8fafc;
        }

        h1, h2, h3, h4, h5, h6, label, p, .stMarkdown {
            color: #f8fafc !important;
        }

        [data-testid="stSidebar"] {
            background-color: #1e293b;
            border-right: 1px solid #334155;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: #f8fafc !important;
        }

        /* 사이드바 현재 입력 요약 - 코드블록 대신 정보 카드 */
        .sidebar-summary-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 6px 14px;
            margin: 8px 0 10px 0;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.025);
        }
        .sidebar-summary-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 10px 0;
            border-bottom: 1px solid rgba(71, 85, 105, 0.5);
        }
        .sidebar-summary-row:last-child { border-bottom: none; }
        .sidebar-summary-label {
            color: #94a3b8 !important;
            font-size: 0.76rem;
            font-weight: 650;
            white-space: nowrap;
        }
        .sidebar-summary-value {
            color: #f8fafc !important;
            font-size: 0.84rem;
            font-weight: 750;
            text-align: right;
            line-height: 1.35;
        }
        .sidebar-view-pills {
            display: flex;
            justify-content: flex-end;
            gap: 5px;
            flex-wrap: wrap;
        }
        .sidebar-view-pill {
            color: #cbd5e1 !important;
            background: rgba(51, 65, 85, 0.62);
            border: 1px solid #475569;
            border-radius: 999px;
            padding: 2px 7px;
            font-size: 0.68rem;
            font-weight: 750;
        }
        .sidebar-engine-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
            margin: 10px 0 4px 0;
        }
        .sidebar-engine-card {
            background: rgba(30, 41, 59, 0.78);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 10px 10px 9px 10px;
        }
        .sidebar-engine-view {
            color: #94a3b8 !important;
            font-size: 0.66rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            margin-bottom: 3px;
        }
        .sidebar-engine-name {
            color: #f8fafc !important;
            font-size: 0.78rem;
            font-weight: 800;
            line-height: 1.25;
        }
        .sidebar-engine-type {
            color: #64748b !important;
            font-size: 0.65rem;
            font-weight: 650;
            margin-top: 3px;
            line-height: 1.3;
        }
        .sidebar-connection-card {
            background: rgba(15, 23, 42, 0.56);
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 11px 12px;
            margin-top: 4px;
        }
        .sidebar-connection-label {
            color: #64748b !important;
            font-size: 0.68rem;
            font-weight: 750;
            margin-bottom: 3px;
        }
        .sidebar-connection-value {
            color: #cbd5e1 !important;
            font-size: 0.72rem;
            font-weight: 650;
            line-height: 1.4;
            word-break: break-all;
            margin-bottom: 9px;
        }
        .sidebar-connection-value:last-child { margin-bottom: 0; }

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            color: #0f172a !important;
            -webkit-text-fill-color: #0f172a !important;
            font-weight: 600;
        }
        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {
            color: #64748b !important;
            -webkit-text-fill-color: #64748b !important;
        }

        code {
            background-color: #1e293b !important;
            color: #38bdf8 !important;
            border: 1px solid #334155 !important;
            padding: 2px 6px !important;
            border-radius: 4px !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
        }

        .dashboard-header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 24px 28px;
            margin-bottom: 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 24px;
        }
        .dashboard-title {
            font-size: 1.8rem;
            font-weight: 800;
            color: #f8fafc !important;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 12px;
            flex-wrap: wrap;
        }
        .dashboard-title-badge {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            color: #ffffff !important;
            font-size: 0.82rem;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            letter-spacing: 0.05em;
        }
        .dashboard-subtitle {
            font-size: 0.92rem;
            color: #94a3b8 !important;
            margin-top: 6px;
        }
        .status-stack {
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 7px;
            min-width: 250px;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 6px 16px;
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 700;
        }
        .status-online {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399 !important;
            border: 1px solid rgba(52, 211, 153, 0.3);
        }
        .status-offline {
            background: rgba(239, 68, 68, 0.15);
            color: #f87171 !important;
            border: 1px solid rgba(248, 113, 113, 0.3);
        }
        .dot-online {
            width: 8px; height: 8px; border-radius: 50%;
            background-color: #34d399; box-shadow: 0 0 10px #34d399;
        }
        .dot-offline {
            width: 8px; height: 8px; border-radius: 50%; background-color: #f87171;
        }
        .engine-chip {
            background: rgba(15, 23, 42, 0.78);
            border: 1px solid #334155;
            padding: 5px 11px;
            border-radius: 7px;
            font-size: 0.78rem;
            color: #cbd5e1 !important;
            font-weight: 600;
            white-space: nowrap;
        }

        .decision-card {
            padding: 20px 24px;
            border-radius: 14px;
            margin-bottom: 16px;
            border: 1px solid #334155;
            border-left: 8px solid #64748b;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
        }
        .decision-release {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.62), rgba(15, 23, 42, 0.92));
            border-left-color: #10b981;
            border-color: rgba(16, 185, 129, 0.32);
        }
        .decision-hold {
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.62), rgba(15, 23, 42, 0.92));
            border-left-color: #ef4444;
            border-color: rgba(239, 68, 68, 0.32);
        }
        .decision-review {
            background: linear-gradient(135deg, rgba(120, 53, 15, 0.62), rgba(15, 23, 42, 0.92));
            border-left-color: #f59e0b;
            border-color: rgba(245, 158, 11, 0.32);
        }
        .decision-card h3 {
            margin: 0 0 8px 0;
            font-size: 1.35rem;
            font-weight: 800;
            color: #f8fafc !important;
        }
        .decision-card p {
            margin: 4px 0;
            font-size: 0.92rem;
            color: #cbd5e1 !important;
        }

        .metric-card {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 18px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, border-color 0.2s ease;
            min-height: 94px;
        }
        .metric-card:hover { border-color: #475569; transform: translateY(-2px); }
        .metric-label {
            font-size: 0.78rem;
            color: #94a3b8 !important;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.045em;
            margin-bottom: 6px;
        }
        .metric-value {
            font-size: 1.45rem;
            font-weight: 800;
            color: #f1f5f9 !important;
            line-height: 1.25;
        }

        .detection-card {
            background: #0f172a;
            border: 1px solid #334155;
            padding: 12px 14px;
            border-radius: 8px;
            margin-bottom: 8px;
        }
        .detection-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }
        .detection-title { font-weight: 700; color: #f8fafc !important; }
        .detection-meta { font-size: 0.82rem; color: #cbd5e1 !important; margin-top: 8px; }
        .confidence-badge-good, .confidence-badge-bad {
            padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; font-weight: 700;
        }
        .confidence-badge-good {
            background: rgba(16, 185, 129, 0.14); color: #34d399 !important;
            border: 1px solid rgba(52, 211, 153, 0.32);
        }
        .confidence-badge-bad {
            background: rgba(239, 68, 68, 0.14); color: #f87171 !important;
            border: 1px solid rgba(248, 113, 113, 0.32);
        }

        .flow-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 8px;
        }
        .flow-node {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 14px 12px;
            text-align: center;
            min-height: 126px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.18);
        }
        .flow-node-title { color: #94a3b8 !important; font-size: 0.82rem; font-weight: 700; margin-bottom: 6px; }
        .flow-node-result { color: #f8fafc !important; font-size: 1.05rem; font-weight: 800; margin-bottom: 4px; }
        .flow-node-detail { color: #cbd5e1 !important; font-size: 0.82rem; line-height: 1.45; }
        .flow-arrow { text-align: center; color: #64748b !important; font-size: 1.8rem; line-height: 1; margin: 8px 0; }
        .flow-final {
            border-radius: 12px; padding: 16px 18px; text-align: center;
            border: 1px solid #334155; box-shadow: 0 6px 12px -4px rgba(0, 0, 0, 0.28);
        }
        .flow-final-release {
            background: linear-gradient(135deg, rgba(6, 78, 59, 0.62), rgba(15, 23, 42, 0.92));
            border-color: rgba(16, 185, 129, 0.36); color: #6ee7b7 !important;
        }
        .flow-final-hold {
            background: linear-gradient(135deg, rgba(127, 29, 29, 0.62), rgba(15, 23, 42, 0.92));
            border-color: rgba(239, 68, 68, 0.36); color: #fca5a5 !important;
        }
        .flow-final-review {
            background: linear-gradient(135deg, rgba(120, 53, 15, 0.62), rgba(15, 23, 42, 0.92));
            border-color: rgba(245, 158, 11, 0.36); color: #fcd34d !important;
        }
        .flow-final-status { font-size: 1.35rem; font-weight: 900; margin-bottom: 4px; }
        .flow-final-reason { font-size: 0.9rem; font-weight: 600; }

        .custom-table-container {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            overflow-x: auto;
            margin-top: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        .custom-table { width: 100%; border-collapse: collapse; text-align: left; font-size: 0.88rem; min-width: 880px; }
        .custom-table th {
            background: #0f172a; color: #cbd5e1 !important; padding: 13px 16px;
            font-weight: 700; border-bottom: 1px solid #334155;
        }
        .custom-table td {
            padding: 13px 16px; border-bottom: 1px solid #334155;
            color: #e2e8f0 !important; vertical-align: middle;
        }
        .custom-table tr:last-child td { border-bottom: none; }
        .status-chip { padding: 4px 10px; border-radius: 6px; font-size: 0.78rem; font-weight: 800; display: inline-block; }

        [data-testid="stMetric"] {
            background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 12px 14px;
        }
        [data-testid="stMetricLabel"] p { color: #94a3b8 !important; }
        [data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; color: #f8fafc !important; }

        [data-testid="stVerticalBlockBorderWrapper"] {
            border-color: #334155 !important;
            background: rgba(30, 41, 59, 0.52);
            border-radius: 12px;
        }
        [data-testid="stExpander"] {
            border-color: #334155 !important;
            background: rgba(30, 41, 59, 0.52);
        }

        .stButton > button, .stDownloadButton > button,
        [data-testid="stFormSubmitButton"] > button {
            border-radius: 8px; font-weight: 700; min-height: 3.1em;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.22);
        }
        .stButton > button[kind="primary"],
        [data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
            border: 1px solid #3b82f6 !important; color: #ffffff !important;
        }

        [data-testid="stImageCaption"] { color: #94a3b8 !important; }
        hr { border-color: #334155 !important; }


        /* ---------------------------------------------------------
           작업자 중심 산업용 UI · Carbon / Linear 계열 정돈
           --------------------------------------------------------- */
        :root {
            --lg-bg: #161616;
            --lg-surface: #202020;
            --lg-surface-2: #262626;
            --lg-border: #393939;
            --lg-text: #f4f4f4;
            --lg-muted: #a8a8a8;
            --lg-blue: #0f62fe;
            --lg-green: #42be65;
            --lg-amber: #f1c21b;
            --lg-red: #fa4d56;
        }
        .stApp { background: var(--lg-bg) !important; }
        [data-testid="stSidebar"] {
            background: #1f1f1f !important;
            border-right: 1px solid var(--lg-border) !important;
        }
        .dashboard-header {
            background: #1f1f1f !important;
            border: 1px solid var(--lg-border) !important;
            border-radius: 8px !important;
            box-shadow: none !important;
            padding: 20px 24px !important;
        }
        .dashboard-title { font-size: 1.55rem !important; letter-spacing: -0.015em !important; }
        .dashboard-title-badge {
            background: #262626 !important;
            border: 1px solid #525252 !important;
            color: #c6c6c6 !important;
            border-radius: 4px !important;
            padding: 3px 8px !important;
            font-size: 0.7rem !important;
        }
        .dashboard-subtitle { color: #a8a8a8 !important; }
        .status-pill {
            border-radius: 4px !important;
            padding: 6px 10px !important;
            letter-spacing: 0.02em;
        }
        .status-online { background: rgba(66,190,101,.10) !important; color: var(--lg-green) !important; border-color: rgba(66,190,101,.45) !important; }
        .status-offline { background: rgba(250,77,86,.10) !important; color: var(--lg-red) !important; border-color: rgba(250,77,86,.45) !important; }
        .dot-online { background: var(--lg-green) !important; box-shadow: none !important; }
        .dot-offline { background: var(--lg-red) !important; }
        .system-summary { color:#a8a8a8 !important; font-size:.78rem; font-weight:600; }
        .engine-chip { display:none !important; }

        .metric-card {
            background: #202020 !important;
            border: 1px solid var(--lg-border) !important;
            border-radius: 6px !important;
            box-shadow: none !important;
            transition: none !important;
            min-height: 88px !important;
        }
        .metric-card:hover { transform:none !important; border-color:#525252 !important; }
        .metric-label { color:#a8a8a8 !important; text-transform:none !important; letter-spacing:0 !important; }
        .metric-value { color:#f4f4f4 !important; }

        .decision-card {
            background: #202020 !important;
            border-radius: 6px !important;
            border-width: 1px !important;
            border-left-width: 5px !important;
            box-shadow: none !important;
            padding: 16px 18px !important;
        }
        .decision-release { border-color:#2e6f3e !important; border-left-color:var(--lg-green) !important; }
        .decision-review { border-color:#6f5d10 !important; border-left-color:var(--lg-amber) !important; }
        .decision-hold { border-color:#7d2d31 !important; border-left-color:var(--lg-red) !important; }

        .final-status-card {
            background:#202020;
            border:1px solid var(--lg-border);
            border-left:7px solid #8d8d8d;
            border-radius:6px;
            padding:24px 26px;
            margin:8px 0 18px 0;
        }
        .final-status-release { border-left-color:var(--lg-green); }
        .final-status-review { border-left-color:var(--lg-amber); }
        .final-status-hold { border-left-color:var(--lg-red); }
        .final-status-eyebrow { color:#a8a8a8 !important; font-size:.76rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
        .final-status-row { display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin:7px 0 16px 0; }
        .final-status-value { font-size:2.25rem; line-height:1; font-weight:800; letter-spacing:-.02em; }
        .final-status-release .final-status-value { color:var(--lg-green) !important; }
        .final-status-review .final-status-value { color:var(--lg-amber) !important; }
        .final-status-hold .final-status-value { color:var(--lg-red) !important; }
        .final-status-label { color:#c6c6c6 !important; font-size:.9rem; font-weight:600; }
        .final-reason-block { border-top:1px solid var(--lg-border); padding-top:14px; }
        .final-reason-label { color:#8d8d8d !important; font-size:.74rem; font-weight:700; margin-bottom:5px; }
        .final-reason-text { color:#f4f4f4 !important; font-size:1rem; line-height:1.55; font-weight:600; }

        .action-panel {
            background:#202020;
            border:1px solid var(--lg-border);
            border-left:5px solid var(--lg-blue);
            border-radius:6px;
            padding:16px 18px;
            margin:8px 0 4px 0;
        }
        .action-title { color:#a8a8a8 !important; font-size:.76rem; font-weight:700; margin-bottom:6px; }
        .action-text { color:#f4f4f4 !important; font-size:1rem; font-weight:650; line-height:1.5; }

        .upload-head {
            background:#202020;
            border:1px solid var(--lg-border);
            border-radius:6px;
            padding:13px 14px;
            margin-bottom:8px;
        }
        .upload-head-top { display:flex; justify-content:space-between; gap:10px; align-items:center; }
        .upload-view-title { color:#f4f4f4 !important; font-size:.96rem; font-weight:750; }
        .upload-view-desc { color:#8d8d8d !important; font-size:.74rem; line-height:1.4; margin-top:4px; }
        .upload-state { border-radius:999px; padding:3px 8px; font-size:.68rem; font-weight:750; white-space:nowrap; }
        .upload-required { color:#f1c21b !important; background:rgba(241,194,27,.08); border:1px solid rgba(241,194,27,.40); }
        .upload-complete { color:#42be65 !important; background:rgba(66,190,101,.08); border:1px solid rgba(66,190,101,.40); }
        .input-readiness {
            display:flex; justify-content:space-between; align-items:center; gap:12px;
            background:#202020; border:1px solid var(--lg-border); border-radius:6px;
            padding:11px 14px; margin:14px 0 6px 0;
        }
        .input-readiness-label { color:#c6c6c6 !important; font-size:.82rem; font-weight:650; }
        .input-readiness-value { font-size:.8rem; font-weight:750; }
        .ready-ok { color:var(--lg-green) !important; }
        .ready-wait { color:var(--lg-amber) !important; }

        [data-testid="stFileUploader"] { background:#1f1f1f !important; border-radius:6px !important; }
        [data-testid="stFileUploaderDropzone"] { background:#202020 !important; border-color:#525252 !important; border-radius:6px !important; }
        [data-testid="stFileUploaderDropzone"] button { border-radius:4px !important; }

        .detection-card { background:#1f1f1f !important; border-color:#393939 !important; border-radius:4px !important; }
        .detection-meta { color:#8d8d8d !important; }
        .confidence-badge-good { color:var(--lg-green) !important; border-color:rgba(66,190,101,.35) !important; background:rgba(66,190,101,.08) !important; }
        .confidence-badge-bad { color:var(--lg-red) !important; border-color:rgba(250,77,86,.35) !important; background:rgba(250,77,86,.08) !important; }

        .flow-node, .flow-final, .custom-table-container,
        [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] {
            background:#202020 !important;
            border-color:#393939 !important;
            border-radius:6px !important;
            box-shadow:none !important;
        }
        .flow-final-release { background:#202020 !important; border-left:5px solid var(--lg-green) !important; color:var(--lg-green) !important; }
        .flow-final-review { background:#202020 !important; border-left:5px solid var(--lg-amber) !important; color:var(--lg-amber) !important; }
        .flow-final-hold { background:#202020 !important; border-left:5px solid var(--lg-red) !important; color:var(--lg-red) !important; }
        .custom-table th { background:#161616 !important; color:#c6c6c6 !important; }
        .custom-table td { border-color:#393939 !important; }

        .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
            border-radius:4px !important; box-shadow:none !important; font-weight:700 !important;
        }
        .stButton > button[kind="primary"], [data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background:var(--lg-blue) !important; border-color:var(--lg-blue) !important;
        }

        /* ---------------------------------------------------------
           Streamlit 아이콘 / 업로더 / Expander 호환성 보정
           --------------------------------------------------------- */

        /* Streamlit Material Symbols 아이콘 폰트 복구
           - sidebar 접기/펼치기
           - file uploader의 upload 아이콘
           - expander 화살표
           등에 아이콘 이름이 문자열로 노출되는 현상을 방지합니다. */
        [data-testid="stIconMaterial"],
        .material-symbols-rounded,
        .material-symbols-outlined,
        span[class*="material-symbols"] {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
            font-weight: normal !important;
            font-style: normal !important;
            font-size: 1.25rem !important;
            line-height: 1 !important;
            letter-spacing: normal !important;
            text-transform: none !important;
            white-space: nowrap !important;
            word-wrap: normal !important;
            direction: ltr !important;
            -webkit-font-feature-settings: "liga" !important;
            -webkit-font-smoothing: antialiased !important;
            font-feature-settings: "liga" !important;
        }

        /* 일부 Streamlit 버전은 아이콘 span에 translate="no"를 사용합니다. */
        span[translate="no"] {
            font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
            font-variant-ligatures: normal !important;
            font-feature-settings: "liga" !important;
        }

        /* File uploader: 흰 버튼 + 흰 글자 충돌 방지 */
        [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stFileUploaderDropzone"] button[kind="secondary"] {
            background: #393939 !important;
            border: 1px solid #525252 !important;
            color: #f4f4f4 !important;
            min-height: 2.7rem !important;
            padding: 0.45rem 0.85rem !important;
            box-shadow: none !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover,
        [data-testid="stFileUploaderDropzone"] button[kind="secondary"]:hover {
            background: #4c4c4c !important;
            border-color: #6f6f6f !important;
        }
        [data-testid="stFileUploaderDropzone"] button p,
        [data-testid="stFileUploaderDropzone"] button span,
        [data-testid="stFileUploaderDropzone"] button div {
            color: #f4f4f4 !important;
        }
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] div {
            color: #a8a8a8 !important;
        }

        /* Expander 헤더: 아이콘/텍스트가 겹치지 않도록 레이아웃 고정 */
        [data-testid="stExpander"] details > summary {
            display: flex !important;
            align-items: center !important;
            gap: 0.45rem !important;
            min-height: 2.8rem !important;
            padding: 0.55rem 0.8rem !important;
            color: #f4f4f4 !important;
        }
        [data-testid="stExpander"] details > summary p,
        [data-testid="stExpander"] details > summary span:not([data-testid="stIconMaterial"]) {
            color: #f4f4f4 !important;
            line-height: 1.35 !important;
            white-space: normal !important;
        }

        /* Sidebar의 blanket span 색상 규칙 때문에 밝은 입력 버튼 텍스트가 사라지는 것을 보정 */
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button * {
            color: #f4f4f4 !important;
        }

        /* 버튼 안 텍스트는 아이콘과 분리하여 정상 폰트 사용 */
        .stButton > button p,
        .stDownloadButton > button p,
        [data-testid="stFormSubmitButton"] > button p,
        [data-testid="stFileUploaderDropzone"] button p {
            font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, system-ui, Roboto, sans-serif !important;
        }


        /* ---------------------------------------------------------
           Human Approval textarea 가독성 보정
           --------------------------------------------------------- */
        [data-testid="stTextArea"] textarea,
        .stTextArea textarea,
        div[data-baseweb="textarea"] textarea {
            color: #161616 !important;
            -webkit-text-fill-color: #161616 !important;
            caret-color: #0f62fe !important;
            background-color: #f4f4f4 !important;
            font-weight: 500 !important;
            opacity: 1 !important;
        }

        [data-testid="stTextArea"] textarea::placeholder,
        .stTextArea textarea::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {
            color: #6f6f6f !important;
            -webkit-text-fill-color: #6f6f6f !important;
            opacity: 1 !important;
        }

        [data-testid="stTextArea"] textarea:focus,
        .stTextArea textarea:focus,
        div[data-baseweb="textarea"] textarea:focus {
            color: #161616 !important;
            -webkit-text-fill-color: #161616 !important;
            background-color: #ffffff !important;
        }

        [data-testid="stTextArea"] textarea::selection,
        .stTextArea textarea::selection,
        div[data-baseweb="textarea"] textarea::selection {
            background: #0f62fe !important;
            color: #ffffff !important;
            -webkit-text-fill-color: #ffffff !important;
        }

        @media (max-width: 900px) {
            .flow-grid { grid-template-columns: 1fr; }
            .dashboard-header { align-items: flex-start; flex-direction: column; }
            .status-stack { align-items: flex-start; min-width: 0; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 세션 상태 안전 초기화
# ---------------------------------------------------------

if "inspection_results" not in st.session_state:
    st.session_state["inspection_results"] = {}

if "original_images" not in st.session_state:
    st.session_state["original_images"] = {}

if "inspection_case" not in st.session_state:
    st.session_state["inspection_case"] = {}

if "human_approval" not in st.session_state:
    st.session_state["human_approval"] = None


# ---------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------

def check_api_health() -> tuple[bool, dict[str, Any] | None]:
    """V2/V2.1용 로컬 FastAPI 서버 상태를 확인한다."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return True, response.json()
    except requests.RequestException:
        return False, None


def check_superb_v1_ready() -> tuple[bool, dict[str, Any]]:
    """V1 Superb AI 호출에 필요한 로컬 설정이 준비됐는지 확인한다.

    Streamlit rerun마다 원격 deployment.get()을 호출하지 않기 위해
    여기서는 SDK 설치 여부와 API Key 존재 여부만 검사한다.
    실제 배포 상태/권한/워밍업 오류는 predict 호출 시 사용자에게 표시한다.
    """
    if SuperbAIClient is None:
        return False, {
            "status": "SDK_MISSING",
            "message": "superb-ai 패키지가 설치되어 있지 않습니다.",
        }

    if not os.getenv("SUPERB_AI_API_KEY"):
        return False, {
            "status": "API_KEY_MISSING",
            "message": "SUPERB_AI_API_KEY 환경변수가 설정되어 있지 않습니다.",
        }

    return True, {
        "status": "CONFIGURED",
        "message": "V1 Superb AI 호출 설정 완료",
        "tenant": SUPERB_TENANT,
        "deployment_id": V1_DEPLOYMENT_ID,
    }


def get_superb_client():
    """환경변수 인증을 사용하는 Superb AI Client를 생성한다."""
    ready, info = check_superb_v1_ready()
    if not ready:
        raise RuntimeError(info.get("message", "Superb AI 설정이 완료되지 않았습니다."))

    # SUPERB_AI_API_KEY는 SDK가 환경변수에서 자동으로 읽습니다.
    return SuperbAIClient(tenant=SUPERB_TENANT)


def _to_plain_dict(value: Any) -> Any:
    """Pydantic 모델/객체를 JSON 친화적인 기본 타입으로 재귀 변환한다."""
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _to_plain_dict(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_plain_dict(item) for item in value]

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_plain_dict(model_dump(mode="json"))
        except TypeError:
            return _to_plain_dict(model_dump())

    as_dict = getattr(value, "dict", None)
    if callable(as_dict):
        return _to_plain_dict(as_dict())

    if hasattr(value, "__dict__"):
        return {
            str(key): _to_plain_dict(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }

    return value


def _point_from_value(value: Any) -> tuple[float, float] | None:
    """[x, y] 또는 {x, y} 형태를 좌표 튜플로 변환한다."""
    value = _to_plain_dict(value)

    if isinstance(value, dict):
        if "x" in value and "y" in value:
            try:
                return float(value["x"]), float(value["y"])
            except (TypeError, ValueError):
                return None

    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None

    return None


def _normalize_polygon_points(
    raw_points: Any,
    image_width: int,
    image_height: int,
) -> list[list[float]]:
    """다양한 좌표 표현을 [[x, y], ...] 형식으로 정규화한다."""
    raw_points = _to_plain_dict(raw_points)
    points: list[tuple[float, float]] = []

    if not isinstance(raw_points, (list, tuple)):
        return []

    # [x1, y1, x2, y2, ...] 형태도 지원
    if raw_points and all(isinstance(item, (int, float)) for item in raw_points):
        if len(raw_points) % 2 != 0:
            return []
        for index in range(0, len(raw_points), 2):
            points.append((float(raw_points[index]), float(raw_points[index + 1])))
    else:
        for item in raw_points:
            point = _point_from_value(item)
            if point is not None:
                points.append(point)

    if len(points) < 3:
        return []

    # 혹시 0~1 정규화 좌표가 반환되는 SDK/모델이라면 원본 픽셀로 복원한다.
    max_abs_x = max(abs(point[0]) for point in points)
    max_abs_y = max(abs(point[1]) for point in points)
    if max_abs_x <= 1.5 and max_abs_y <= 1.5 and image_width > 2 and image_height > 2:
        points = [
            (point[0] * image_width, point[1] * image_height)
            for point in points
        ]

    return [[float(x), float(y)] for x, y in points]


def _extract_polygon_parts(
    geometry: Any,
    image_width: int,
    image_height: int,
) -> list[list[list[float]]]:
    """Superb AI geometry에서 polygon part들을 최대한 보수적으로 추출한다."""
    geometry = _to_plain_dict(geometry)
    polygon_parts: list[list[list[float]]] = []

    def visit(node: Any, key_hint: str = "") -> None:
        node = _to_plain_dict(node)

        if isinstance(node, dict):
            # polygon/segmentation에서 자주 쓰이는 키를 우선 탐색
            for key in (
                "points",
                "vertices",
                "coordinates",
                "polygon",
                "polygons",
                "parts",
                "segmentation",
                "exterior",
            ):
                if key in node:
                    visit(node[key], key)
            return

        if not isinstance(node, (list, tuple)) or not node:
            return

        # 현재 리스트 자체가 좌표열인지 먼저 확인
        normalized = _normalize_polygon_points(node, image_width, image_height)
        if normalized:
            polygon_parts.append(normalized)
            return

        # 멀티파트/중첩 좌표열일 수 있으므로 하위 요소 탐색
        for item in node:
            visit(item, key_hint)

    visit(geometry)

    # 동일 좌표가 여러 키를 통해 중복 수집될 수 있으므로 간단히 dedupe
    unique_parts: list[list[list[float]]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for part in polygon_parts:
        signature = tuple((round(point[0], 3), round(point[1], 3)) for point in part)
        if signature not in seen:
            seen.add(signature)
            unique_parts.append(part)

    return unique_parts


def _extract_bbox_xyxy(
    geometry: Any,
    polygon_parts: list[list[list[float]]],
    image_width: int,
    image_height: int,
) -> list[float]:
    """polygon 또는 다양한 BBox geometry를 [x1, y1, x2, y2]로 변환한다.

    Superb AI 응답에서 geometry가 data/value/shape 등 하위 객체에 중첩되어도
    재귀적으로 탐색한다. 좌표가 0~1 정규화 값이면 원본 이미지 픽셀로 환산한다.
    """
    all_points = [point for part in polygon_parts for point in part]
    if all_points:
        xs = [point[0] for point in all_points]
        ys = [point[1] for point in all_points]
        return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

    def scale_box(x1: float, y1: float, x2: float, y2: float) -> list[float]:
        vals = [x1, y1, x2, y2]
        if max(abs(v) for v in vals) <= 1.5:
            x1 *= image_width
            x2 *= image_width
            y1 *= image_height
            y2 *= image_height
        return [float(x1), float(y1), float(x2), float(y2)]

    def from_mapping(node: dict[str, Any], depth: int = 0) -> list[float]:
        if depth > 8:
            return []

        # x1/y1/x2/y2, xmin/ymin/xmax/ymax
        for keys in (("x1", "y1", "x2", "y2"), ("xmin", "ymin", "xmax", "ymax")):
            if all(k in node for k in keys):
                try:
                    return scale_box(*(float(node[k]) for k in keys))
                except (TypeError, ValueError):
                    pass

        # left/top/right/bottom
        if all(k in node for k in ("left", "top", "right", "bottom")):
            try:
                return scale_box(
                    float(node["left"]), float(node["top"]),
                    float(node["right"]), float(node["bottom"]),
                )
            except (TypeError, ValueError):
                pass

        # x/y/width/height 또는 x/y/w/h (좌상단 기준)
        width_key = "width" if "width" in node else ("w" if "w" in node else None)
        height_key = "height" if "height" in node else ("h" if "h" in node else None)
        if "x" in node and "y" in node and width_key and height_key:
            try:
                x = float(node["x"]); y = float(node["y"])
                w = float(node[width_key]); h = float(node[height_key])
                if max(abs(x), abs(y), abs(w), abs(h)) <= 1.5:
                    x *= image_width; w *= image_width
                    y *= image_height; h *= image_height
                return [x, y, x + w, y + h]
            except (TypeError, ValueError):
                pass

        # center_x/center_y/width/height 또는 cx/cy/w/h (중심 기준)
        cx_key = "center_x" if "center_x" in node else ("cx" if "cx" in node else None)
        cy_key = "center_y" if "center_y" in node else ("cy" if "cy" in node else None)
        width_key = "width" if "width" in node else ("w" if "w" in node else None)
        height_key = "height" if "height" in node else ("h" if "h" in node else None)
        if cx_key and cy_key and width_key and height_key:
            try:
                cx = float(node[cx_key]); cy = float(node[cy_key])
                w = float(node[width_key]); h = float(node[height_key])
                if max(abs(cx), abs(cy), abs(w), abs(h)) <= 1.5:
                    cx *= image_width; w *= image_width
                    cy *= image_height; h *= image_height
                return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]
            except (TypeError, ValueError):
                pass

        # 명시적인 bbox/box/bounds/rect 키 우선
        for key in ("bbox", "box", "bounds", "rect", "rectangle"):
            if key not in node:
                continue
            nested = _to_plain_dict(node.get(key))
            if isinstance(nested, dict):
                result = from_mapping(nested, depth + 1)
                if result:
                    return result
            elif isinstance(nested, (list, tuple)) and len(nested) >= 4:
                try:
                    x1, y1, a, b = [float(v) for v in nested[:4]]
                    # bbox 배열은 보통 xyxy 또는 xywh. 값 관계로 우선 판별
                    if a > x1 and b > y1:
                        return scale_box(x1, y1, a, b)
                    if max(abs(x1), abs(y1), abs(a), abs(b)) <= 1.5:
                        x1 *= image_width; a *= image_width
                        y1 *= image_height; b *= image_height
                    return [x1, y1, x1 + a, y1 + b]
                except (TypeError, ValueError):
                    pass

        # Superb 응답이 geometry.data / geometry.value / geometry.shape처럼 중첩될 수 있어
        # 모든 하위 dict를 재귀 탐색한다.
        for value in node.values():
            child = _to_plain_dict(value)
            if isinstance(child, dict):
                result = from_mapping(child, depth + 1)
                if result:
                    return result
            elif isinstance(child, list):
                for item in child:
                    item_plain = _to_plain_dict(item)
                    if isinstance(item_plain, dict):
                        result = from_mapping(item_plain, depth + 1)
                        if result:
                            return result
        return []

    geometry_plain = _to_plain_dict(geometry)
    if isinstance(geometry_plain, dict):
        return from_mapping(geometry_plain)

    return []


def _polygon_area(points: list[list[float]]) -> float:
    """Shoelace formula로 polygon 면적을 계산한다."""
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return abs(area) / 2.0


def _build_v1_overlay_b64(
    image_bytes: bytes,
    detections: list[dict[str, Any]],
) -> str | None:
    """Superb AI detection BBox/polygon 결과를 원본 이미지 위에 직접 그려 PNG base64로 반환한다."""
    # 예측이 하나도 없을 때 원본 이미지를 Overlay처럼 다시 보여주지 않습니다.
    if not detections:
        return None

    try:
        base_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    except Exception:
        return None

    overlay_layer = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)

    for detection in detections:
        class_name = str(detection.get("class_name") or "unknown")
        confidence = float(detection.get("confidence") or 0.0)
        polygon_parts = detection.get("polygon_parts") or []
        bbox = detection.get("bbox_xyxy") or []

        if class_name == "no_good":
            outline_color = (239, 68, 68, 255)
            fill_color = (239, 68, 68, 70)
        elif class_name == "good":
            outline_color = (16, 185, 129, 255)
            fill_color = (16, 185, 129, 55)
        else:
            outline_color = (245, 158, 11, 255)
            fill_color = (245, 158, 11, 55)

        for part in polygon_parts:
            if len(part) < 3:
                continue
            xy = [(float(point[0]), float(point[1])) for point in part]
            draw.polygon(xy, fill=fill_color)
            draw.line(xy + [xy[0]], fill=outline_color, width=4, joint="curve")

        if not polygon_parts and len(bbox) >= 4:
            draw.rectangle(
                [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                outline=outline_color,
                width=4,
            )

        if len(bbox) >= 4:
            label = f"{class_name} {confidence * 100:.1f}%"
            text_x = max(0, int(float(bbox[0])))
            text_y = max(0, int(float(bbox[1])) - 18)
            # 텍스트 가독성을 위해 작은 배경 박스 표시
            text_box = draw.textbbox((text_x, text_y), label)
            draw.rectangle(text_box, fill=(15, 23, 42, 210))
            draw.text((text_x, text_y), label, fill=(255, 255, 255, 255))

    composed = Image.alpha_composite(base_image, overlay_layer).convert("RGB")
    buffer = BytesIO()
    composed.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _format_superb_error(exc: Exception) -> str:
    """SDK typed error의 code/hint/request_id가 있으면 함께 표시한다."""
    parts = [str(exc)]
    code = getattr(exc, "code", None)
    hint = getattr(exc, "hint", None)
    request_id = getattr(exc, "request_id", None)

    if code:
        parts.append(f"code={code}")
    if hint:
        parts.append(f"hint={hint}")
    if request_id:
        parts.append(f"request_id={request_id}")

    return " | ".join(parts)


def _normalize_v1_class_name(class_name: Any) -> str:
    """Superb AI 배포 클래스명을 LineGate 내부 클래스(good/no_good)로 정규화한다.

    Superb AI Detection 배포에서는 프로젝트/라벨 설정에 따라 클래스가
    good_BBox, good_b, no_good_BBox, no_good_b처럼 접미사가 붙어서
    반환될 수 있다. V1은 good/no_good 2개 클래스만 사용하는 모델이므로
    이 변형들을 LineGate 표준 클래스명으로 안전하게 매핑한다.
    """
    raw = str(class_name).strip() if class_name is not None else "unknown"
    key = raw.casefold().strip()

    # 공백/하이픈을 언더스코어로 통일하고 중복 구분자를 정리한다.
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"_+", "_", key).strip("_")

    # 명확한 별칭을 먼저 처리한다.
    aliases = {
        "good": "good",
        "normal": "good",
        "ok": "good",
        "no_good": "no_good",
        "nogood": "no_good",
        "bad": "no_good",
        "ng": "no_good",
    }
    if key in aliases:
        return aliases[key]

    # no_good 계열을 good보다 먼저 검사해야 no_good_b가 good으로 잘못 매핑되지 않는다.
    if re.fullmatch(r"(?:no_good|nogood)(?:_.*)?", key):
        return "no_good"

    # good_BBox, good_bbox, good_b, good_box 등 Detection 라벨 변형을 흡수한다.
    if re.fullmatch(r"good(?:_.*)?", key):
        return "good"

    # normal_bbox / ok_bbox 같은 별칭 변형도 허용한다.
    if re.fullmatch(r"(?:normal|ok)(?:_.*)?", key):
        return "good"
    if re.fullmatch(r"(?:bad|ng)(?:_.*)?", key):
        return "no_good"

    return key or "unknown"


def convert_superb_v1_result(
    superb_result: Any,
    image_bytes: bytes,
    filename: str,
    confidence_threshold: float,
    inference_time_ms: float,
) -> dict[str, Any]:
    """Superb AI PredictResponse를 기존 LineGate 결과 JSON 계약으로 변환한다."""
    with Image.open(BytesIO(image_bytes)) as image:
        image_width, image_height = image.size

    raw_result = _to_plain_dict(superb_result)
    raw_predictions = []

    if isinstance(raw_result, dict):
        raw_predictions = raw_result.get("predictions") or []
    elif hasattr(superb_result, "predictions"):
        raw_predictions = getattr(superb_result, "predictions") or []

    detections: list[dict[str, Any]] = []

    for prediction in raw_predictions:
        prediction_dict = _to_plain_dict(prediction)
        if not isinstance(prediction_dict, dict):
            continue

        raw_class_name = prediction_dict.get("class_name")
        if raw_class_name is None:
            raw_class_name = prediction_dict.get("label") or prediction_dict.get("class")
        class_name = _normalize_v1_class_name(raw_class_name)

        raw_confidence = prediction_dict.get("confidence")
        if raw_confidence is None:
            raw_confidence = prediction_dict.get("score", 0.0)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = 0.0

        geometry = prediction_dict.get("geometry")
        polygon_parts = _extract_polygon_parts(
            geometry,
            image_width=image_width,
            image_height=image_height,
        )
        bbox_xyxy = _extract_bbox_xyxy(
            geometry,
            polygon_parts=polygon_parts,
            image_width=image_width,
            image_height=image_height,
        )

        # 기존 UI는 단일 polygon 필드를 사용하므로 가장 큰 part를 대표 polygon으로 둔다.
        representative_polygon: list[list[float]] = []
        if polygon_parts:
            representative_polygon = max(polygon_parts, key=_polygon_area)

        mask_area = int(round(sum(_polygon_area(part) for part in polygon_parts)))
        if mask_area <= 0 and len(bbox_xyxy) >= 4:
            mask_area = int(
                max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0]))
                * max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))
            )

        detections.append(
            {
                "class_name": class_name,
                "raw_class_name": (str(raw_class_name).strip() if raw_class_name is not None else "unknown"),
                "confidence": confidence,
                "geometry": geometry,
                "polygon": representative_polygon,
                "polygon_parts": polygon_parts,
                "bbox_xyxy": bbox_xyxy,
                "mask_area_px": mask_area if mask_area > 0 else None,
            }
        )

    detections.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)

    top_class = detections[0]["class_name"] if detections else None
    top_confidence = detections[0]["confidence"] if detections else None

    # 운영 임계값을 통과한 예측만 최종 판정에 사용합니다.
    # 서버 호출 단계에서는 V1_RETRIEVAL_CONFIDENCE(기본 0.05)로 후보를 넓게 받아오므로,
    # 임계값 미달 예측도 화면에서 클래스/신뢰도/BBox를 확인할 수 있습니다.
    accepted_detections = [
        item for item in detections
        if float(item.get("confidence", 0.0)) >= confidence_threshold
    ]
    accepted_class_names = {
        str(item.get("class_name")) for item in accepted_detections
    }

    # V1 운영 판정: 예측 없음 → REVIEW / 임계값 미달 → REVIEW / no_good → HOLD / good → RELEASE
    if not detections:
        decision = {
            "status": "REVIEW",
            "reason": (
                f"V1 배포 모델에서 Confidence {V1_RETRIEVAL_CONFIDENCE:.2f} 이상 후보 예측이 생성되지 않음"
            ),
            "route": "같은 이미지를 더 낮은 조회 기준으로 재확인하거나 V1 배포 모델 성능/입력 조건을 점검하세요.",
        }
    elif not accepted_detections:
        decision = {
            "status": "REVIEW",
            "reason": (
                f"V1 예측은 생성됐지만 최고 신뢰도 {float(top_confidence) * 100:.1f}%가 "
                f"운영 기준 {confidence_threshold * 100:.1f}% 미만"
            ),
            "route": "BBox와 원본 이미지를 확인한 뒤 전문가 검토 또는 Confidence 기준 재설정을 검토하세요.",
        }
    elif "no_good" in accepted_class_names:
        no_good_conf = max(
            float(item.get("confidence", 0.0))
            for item in accepted_detections
            if item.get("class_name") == "no_good"
        )
        decision = {
            "status": "HOLD",
            "reason": f"V1에서 부품 위치 이상(no_good) 검출 · 신뢰도 {no_good_conf * 100:.1f}%",
            "route": "부품 실장 위치, 장착 프로그램 및 Setup을 재확인하세요.",
        }
    elif "good" in accepted_class_names:
        good_conf = max(
            float(item.get("confidence", 0.0))
            for item in accepted_detections
            if item.get("class_name") == "good"
        )
        decision = {
            "status": "RELEASE",
            "reason": f"V1 부품 실장 상태가 정상(good)으로 검출됨 · 신뢰도 {good_conf * 100:.1f}%",
            "route": "V2/V2.1 납땜 검사 결과와 함께 최종 로트 판정을 수행합니다.",
        }
    else:
        unknown_classes = ", ".join(sorted(accepted_class_names))
        decision = {
            "status": "REVIEW",
            "reason": f"V1 배포 모델에서 예상하지 않은 클래스 검출: {unknown_classes}",
            "route": "배포 모델의 클래스 맵과 LineGate CLASS_LABELS 설정을 확인하세요.",
        }

    overlay_b64 = _build_v1_overlay_b64(image_bytes, detections)

    response_request_id = None
    if isinstance(raw_result, dict):
        response_request_id = raw_result.get("request_id") or raw_result.get("id")

    return {
        "filename": filename,
        "request_id": response_request_id,
        "view": "V1",
        "source": "superb_ai_deployment",
        "deployment_id": V1_DEPLOYMENT_ID,
        "detections": detections,
        "summary": {
            "top_class": top_class,
            "top_confidence": top_confidence,
            "total_detection_count": len(detections),
            "accepted_detection_count": len(accepted_detections),
            "operating_confidence_threshold": confidence_threshold,
            "retrieval_confidence_threshold": V1_RETRIEVAL_CONFIDENCE,
        },
        "decision": decision,
        "inference": {
            "time_ms": inference_time_ms,
            "engine": "Superb AI Deployment",
        },
        "overlay": {
            "url": None,
            "image_b64": overlay_b64,
            "source": "local_render_from_superb_geometry" if overlay_b64 else None,
        },
        # 문제 발생 시 실제 SDK 응답을 화면 JSON에서 확인할 수 있도록 보존
        "raw_superb_response": raw_result,
    }


def request_v1_prediction(
    uploaded_file,
    confidence_threshold: float,
) -> dict[str, Any]:
    """V1 이미지를 Superb AI Deployment에 보내 기존 LineGate 결과 형식으로 반환한다."""
    image_bytes = uploaded_file.getvalue()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    client = get_superb_client()

    started_at = time.perf_counter()
    # 현재 공식 Superb AI SDK 문서의 deployment.predict는 image_b64를 받습니다.
    superb_result = client.deployments.predict(
        V1_DEPLOYMENT_ID,
        image_b64=image_b64,
        # 서버에서 운영 임계값(예: 0.70)으로 잘라버리면 낮은 신뢰도 예측 자체를 볼 수 없습니다.
        # 후보는 낮은 기준으로 받아오고 실제 판정 임계값은 convert_superb_v1_result에서 적용합니다.
        confidence=V1_RETRIEVAL_CONFIDENCE,
    )
    inference_time_ms = (time.perf_counter() - started_at) * 1000.0

    return convert_superb_v1_result(
        superb_result=superb_result,
        image_bytes=image_bytes,
        filename=uploaded_file.name,
        confidence_threshold=confidence_threshold,
        inference_time_ms=inference_time_ms,
    )


def request_prediction(
    uploaded_file, view: str, confidence_threshold: float
) -> dict[str, Any]:
    file_bytes = uploaded_file.getvalue()
    files = {
        "file": (
            uploaded_file.name,
            file_bytes,
            uploaded_file.type or "image/jpeg",
        )
    }
    data = {"view": view, "confidence_threshold": str(confidence_threshold)}

    response = requests.post(
        f"{API_BASE_URL}/predict/solder", files=files, data=data, timeout=120
    )
    response.raise_for_status()
    return response.json()


def fetch_overlay_image(overlay_url: str) -> Image.Image:
    """FastAPI overlay URL(상대/절대)을 불러온다."""
    request_url = (
        overlay_url
        if overlay_url.startswith(("http://", "https://"))
        else f"{API_BASE_URL}{overlay_url}"
    )
    response = requests.get(request_url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def load_result_overlay(result: dict[str, Any]) -> Image.Image | None:
    """V1의 로컬 base64 Overlay 또는 V2/V2.1의 FastAPI URL Overlay를 읽는다."""
    overlay = result.get("overlay", {}) or {}

    image_b64 = overlay.get("image_b64")
    if image_b64:
        try:
            return Image.open(BytesIO(base64.b64decode(image_b64))).convert("RGB")
        except Exception as exc:
            raise ValueError(f"base64 Overlay 디코딩 실패: {exc}") from exc

    overlay_url = overlay.get("url")
    if overlay_url:
        return fetch_overlay_image(str(overlay_url))

    return None


def get_class_label(class_name: str | None, view: str) -> str:
    """View 역할에 맞게 클래스명을 작업자 친화 문구로 변환한다."""
    if class_name == "good":
        return "정상 실장" if view == "V1" else "정상 납땜"
    return CLASS_LABELS.get(class_name or "", class_name or "N/A")


def detected_classes(result: dict[str, Any]) -> set[str]:
    """요약값과 객체 검출값을 함께 사용해 해당 View의 검출 클래스를 모은다."""
    classes = {
        str(item.get("class_name"))
        for item in result.get("detections", [])
        if item.get("class_name")
    }
    top_class = result.get("summary", {}).get("top_class")
    if top_class:
        classes.add(str(top_class))
    return classes


def get_lot_impact(view: str, result: dict[str, Any]) -> tuple[str, str]:
    """View 결과가 최종 로트 판정에서 갖는 의미를 작업자용 문구로 반환한다."""
    status = result.get("decision", {}).get("status", "REVIEW")
    classes = detected_classes(result)

    if status == "REVIEW":
        return "검토 필요", "#f59e0b"

    if view == "V1":
        if status == "HOLD" or "no_good" in classes:
            return "HOLD 원인", "#ef4444"
        return "위치 정상 증거", "#10b981"

    if status == "HOLD" or classes.intersection(SOLDER_DEFECT_CLASSES):
        return "HOLD 원인", "#ef4444"
    return "납땜 정상 증거", "#10b981"


def render_lot_decision_flow(
    results: dict[str, dict[str, Any]],
    combined_decision: dict[str, Any],
) -> None:
    """V1·V2·V2.1 결과가 최종 로트 판정으로 이어지는 흐름을 표시한다."""
    nodes: list[str] = []

    for view in ("V1", "V2", "V2.1"):
        result = results.get(view)
        if result is None:
            result_label = "결과 없음"
            confidence_label = "추론 결과 미생성"
            impact_label = "검토 필요"
        else:
            summary = result.get("summary", {})
            top_class = summary.get("top_class")
            top_confidence = summary.get("top_confidence")
            result_label = get_class_label(top_class, view)
            confidence_label = (
                f"신뢰도 {float(top_confidence) * 100:.1f}%"
                if top_confidence is not None
                else "신뢰도 N/A"
            )
            impact_label, _ = get_lot_impact(view, result)

        # Streamlit Markdown가 들여쓰기된 HTML을 코드 블록으로 해석하지 않도록
        # 각 노드를 공백 없는 단일 HTML 문자열로 생성한다.
        nodes.append(
            f'<div class="flow-node">'
            f'<div class="flow-node-title">{VIEW_LABELS.get(view, view)}</div>'
            f'<div class="flow-node-result">{result_label}</div>'
            f'<div class="flow-node-detail">{confidence_label}<br>{impact_label}</div>'
            f'</div>'
        )

    final_status = combined_decision.get("status", "REVIEW")
    final_reason = combined_decision.get("reason", "판정 사유 없음")
    final_class = {
        "RELEASE": "flow-final-release",
        "HOLD": "flow-final-hold",
        "REVIEW": "flow-final-review",
    }.get(final_status, "flow-final-review")

    # HTML 전체도 줄 시작 들여쓰기 없이 결합해야 Markdown 코드 블록으로 노출되지 않는다.
    flow_html = (
        f'<div class="flow-grid">{"".join(nodes)}</div>'
        f'<div class="flow-arrow">↓</div>'
        f'<div class="flow-final {final_class}">'
        f'<div class="flow-final-status">최종 로트 판정 · {final_status}</div>'
        f'<div class="flow-final-reason">{final_reason}</div>'
        f'</div>'
    )
    st.markdown(flow_html, unsafe_allow_html=True)


def render_decision(decision: dict[str, Any], is_main: bool = False) -> None:
    """판정 상태와 판정 사유를 우선 표시한다. 권장 조치는 검출 결과 뒤에서 별도로 표시한다."""
    status = decision.get("status", "REVIEW")
    reason = decision.get("reason", "판정 사유 없음")
    status_label = STATUS_LABELS.get(status, status)

    if is_main:
        css_class = {
            "RELEASE": "final-status-release",
            "HOLD": "final-status-hold",
            "REVIEW": "final-status-review",
        }.get(status, "final-status-review")
        st.markdown(
            f'<div class="final-status-card {css_class}">'
            f'<div class="final-status-eyebrow">Final lot decision</div>'
            f'<div class="final-status-row"><div class="final-status-value">{status}</div>'
            f'<div class="final-status-label">{status_label}</div></div>'
            f'<div class="final-reason-block"><div class="final-reason-label">판정 사유</div>'
            f'<div class="final-reason-text">{html.escape(str(reason))}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        return

    css_class = {
        "RELEASE": "decision-release",
        "HOLD": "decision-hold",
        "REVIEW": "decision-review",
    }.get(status, "decision-review")
    status_color = {
        "RELEASE": "#42be65",
        "HOLD": "#fa4d56",
        "REVIEW": "#f1c21b",
    }.get(status, "#c6c6c6")
    st.markdown(
        f'<div class="decision-card {css_class}">'
        f'<div style="display:flex;justify-content:space-between;gap:10px;align-items:center;">'
        f'<div style="font-size:1.05rem;font-weight:800;color:{status_color};">{status}</div>'
        f'<div style="font-size:.75rem;color:#a8a8a8;">{status_label}</div></div>'
        f'<div style="margin-top:9px;color:#c6c6c6;font-size:.84rem;line-height:1.5;">'
        f'<b style="color:#f4f4f4;">판정 사유</b><br>{html.escape(str(reason))}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_detection_card(detection: dict[str, Any], index: int, view: str) -> None:
    class_name = detection.get("class_name", "unknown")
    class_label = get_class_label(class_name, view)
    confidence = float(detection.get("confidence", 0))
    mask_area = detection.get("mask_area_px")
    polygon = detection.get("polygon") or []
    bbox = detection.get("bbox_xyxy") or []

    badge_class = "confidence-badge-good" if class_name == "good" else "confidence-badge-bad"
    region_text = "실장 영역" if view == "V1" else "납땜 영역"

    st.markdown(
        '<div class="detection-card">'
        '<div class="detection-row">'
        f'<span class="detection-title">#{index} {class_label}</span>'
        f'<span class="{badge_class}">{confidence * 100:.1f}%</span>'
        '</div>'
        f'<div class="detection-meta">{region_text} 검출 결과</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    coord_label = "기술 상세 보기"
    if st.checkbox(coord_label, key=f"coords_{view}_{index}"):
        st.caption(f"**Bounding Box:** `{bbox}`")
        if isinstance(mask_area, int):
            st.caption(f"**Mask 면적:** `{mask_area:,} px`")
        if polygon:
            st.caption(f"**Polygon Points:** {len(polygon)}개 (일부: `{polygon[:6]}`)")


def combine_view_decisions(
    results: dict[str, dict[str, Any]],
    required_views: list[str],
    verified_pair: bool = False,
) -> dict[str, Any]:
    """V1 위치 검사와 V2/V2.1 납땜 검사를 동시에 종합한다.

    판정 우선순위는 입력 누락 → 불확실 → 결함 HOLD → 전체 정상 RELEASE다.
    위치 이상만 발생한 경우와 납땜 이상이 함께 발생한 경우를 구분해 안내한다.
    """
    missing_views = [view for view in required_views if view not in results]
    if missing_views:
        return {
            "status": "REVIEW",
            "reason": "필수 검사 이미지 누락: " + ", ".join(VIEW_LABELS.get(v, v) for v in missing_views),
            "route": "누락된 View 이미지를 추가한 뒤 로트 판정을 다시 실행하세요.",
            "evidence_complete": False,
            "missing_views": missing_views,
            "view_conflict": False,
            "placement_defect": False,
            "solder_defect": False,
        }

    review_views = [
        view for view in required_views
        if results[view].get("decision", {}).get("status", "REVIEW") == "REVIEW"
    ]
    if review_views:
        return {
            "status": "REVIEW",
            "reason": "신뢰도 부족 또는 판정 불확실 검사 존재: " + ", ".join(VIEW_LABELS.get(v, v) for v in review_views),
            "route": "숙련 검사자가 원본과 Mask Overlay를 확인하세요.",
            "evidence_complete": True,
            "missing_views": [],
            "view_conflict": False,
            "placement_defect": False,
            "solder_defect": False,
        }

    v1_classes = detected_classes(results.get("V1", {}))
    placement_defect = (
        "V1" in required_views
        and (
            "no_good" in v1_classes
            or results["V1"].get("decision", {}).get("status") == "HOLD"
        )
    )

    solder_defect_views: list[str] = []
    for view in ("V2", "V2.1"):
        if view not in required_views:
            continue
        classes = detected_classes(results[view])
        if (
            classes.intersection(SOLDER_DEFECT_CLASSES)
            or results[view].get("decision", {}).get("status") == "HOLD"
        ):
            solder_defect_views.append(view)

    solder_defect = bool(solder_defect_views)

    # 동일 대상 Pairing이 확인된 경우에만 서로 반대되는 납땜 View 결과를 충돌로 표시한다.
    if verified_pair and all(view in required_views for view in ("V2", "V2.1")):
        v2_status = results["V2"].get("decision", {}).get("status")
        v21_status = results["V2.1"].get("decision", {}).get("status")
        if {v2_status, v21_status} == {"HOLD", "RELEASE"}:
            return {
                "status": "REVIEW",
                "reason": "동일 검사 대상의 V2·V2.1 납땜 판정이 서로 충돌함",
                "route": "두 View의 원본과 Mask를 비교해 실제 납땜 상태를 확정하세요.",
                "evidence_complete": True,
                "missing_views": [],
                "view_conflict": True,
                "placement_defect": placement_defect,
                "solder_defect": solder_defect,
            }

    if placement_defect and solder_defect:
        return {
            "status": "HOLD",
            "reason": (
                "실장 상태 이상과 납땜 이상이 함께 검출됨: "
                + ", ".join(VIEW_LABELS.get(v, v) for v in solder_defect_views)
            ),
            "route": "로트 생산을 보류하고 실장 Setup과 해당 납땜부를 모두 재검하세요.",
            "evidence_complete": True,
            "missing_views": [],
            "view_conflict": False,
            "placement_defect": True,
            "solder_defect": True,
        }

    if placement_defect:
        return {
            "status": "HOLD",
            "reason": "V1 실장 상태에서 위치 이상 검출 · 좌·우 납땜 이상은 미검출",
            "route": "로트 생산을 보류하고 부품 실장 위치, 장착 프로그램 및 Setup을 재확인하세요.",
            "evidence_complete": True,
            "missing_views": [],
            "view_conflict": False,
            "placement_defect": True,
            "solder_defect": False,
        }

    if solder_defect:
        return {
            "status": "HOLD",
            "reason": "V1 실장 상태는 정상이나 납땜 이상 검출: " + ", ".join(VIEW_LABELS.get(v, v) for v in solder_defect_views),
            "route": "로트 생산을 보류하고 결함 위치를 재검한 뒤 재작업 가능성을 검토하세요.",
            "evidence_complete": True,
            "missing_views": [],
            "view_conflict": False,
            "placement_defect": False,
            "solder_defect": True,
        }

    return {
        "status": "RELEASE",
        "reason": "실장 상태와 좌·우 납땜 검사 결과가 모두 정상",
        "route": "필수 검사 증거가 모두 정상으로 확인되어 다음 생산 진행을 권고합니다.",
        "evidence_complete": True,
        "missing_views": [],
        "view_conflict": False,
        "placement_defect": False,
        "solder_defect": False,
    }

def resolve_human_final_status(selected_action: str, ai_status: str) -> str:
    if selected_action == "APPROVE_AI":
        return ai_status
    if selected_action == "REWORK_REINSPECTION":
        return "HOLD"
    if selected_action == "ADDITIONAL_INSPECTION":
        return "REVIEW"
    return selected_action


def render_missing_view_card(view: str, original_image_bytes: bytes | None = None) -> None:
    with st.container(border=True):
        st.markdown(f"### {VIEW_LABELS.get(view, view)}")
        st.warning("검사 결과 없음")
        if original_image_bytes:
            try:
                image = Image.open(BytesIO(original_image_bytes)).convert("RGB")
                st.image(image, caption=f"{view} 업로드 원본", width="stretch")
            except Exception:
                st.caption("원본 이미지를 표시하지 못했습니다.")
        st.markdown("**상태:** 추론 결과가 생성되지 않았거나 필수 입력이 누락되었습니다.")
        st.info("가능한 원인: 모델 미연결, API View 미지원, 추론 오류 또는 입력 누락")


def build_human_approval_record(inspection_case: dict[str, Any], selected_action: str, reviewer: str, comment: str) -> dict[str, Any]:
    decision = inspection_case.get("combined_decision", {})
    ai_status = decision.get("status", "REVIEW")
    final_status = resolve_human_final_status(selected_action, ai_status)
    return {
        "line_id": inspection_case.get("line_id"),
        "batch_id": inspection_case.get("batch_id"),
        "ai_status": ai_status,
        "ai_reason": decision.get("reason", ""),
        "ai_route": decision.get("route", ""),
        "placement_defect": bool(decision.get("placement_defect")),
        "solder_defect": bool(decision.get("solder_defect")),
        "evidence_complete": bool(decision.get("evidence_complete")),
        "view_conflict": bool(decision.get("view_conflict")),
        "selected_action": selected_action,
        "final_status": final_status,
        "changed_from_ai": final_status != ai_status,
        "reviewer": reviewer.strip(),
        "comment": comment.strip(),
        "reviewed_at": datetime.now().astimezone().isoformat(),
    }


def render_view_result(
    view: str, result: dict[str, Any], original_image_bytes: bytes
) -> None:
    """작업자 결과 카드: Overlay를 주 증거로 표시하고 원본은 필요할 때만 펼쳐본다."""
    decision = result.get("decision", {})
    summary = result.get("summary", {})

    with st.container(border=True):
        st.markdown(f"### {VIEW_LABELS.get(view, view)}")
        render_decision(decision, is_main=False)

        # 결과 화면에서는 AI가 표시한 증거를 가장 먼저 크게 보여준다.
        # V1은 BBox Overlay, V2/V2.1은 Segmentation Mask Overlay를 사용한다.
        try:
            overlay_image = load_result_overlay(result)
            if overlay_image is not None:
                overlay_engine = (
                    "BBox Overlay"
                    if result.get("source") == "superb_ai_deployment"
                    else "Mask Overlay"
                )
                st.image(
                    overlay_image,
                    caption=f"{VIEW_LABELS.get(view, view)} · AI {overlay_engine}",
                    width="stretch",
                )
            else:
                st.info("표시 가능한 AI Overlay가 없습니다.")
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Overlay 로드 실패: {exc}")

        top_class = summary.get("top_class")
        top_confidence = summary.get("top_confidence")
        defect_label = get_class_label(top_class, view)
        defect_color = "#34d399" if top_class == "good" else ("#f87171" if top_class else "#94a3b8")
        conf_val_str = f"{float(top_confidence) * 100:.1f}%" if top_confidence is not None else "N/A"
        impact_label, impact_color = get_lot_impact(view, result)

        metric1, metric2, metric3 = st.columns(3)
        with metric1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">주요 결함</div>'
                f'<div class="metric-value" style="color:{defect_color} !important;">{defect_label}</div></div>',
                unsafe_allow_html=True,
            )
        with metric2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">최고 신뢰도</div>'
                f'<div class="metric-value" style="color:#38bdf8 !important;">{conf_val_str}</div></div>',
                unsafe_allow_html=True,
            )
        with metric3:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">로트 판정 영향</div>'
                f'<div class="metric-value" style="color:{impact_color} !important;font-size:1.25rem;">{impact_label}</div></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        detections = result.get("detections", [])
        st.markdown(f"**검출된 객체 상세 ({len(detections)}개)**")
        with st.container(height=360, border=True):
            if detections:
                for index, detection in enumerate(detections, start=1):
                    render_detection_card(detection=detection, index=index, view=view)
            else:
                if view == "V1":
                    st.warning("V1에서 표시할 후보 예측이 없습니다. 이는 정상 판정이 아니라 REVIEW 대상입니다.")
                else:
                    st.info("검출된 결함이 없습니다. 정상입니다.")

        # 원본은 기본 결과 화면에서 숨기고, 검토가 필요할 때만 확인한다.
        with st.expander("원본 이미지 보기", expanded=False):
            if original_image_bytes:
                try:
                    original_image = Image.open(BytesIO(original_image_bytes)).convert("RGB")
                    st.image(
                        original_image,
                        caption=f"{VIEW_LABELS.get(view, view)} · 원본 이미지",
                        width="stretch",
                    )
                    if decision.get("status") == "REVIEW":
                        st.caption("REVIEW 판정입니다. AI Overlay와 원본 이미지를 함께 비교해 확인하세요.")
                except Exception as exc:
                    st.warning(f"원본 이미지를 표시하지 못했습니다: {exc}")
            else:
                st.info("저장된 원본 이미지가 없습니다.")

        view_route = decision.get("route", "후속 조치 없음")
        st.markdown(
            f'<div class="action-panel" style="margin-top:12px;">'
            f'<div class="action-title">이 View의 권장 조치</div>'
            f'<div class="action-text">{html.escape(str(view_route))}</div></div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------
# 작업자용 상단 헤더
# ---------------------------------------------------------

api_ok, health_data = check_api_health()
fastapi_model_name = (
    health_data.get("model_name", "unknown")
    if (api_ok and health_data)
    else "N/A"
)
superb_v1_ok, superb_v1_info = check_superb_v1_ready()
system_ready = api_ok and superb_v1_ok

if system_ready:
    status_html = (
        '<div class="status-stack">'
        '<div class="status-pill status-online"><div class="dot-online"></div>SYSTEM READY</div>'
        '<div class="system-summary">필수 검사 엔진 3/3 준비</div>'
        '</div>'
    )
else:
    missing_parts = []
    if not superb_v1_ok:
        missing_parts.append("V1 실장 상태")
    if not api_ok:
        missing_parts.append("V2/V2.1 납땜")
    missing_text = " / ".join(missing_parts) or "Unknown"
    status_html = (
        '<div class="status-stack">'
        '<div class="status-pill status-offline"><div class="dot-offline"></div>SETUP REQUIRED</div>'
        f'<div class="system-summary" style="color:#fa4d56 !important;">확인 필요 · {missing_text}</div>'
        '</div>'
    )

header_html = (
    '<div class="dashboard-header"><div>'
    '<div class="dashboard-title"><span>LineGate AOI</span>'
    '<span class="dashboard-title-badge">FIRST-BOARD QUALITY GATE</span></div>'
    '<div class="dashboard-subtitle">실장 상태와 좌·우 납땜 증거를 통합해 생산 진행 여부를 지원합니다.</div>'
    '</div>'
    f'<div>{status_html}</div></div>'
)
st.markdown(header_html, unsafe_allow_html=True)

# ---------------------------------------------------------
# 사이드바 · 작업자 설정 / 고급 설정 분리
# ---------------------------------------------------------

required_views = ["V1", "V2", "V2.1"]
verified_pair = True

with st.sidebar:
    st.markdown("### 검사 설정")
    st.caption("현장 작업에 필요한 기본 정보만 입력합니다.")

    line_id = st.text_input("생산 라인", value="LINE-02")
    batch_id = st.text_input("배치 번호", value="B-204")

    st.markdown(
        '<div class="sidebar-summary-card">'
        '<div class="sidebar-summary-row"><div class="sidebar-summary-label">검사 대상</div>'
        '<div class="sidebar-summary-value">동일 부품 3각도</div></div>'
        '<div class="sidebar-summary-row"><div class="sidebar-summary-label">필수 검사</div>'
        '<div class="sidebar-summary-value">실장 상태 + 좌·우 납땜</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.expander("고급 설정", expanded=False):
        confidence_threshold = st.slider(
            "판정 신뢰도 기준",
            min_value=0.10,
            max_value=0.99,
            value=0.70,
            step=0.01,
            help="AI 결과를 RELEASE/HOLD로 확정할 최소 신뢰도입니다. 일반 작업자는 기본값 사용을 권장합니다.",
        )
        st.caption("일반 작업자는 기본값을 유지해도 됩니다.")

    with st.expander("시스템 정보", expanded=False):
        st.markdown(f"**V1 실장 상태** · Superb AI Detection")
        st.caption(f"Deployment: {V1_DEPLOYMENT_ID}")
        st.caption(f"Tenant: {SUPERB_TENANT}")
        st.markdown(f"**V2 / V2.1 납땜** · Local FastAPI")
        st.caption(f"Model: {fastapi_model_name}")
        st.caption("상태: " + ("정상" if system_ready else "확인 필요"))

# ---------------------------------------------------------
# 1. 검사 이미지 준비
# ---------------------------------------------------------

st.markdown("### 1. 검사 이미지 준비")
st.caption("동일 부품의 세 시점 이미지를 모두 등록하면 검사를 시작할 수 있습니다.")

upload_col1, upload_col2, upload_col3 = st.columns(3)

def _render_upload_head(slot, view: str, uploaded_file) -> None:
    completed = uploaded_file is not None
    state_label = "업로드 완료" if completed else "이미지 필요"
    state_class = "upload-complete" if completed else "upload-required"
    slot.markdown(
        f'<div class="upload-head"><div class="upload-head-top">'
        f'<div class="upload-view-title">{VIEW_LABELS[view]}</div>'
        f'<div class="upload-state {state_class}">{state_label}</div></div>'
        f'<div class="upload-view-desc">{VIEW_DESCRIPTIONS[view]}</div></div>',
        unsafe_allow_html=True,
    )

with upload_col1:
    v1_status_slot = st.empty()
    uploaded_v1 = st.file_uploader(
        "V1 실장 상태 이미지", type=["jpg", "jpeg", "png"], key="upload_v1",
        label_visibility="collapsed",
    )
    _render_upload_head(v1_status_slot, "V1", uploaded_v1)
    if uploaded_v1 is not None:
        st.image(Image.open(uploaded_v1), caption=f"V1 실장 상태 · {uploaded_v1.name}", width="stretch")

with upload_col2:
    v2_status_slot = st.empty()
    uploaded_v2 = st.file_uploader(
        "V2 좌측 납땜 이미지", type=["jpg", "jpeg", "png"], key="upload_v2",
        label_visibility="collapsed",
    )
    _render_upload_head(v2_status_slot, "V2", uploaded_v2)
    if uploaded_v2 is not None:
        st.image(Image.open(uploaded_v2), caption=f"V2 좌측 납땜 · {uploaded_v2.name}", width="stretch")

with upload_col3:
    v21_status_slot = st.empty()
    uploaded_v21 = st.file_uploader(
        "V2.1 우측 납땜 이미지", type=["jpg", "jpeg", "png"], key="upload_v21",
        label_visibility="collapsed",
    )
    _render_upload_head(v21_status_slot, "V2.1", uploaded_v21)
    if uploaded_v21 is not None:
        st.image(Image.open(uploaded_v21), caption=f"V2.1 우측 납땜 · {uploaded_v21.name}", width="stretch")

# ---------------------------------------------------------
# 추론 실행 버튼
# ---------------------------------------------------------

uploaded_files = {"V1": uploaded_v1, "V2": uploaded_v2, "V2.1": uploaded_v21}
selected_uploads = {
    view: uploaded_files[view]
    for view in required_views
    if uploaded_files.get(view) is not None
}

uploaded_count = len(selected_uploads)
can_run = api_ok and superb_v1_ok and uploaded_count == 3
readiness_class = "ready-ok" if uploaded_count == 3 else "ready-wait"
readiness_text = "검사 준비 완료" if uploaded_count == 3 else f"필수 이미지 {uploaded_count}/3 등록"
st.markdown(
    f'<div class="input-readiness"><div class="input-readiness-label">입력 상태</div>'
    f'<div class="input-readiness-value {readiness_class}">{readiness_text}</div></div>',
    unsafe_allow_html=True,
)

if not superb_v1_ok:
    st.warning(
        "V1 Superb AI 연결 준비 필요: "
        + superb_v1_info.get("message", "설정을 확인하세요.")
    )

if not api_ok:
    st.warning(
        "V2/V2.1 FastAPI 서버에 연결할 수 없습니다. "
        "기존 uvicorn 서버를 먼저 실행하세요."
    )

st.write("")
run_prediction = st.button(
    "AI 검사 시작",
    type="primary",
    disabled=not can_run,
    use_container_width=True,
)

if run_prediction:
    st.session_state["inspection_results"] = {}
    st.session_state["original_images"] = {}
    st.session_state["human_approval"] = None

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_views = len(selected_uploads)

    for index, (current_view, uploaded_file) in enumerate(selected_uploads.items(), start=1):
        status_text.write(f"{VIEW_LABELS.get(current_view, current_view)} 이미지를 분석하고 있습니다...")

        try:
            if current_view == "V1":
                result = request_v1_prediction(
                    uploaded_file=uploaded_file,
                    confidence_threshold=confidence_threshold,
                )
            else:
                result = request_prediction(
                    uploaded_file=uploaded_file,
                    view=current_view,
                    confidence_threshold=confidence_threshold,
                )

            st.session_state["inspection_results"][current_view] = result
            st.session_state["original_images"][current_view] = uploaded_file.getvalue()

        except requests.HTTPError as exc:
            st.error(f"{current_view} 추론 실패: {exc}")
        except requests.RequestException as exc:
            st.error(f"{current_view} API 통신 실패: {exc}")
        except Exception as exc:
            # Superb AI SDK의 Authentication/Unavailable/RateLimit 등 typed error도 여기서
            # code/hint/request_id를 포함해 화면에 노출합니다.
            if current_view == "V1":
                st.error(f"V1 Superb AI 추론 실패: {_format_superb_error(exc)}")
            else:
                st.error(f"{current_view} 추론 중 예기치 않은 오류: {exc}")

        progress_bar.progress(index / total_views)

    results = st.session_state.get("inspection_results", {})
    combined_decision = combine_view_decisions(
        results=results,
        required_views=required_views,
        verified_pair=verified_pair,
    )

    st.session_state["inspection_case"] = {
        "line_id": line_id,
        "batch_id": batch_id,
        "required_views": required_views,
        "verified_pair": verified_pair,
        "confidence_threshold": confidence_threshold,
        "combined_decision": combined_decision,
    }

    status_text.empty()
    progress_bar.empty()
    st.success("모든 필수 View 분석과 로트 합격 판정이 완료되었습니다.")

# ---------------------------------------------------------
# 2. 결과 출력 영역
# ---------------------------------------------------------

results = st.session_state.get("inspection_results", {})
inspection_case = st.session_state.get("inspection_case", {})

if results and inspection_case:
    st.divider()

    combined_decision = inspection_case.get("combined_decision", {})

    # ---------------------------------------------------------
    # 2. 최종 판정 · 가장 먼저 크게 표시
    # ---------------------------------------------------------
    st.markdown("### 2. 최종 로트 판정")
    render_decision(combined_decision, is_main=True)

    # ---------------------------------------------------------
    # 3. 검출 결과 · 판정 사유 다음에 증거 확인
    # ---------------------------------------------------------
    st.markdown("### 3. 검출 결과 및 검사 증거")
    st.caption("AI Overlay를 중심으로 실장 상태와 좌·우 납땜 결과를 비교합니다. 원본 이미지는 각 카드의 ‘원본 이미지 보기’에서 확인할 수 있습니다.")

    board_views = ["V1", "V2", "V2.1"]
    evidence_columns = st.columns(3)
    original_images = st.session_state.get("original_images", {})

    for column, view_name in zip(evidence_columns, board_views):
        with column:
            if view_name in results:
                render_view_result(
                    view=view_name,
                    result=results[view_name],
                    original_image_bytes=original_images.get(view_name, b""),
                )
            else:
                uploaded_file = uploaded_files.get(view_name)
                uploaded_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
                render_missing_view_card(view_name, uploaded_bytes)

    st.markdown("#### 3각도 검사 흐름")
    st.caption("세 검사 결과가 최종 로트 판정에 어떻게 반영됐는지 보여줍니다.")
    render_lot_decision_flow(results, combined_decision)

    # ---------------------------------------------------------
    # 4. 권장 조치 · 검출 결과 확인 후 행동 제시
    # ---------------------------------------------------------
    st.markdown("### 4. 권장 조치")
    main_route = combined_decision.get("route", "후속 조치 없음")
    st.markdown(
        f'<div class="action-panel"><div class="action-title">NEXT ACTION</div>'
        f'<div class="action-text">{html.escape(str(main_route))}</div></div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # 5. 검사 요약 · 메타데이터/비교표는 후순위
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### 5. 검사 요약")

    total_views_count = len(results)
    total_defects_count = sum(
        r.get("summary", {}).get("total_detection_count", 0)
        for r in results.values()
    )
    overall_status = combined_decision.get("status", "REVIEW")
    overall_color = {
        "RELEASE": "#42be65",
        "HOLD": "#fa4d56",
        "REVIEW": "#f1c21b",
    }.get(overall_status, "#f4f4f4")

    info1, info2, info3, info4 = st.columns(4)
    with info1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">생산 라인</div>'
            f'<div class="metric-value">{html.escape(str(inspection_case.get("line_id", "N/A")))}</div></div>',
            unsafe_allow_html=True,
        )
    with info2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">배치 번호</div>'
            f'<div class="metric-value">{html.escape(str(inspection_case.get("batch_id", "N/A")))}</div></div>',
            unsafe_allow_html=True,
        )
    with info3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">검사 완료</div>'
            f'<div class="metric-value">{total_views_count} / 3</div></div>',
            unsafe_allow_html=True,
        )
    with info4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">최종 상태</div>'
            f'<div class="metric-value" style="color:{overall_color} !important;">{overall_status}</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    table_rows = ""
    for view_name in ("V1", "V2", "V2.1"):
        if view_name not in results:
            continue
        result = results[view_name]
        summary = result.get("summary", {})
        decision = result.get("decision", {})
        top_confidence = summary.get("top_confidence")
        status = decision.get("status", "REVIEW")
        raw_top_class = summary.get("top_class")
        top_class = get_class_label(raw_top_class, view_name)
        top_conf = f"{float(top_confidence) * 100:.1f}%" if top_confidence is not None else "N/A"
        detection_count = summary.get("total_detection_count", 0)
        reason = decision.get("reason", "-")

        status_color = {
            "RELEASE": "#42be65",
            "HOLD": "#fa4d56",
            "REVIEW": "#f1c21b",
        }.get(status, "#a8a8a8")
        defect_color = "#42be65" if raw_top_class == "good" else ("#fa4d56" if raw_top_class else "#a8a8a8")

        table_rows += (
            '<tr>'
            f'<td style="font-weight:700;">{VIEW_LABELS.get(view_name, view_name)}</td>'
            f'<td><span style="color:{defect_color};font-weight:700;">{top_class}</span></td>'
            f'<td style="color:#78a9ff !important;font-weight:700;">{top_conf}</td>'
            f'<td>{detection_count}개</td>'
            f'<td><span class="status-chip" style="background:{status_color}14;color:{status_color} !important;border:1px solid {status_color}66;">{status}</span></td>'
            f'<td style="color:#c6c6c6 !important;">{html.escape(str(reason))}</td>'
            '</tr>'
        )

    table_html = (
        '<div class="custom-table-container"><table class="custom-table">'
        '<thead><tr><th>검사 항목</th><th>주요 결과</th><th>신뢰도</th>'
        '<th>검출 수</th><th>판정</th><th>판정 사유</th></tr></thead>'
        f'<tbody>{table_rows}</tbody></table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 6. Human Approval
    # ---------------------------------------------------------
    st.divider()
    st.markdown("### 6. 최종 승인")
    st.caption("AI 권고와 검사 증거를 확인한 뒤 담당자가 최종 조치를 저장합니다.")

    ai_status = combined_decision.get("status", "REVIEW")
    placement_defect = bool(combined_decision.get("placement_defect"))
    solder_defect = bool(combined_decision.get("solder_defect"))
    evidence_complete = bool(combined_decision.get("evidence_complete"))

    h1, h2, h3 = st.columns(3)
    h1.metric("V1 실장 상태", "이상" if placement_defect else ("정상" if "V1" in results else "결과 없음"))
    h2.metric("좌·우 납땜 상태", "이상" if solder_defect else ("정상" if all(v in results for v in ("V2", "V2.1")) else "결과 없음"))
    h3.metric("필수 증거", "완전" if evidence_complete else "불완전")

    if ai_status == "RELEASE":
        st.success("세 검사 항목이 모두 정상입니다. 증거 확인 후 RELEASE를 승인할 수 있습니다.")
    elif ai_status == "HOLD":
        if placement_defect and solder_defect:
            st.error("실장 이상과 납땜 이상이 함께 확인됐습니다. 생산 보류 후 두 영역을 모두 재확인하세요.")
        elif placement_defect:
            st.error("실장 상태 이상으로 HOLD입니다. 납땜이 정상이더라도 생산 진행을 승인하지 않습니다.")
        else:
            st.error("납땜 이상으로 HOLD입니다. 결함 위치 확인 후 재작업 및 재검 여부를 결정하세요.")
    else:
        st.warning("불확실한 결과가 있어 자동 확정할 수 없습니다. 추가 확인 후 최종 판단하세요.")

    approval_options = {
        "APPROVE_AI": f"AI 권고 승인 ({ai_status})",
        "RELEASE": "예외 RELEASE 승인",
        "HOLD": "HOLD 유지",
        "REWORK_REINSPECTION": "REWORK 후 재검",
        "ADDITIONAL_INSPECTION": "추가 검사 요청",
    }

    with st.form("human_approval_form"):
        reviewer = st.text_input("검사자 이름", value="Quality Engineer 01")
        selected_action = st.radio(
            "전문가 최종 조치",
            options=list(approval_options),
            format_func=lambda value: approval_options[value],
        )
        comment = st.text_area(
            "검사자 의견 및 변경 사유",
            placeholder="AI 권고 변경, REWORK 또는 추가 검사 선택 시 판단 근거를 입력하세요.",
            height=120,
        )
        confirm_checked = st.checkbox("V1 실장 상태와 V2/V2.1 납땜 증거를 확인했습니다.")
        save_approval = st.form_submit_button("최종 판단 저장", type="primary", use_container_width=True)

    if save_approval:
        final_status = resolve_human_final_status(selected_action, ai_status)
        requires_comment = final_status != ai_status or selected_action in {"REWORK_REINSPECTION", "ADDITIONAL_INSPECTION"}
        if not reviewer.strip():
            st.error("검사자 이름을 입력하세요.")
        elif not confirm_checked:
            st.error("세 검사 증거를 확인했다는 체크가 필요합니다.")
        elif requires_comment and not comment.strip():
            st.error("AI 권고 변경, REWORK 또는 추가 검사 선택 시 판단 사유가 필요합니다.")
        elif selected_action == "RELEASE" and not evidence_complete:
            st.error("필수 증거가 불완전한 상태에서는 예외 RELEASE를 저장할 수 없습니다. 추가 검사를 요청하세요.")
        else:
            st.session_state["human_approval"] = build_human_approval_record(
                inspection_case, selected_action, reviewer, comment
            )
            st.success("전문가 최종 판단이 저장되었습니다.")

    approval_record = st.session_state.get("human_approval")
    if approval_record:
        with st.container(border=True):
            st.markdown("### 저장된 승인 결과")
            r1, r2, r3 = st.columns(3)
            r1.metric("AI 권고", approval_record["ai_status"])
            r2.metric("전문가 최종 상태", approval_record["final_status"])
            r3.metric("AI 권고 변경", "예" if approval_record["changed_from_ai"] else "아니오")
            st.markdown(f"**검사자:** {approval_record['reviewer']}")
            st.markdown(f"**저장 시각:** {approval_record['reviewed_at']}")
            if approval_record.get("comment"):
                st.markdown(f"**검사자 의견:** {approval_record['comment']}")
            if approval_record["selected_action"] == "REWORK_REINSPECTION":
                st.warning("현재 상태는 HOLD입니다. 재작업 완료 후 세 이미지를 새 검사 건으로 다시 검사해야 합니다.")
            elif approval_record["selected_action"] == "ADDITIONAL_INSPECTION":
                st.warning("현재 상태는 REVIEW입니다. 추가 증거 확보 전에는 생산 진행을 승인하지 않습니다.")
            st.download_button(
                "승인 기록 JSON 다운로드",
                data=json.dumps(approval_record, ensure_ascii=False, indent=2),
                file_name=f"{inspection_case.get('batch_id', 'batch')}_human_approval.json",
                mime="application/json",
                use_container_width=True,
            )
