import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import tempfile
import os
import yt_dlp
from ultralytics import YOLO

st.set_page_config(
    page_title="ForestSafe | 農林作業安全フォーム診断",
    layout="wide"
)

HISTORY_FILE = "diagnosis_history.csv"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            return pd.read_csv(HISTORY_FILE).to_dict(orient="records")
        except Exception:
            return []
    return []

def save_history(history_list):
    df = pd.DataFrame(history_list)
    df.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
st.session_state.history = load_history()

# 洗練されたモダン・フィールドテック風CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Noto Sans JP', sans-serif;
        color: #181C1A;
    }

    /* 背景：上品なウォームストーン（#F5F4F0） */
    .stApp {
        background-color: #F5F4F0;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* サイドバー */
    section[data-testid="stSidebar"] {
        background-color: #ECEAE4 !important;
        border-right: 1px solid #DCD9D0;
    }

    /* トップヘッダー：漆黒×ディープパインのモダンなバナー */
    .modern-header {
        background: linear-gradient(135deg, #141A17 0%, #1E2B24 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: #F8F9FA;
        margin-bottom: 1.75rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 12px 30px rgba(20, 26, 23, 0.12);
        border: 1px solid #2A3830;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(116, 198, 157, 0.12);
        border: 1px solid rgba(116, 198, 157, 0.3);
        color: #84D2A7;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #52B788;
        border-radius: 50%;
        display: inline-block;
    }
    .modern-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin: 0;
        color: #FFFFFF;
    }
    .modern-sub {
        font-size: 0.92rem;
        color: #A3B1AA;
        margin-top: 0.45rem;
    }

    /* カードコンテナ：クリーンな白＋繊細な1pxボーダー */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border-radius: 14px !important;
        border: 1px solid #E2DFD7 !important;
        box-shadow: 0 4px 16px rgba(20, 26, 23, 0.03) !important;
        padding: 0.75rem !important;
    }

    /* セクション見出し（SVGアイコン付き） */
    .sec-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.05rem;
        font-weight: 700;
        color: #141A17;
        margin-bottom: 0.75rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #F0EEE9;
    }

    /* アクションボタン：モダンなダークチャコール */
    div.stButton > button {
        background-color: #141A17;
        color: #FFFFFF;
        font-weight: 600;
        border: 1px solid #141A17;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 0.95rem;
        letter-spacing: 0.04em;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background-color: #26362E;
        border-color: #26362E;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(20, 26, 23, 0.16);
        color: #FFFFFF;
    }

    /* カスタムKPIカード */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 1rem 0;
    }
    .kpi-box {
        background: #F9F8F6;
        border: 1px solid #E6E3DC;
        border-radius: 10px;
        padding: 1rem 1.1rem;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #646E68;
        margin-bottom: 0.3rem;
    }
    .kpi-val-safe {
        font-size: 1.65rem;
        font-weight: 700;
        color: #1B6B46;
    }
    .kpi-val-danger {
        font-size: 1.65rem;
        font-weight: 700;
        color: #C2410C;
    }
    .kpi-val-neutral {
        font-size: 1.65rem;
        font-weight: 700;
        color: #2C3531;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("yolo11n-pose.pt")

def is_holding_tool_fast(kpts):
    l_shoulder, r_shoulder = kpts[5], kpts[6]
    l_wrist, r_wrist = kpts[9], kpts[10]
    l_hip, r_hip = kpts[11], kpts[12]

    if l_wrist[0] <= 0 or r_wrist[0] <= 0:
        return False

    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
    if shoulder_width < 10:
        return False

    wrist_dist = np.hypot(l_wrist[0] - r_wrist[0], l_wrist[1] - r_wrist[1])
    if wrist_dist < shoulder_width * 0.25 or wrist_dist > shoulder_width * 3.0:
        return False

    hip_y = max(l_hip[1], r_hip[1])
    if hip_y > 0 and (l_wrist[1] > hip_y + shoulder_width * 0.5 and r_wrist[1] > hip_y + shoulder_width * 0.5):
        return False

    return True

# サイドバー
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; color: #52796F;">FIELD ANALYTICS</div>
        <div style="font-size: 1.35rem; font-weight: 700; color: #141A17;">ForestSafe</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("**評価アルゴリズム基準**")
    st.markdown("""
    <div style="font-size: 0.86rem; line-height: 1.65; color: #3A423E;">
        <p><strong>01 / チェンソー作業</strong><br>機材保持中、手首座標が肩ラインより上方へ上がった場合に危険姿勢と判定します。</p>
        <p><strong>02 / 刈払機作業</strong><br>機材保持中、右手首が腰ラインより大きく右側へ振り抜かれた場合に危険姿勢と判定します。</p>
        <p><strong>03 / 稼働区間フィルタ</strong><br>両手首と体幹の距離から機材保持中のみを自動抽出し、移動・待機シーンを除外します。</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if st.session_state.analysis_result is not None or len(st.session_state.history) > 0:
        if st.button("履歴データをリセット"):
            st.session_state.analysis_result = None
            st.session_state.history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()

# モダンヘッダーバナー
st.markdown("""
<div class="modern-header">
    <div>
        <div class="header-badge"><span class="status-dot"></span>POSE ESTIMATION // ACTIVE</div>
        <div class="modern-title">農林作業 安全フォーム診断システム</div>
        <div class="modern-sub">骨格座標解析と稼働フィルタリングによる作業姿勢の定量評価</div>
    </div>
    <div style="opacity: 0.85;">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#84D2A7" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2L7 9h3l-4 6h4l-3 5h10l-3-5h4l-4-6h3L12 2z"/>
            <line x1="12" y1="20" x2="12" y2="23"/>
        </svg>
    </div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.35], gap="large")

video_path = None
video_name = "未指定"

with col_left:
    with st.container(border=True):
        st.markdown("""
        <div class="sec-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
            <span>01. 対象機材の選択</span>
        </div>
        """, unsafe_allow_html=True)
        tool_type = st.radio(
            "診断する機材モード",
            ["チェンソー", "刈払機（草刈機）"],
            horizontal=True,
            label_visibility="collapsed"
        )

    with st.container(border=True):
        st.markdown("""
        <div class="sec-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2D6A4F" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
