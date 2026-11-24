import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
import tempfile
import os
import re
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

def normalize_youtube_url(raw_url: str) -> str:
    url = raw_url.strip()
    patterns = [
        r"(?:v=|\/v\/|embed\/|youtu\.be\/|\/shorts\/)([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return f"https://www.youtube.com/watch?v={match.group(1)}"
    return url

def fetch_youtube_video(url: str):
    clean_url = normalize_youtube_url(url)
    if st.session_state.get("yt_cached_url") == clean_url and st.session_state.get("yt_cached_path"):
        if os.path.exists(st.session_state["yt_cached_path"]):
            return st.session_state["yt_cached_path"], clean_url, None

    temp_dir = tempfile.mkdtemp()
    output_template = os.path.join(temp_dir, "youtube_video.%(ext)s")

    configs = [
        {"client": ["tv"], "fmt": "best[height<=720]/best"},
        {"client": ["ios"], "fmt": "bestvideo[height<=720]/best[height<=720]/best"},
        {"client": ["android_vr"], "fmt": "bestvideo[height<=720]/best"},
    ]

    last_error = ""
    for cfg in configs:
        ydl_opts = {
            "format": cfg["fmt"],
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "geo_bypass": True,
            "nocheckcertificate": True,
            "socket_timeout": 20,
            "extractor_args": {
                "youtube": {
                    "player_client": cfg["client"]
                }
            }
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(clean_url, download=True)
                downloaded_path = ydl.prepare_filename(info)
            if os.path.exists(downloaded_path) and os.path.getsize(downloaded_path) > 0:
                st.session_state["yt_cached_url"] = clean_url
                st.session_state["yt_cached_path"] = downloaded_path
                return downloaded_path, clean_url, None
        except Exception as e:
            last_error = str(e)
            continue

    return None, clean_url, last_error

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
st.session_state.history = load_history()

# 高コントラスト・ダークモードCSS（文字の視認性を徹底改善）
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Noto Sans JP', sans-serif;
        color: #F3F6F4 !important;
    }

    /* 全体背景 */
    .stApp {
        background-color: #111614;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* サイドバー */
    section[data-testid="stSidebar"] {
        background-color: #161D1A !important;
        border-right: 1px solid #283530;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: #E2EAE6 !important;
    }

    /* トップヘッダーバナー */
    .modern-header {
        background: linear-gradient(135deg, #18241F 0%, #21332B 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        border: 1px solid #2F453B;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(82, 183, 136, 0.18);
        border: 1px solid rgba(116, 198, 157, 0.4);
        color: #8CE0B5 !important;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        margin-bottom: 0.65rem;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #52B788;
        border-radius: 50%;
        display: inline-block;
    }
    .modern-title {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin: 0;
        color: #FFFFFF !important;
    }
    .modern-sub {
        font-size: 0.92rem;
        color: #C2D1CA !important;
        margin-top: 0.4rem;
    }

    /* カードコンテナ：文字が浮き立つ明るいダークスレート */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #1A221F !important;
        border-radius: 14px !important;
        border: 1px solid #2E3D37 !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25) !important;
        padding: 0.85rem !important;
    }

    /* セクション見出し */
    .sec-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.05rem;
        font-weight: 700;
        color: #FFFFFF !important;
        margin-bottom: 0.85rem;
        padding-bottom: 0.55rem;
        border-bottom: 1px solid #2D3B35;
    }

    /* ラジオボタンの文字をクッキリ白に修正 */
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label p,
    div[data-testid="stRadio"] label span {
        color: #F3F6F4 !important;
        font-weight: 600 !important;
        font-size: 0.98rem !important;
    }

    /* タブの文字をクッキリ修正 */
    button[data-baseweb="tab"] {
        color: #B4C4BC !important;
        font-weight: 600 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #74C69D !important;
    }

    /* キャプション・ラベル類の視認性アップ */
    div[data-testid="stCaptionContainer"] p,
    label[data-testid="stWidgetLabel"] p {
        color: #D0DDD7 !important;
    }

    /* ファイルアップロード枠のダーク化（白浮き防止） */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #131917 !important;
        border: 1px dashed #3B4F47 !important;
        border-radius: 10px !important;
    }
    section[data-testid="stFileUploaderDropzone"] span,
    section[data-testid="stFileUploaderDropzone"] small,
    section[data-testid="stFileUploaderDropzone"] div {
        color: #D5E0DB !important;
    }
    section[data-testid="stFileUploaderDropzone"] button {
        background-color: #26352F !important;
        color: #FFFFFF !important;
        border: 1px solid #40594F !important;
    }

    /* テキスト入力欄のダーク化（白浮き防止） */
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] {
        background-color: #131917 !important;
        border: 1px solid #384A42 !important;
        border-radius: 8px !important;
    }
    input {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
    }
    input::placeholder {
        color: #7F9188 !important;
        -webkit-text-fill-color: #7F9188 !important;
    }

    /* アラート・エラーボックスのダーク馴染み */
    div[data-testid="stAlert"] {
        background-color: #1E2422 !important;
        border: 1px solid #3A4742 !important;
        color: #F3F6F4 !important;
    }
    div[data-testid="stAlert"] p {
        color: #F3F6F4 !important;
    }

    /* アクションボタン */
    div.stButton > button {
        background: linear-gradient(135deg, #2D6A4F 0%, #1B4332 100%);
        color: #FFFFFF !important;
        font-weight: 700;
        border: 1px solid #52B788;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 0.98rem;
        letter-spacing: 0.04em;
        width: 100%;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #40916C 0%, #2D6A4F 100%);
        border-color: #95D5B2;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(82, 183, 136, 0.3);
        color: #FFFFFF !important;
    }

    /* カスタムKPIカード */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin: 1rem 0;
    }
    .kpi-box {
        background: #131917;
        border: 1px solid #2D3B35;
        border-radius: 10px;
        padding: 1rem 1.1rem;
    }
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #B4C4BC !important;
        margin-bottom: 0.3rem;
    }
    .kpi-val-safe {
        font-size: 1.65rem;
        font-weight: 700;
        color: #52B788 !important;
    }
    .kpi-val-danger {
        font-size: 1.65rem;
        font-weight: 700;
        color: #FB923C !important;
    }
    .kpi-val-neutral {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F3F6F4 !important;
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

with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem 0;">
        <div style="font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em; color: #74C69D;">FIELD ANALYTICS</div>
        <div style="font-size: 1.35rem; font-weight: 700; color: #FFFFFF;">ForestSafe</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("**評価アルゴリズム基準**")
    st.markdown("""
    <div style="font-size: 0.86rem; line-height: 1.65; color: #D0DDD7;">
        <p><strong style="color:#FFFFFF;">01 / チェンソー作業</strong><br>機材保持中、手首座標が肩ラインより上方へ上がった場合に危険姿勢と判定します。</p>
        <p><strong style="color:#FFFFFF;">02 / 刈払機作業</strong><br>機材保持中、右手首が腰ラインより大きく右側へ振り抜かれた場合に危険姿勢と判定します。</p>
        <p><strong style="color:#FFFFFF;">03 / 稼働区間フィルタ</strong><br>両手首と体幹の距離から機材保持中のみを自動抽出し、移動・待機シーンを除外します。</p>
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

st.markdown("""
<div class="modern-header">
    <div>
        <div class="header-badge"><span class="status-dot"></span>POSE ESTIMATION // ACTIVE</div>
        <div class="modern-title">農林作業 安全フォーム診断システム</div>
        <div class="modern-sub">骨格座標解析と稼働フィルタリングによる作業姿勢の定量評価</div>
    </div>
    <div style="opacity: 0.9;">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
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
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
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
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
            </svg>
            <span>02. 映像データの入力</span>
        </div>
        """, unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["動画ファイル", "YouTube URL"])

        with tab1:
            uploaded_file = st.file_uploader("MP4 / MOV / AVI 形式に対応", type=["mp4", "mov", "avi"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                video_name = uploaded_file.name
                st.caption(f"セット完了: {uploaded_file.name}")

        with tab2:
            youtube_url = st.text_input(
                "YouTubeのURLを貼り付けてEnterを押してください",
                placeholder="https://www.youtube.com/watch?v=...",
                key="yt_url_input"
            )
            if youtube_url and youtube_url.strip():
                with st.spinner("YouTubeから映像データを取得しています..."):
                    fetched_path, normalized_url, err_msg = fetch_youtube_video(youtube_url)
                if fetched_path:
                    video_path = fetched_path
                    video_name = normalized_url
                    st.success(f"映像データの準備が完了しました（{normalized_url}）")
                else:
                    st.error("YouTube側の制限により自動取得できませんでした。お手数ですが動画をMP4として保存し、「動画ファイル」タブからアップロードしてください。")
                    if err_msg:
                        with st.expander("エラー詳細"):
                            st.code(err_msg)

    if video_path is not None:
        start_btn = st.button("解析を実行する")
    else:
        start_btn = False
        st.caption("映像データをセットすると解析ボタンが有効になります。")

with col_right:
    with st.container(border=True):
        st.markdown("""
        <div class="sec-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
            </svg>
            <span>03. 解析モニター・診断レポート</span>
        </div>
        """, unsafe_allow_html=True)

        if start_btn and video_path is not None:
            model = load_model()
            cap = cv2.VideoCapture(video_path)

            progress_text = st.empty()
            progress_bar = st.progress(0)
            st_frame = st.empty()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0
            skip_frames = 8

            safe_frames = 0
            danger_frames = 0
            standby_frames = 0
            last_annotated_frame = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if frame_count % skip_frames != 0:
                    continue

                if total_frames > 0:
                    percent = min(int((frame_count / total_frames) * 100), 100)
                    progress_bar.progress(percent)
                    progress_text.caption(f"解析進行度: {percent}%")

                h, w = frame.shape[:2]
                if w > 480:
                    scale = 480 / w
                    frame = cv2.resize(frame, (480, int(h * scale)))

                results = model(frame, verbose=False)
                annotated_frame = results[0].plot()

                if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                    kpts = results[0].keypoints.xy[0].cpu().numpy()
                    if len(kpts) >= 13:
                        if is_holding_tool_fast(kpts):
                            l_shoulder, r_shoulder = kpts[5], kpts[6]
                            l_wrist, r_wrist = kpts[9], kpts[10]
                            l_hip, r_hip = kpts[11], kpts[12]

                            is_danger = False
                            if tool_type == "チェンソー":
                                if (l_wrist[1] > 0 and l_shoulder[1] > 0 and l_wrist[1] < l_shoulder[1]) or \
                                   (r_wrist[1] > 0 and r_shoulder[1] > 0 and r_wrist[1] < r_shoulder[1]):
                                    is_danger = True
                            else:
                                if r_wrist[0] > 0 and r_hip[0] > 0 and r_wrist[0] < r_hip[0] - 25:
                                    is_danger = True

                            if is_danger:
                                danger_frames += 1
                                status_text = "DANGER"
                                color = (0, 0, 255)
                            else:
                                safe_frames += 1
                                status_text = "SAFE"
                                color = (0, 200, 0)
                        else:
                            standby_frames += 1
                            status_text = "STANDBY"
                            color = (180, 180, 180)

                        cv2.putText(annotated_frame, status_text, (15, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                last_annotated_frame = annotated_frame_rgb
                st_frame.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

            cap.release()
            progress_bar.progress(100)
            progress_text.caption("解析進行度: 100%（完了）")

            jst = timezone(timedelta(hours=9))
            now_str = datetime.now(jst).strftime("%Y-%m-%d %H:%M:%S")

            total_detected = safe_frames + danger_frames
            if total_detected > 0:
                safe_rate = (safe_frames / total_detected) * 100
                danger_rate = (danger_frames / total_detected) * 100
                st.session_state.analysis_result = {
                    "tool_type": tool_type,
                    "safe_rate": safe_rate,
                    "danger_rate": danger_rate,
                    "safe_frames": safe_frames,
                    "danger_frames": danger_frames,
                    "standby_frames": standby_frames,
                    "last_frame": last_annotated_frame
                }
                new_record = {
                    "診断日時": now_str,
                    "映像ファイル": video_name,
                    "対象機材": tool_type,
                    "安全フォーム率 (%)": round(safe_rate, 1),
                    "危険フォーム率 (%)": round(danger_rate, 1),
                    "有効作業コマ": total_detected,
                    "除外コマ": standby_frames
                }
            else:
                st.session_state.analysis_result = "no_tool_or_person"
                new_record = {
                    "診断日時": now_str,
                    "映像ファイル": video_name,
                    "対象機材": tool_type,
                    "安全フォーム率 (%)": 0.0,
                    "危険フォーム率 (%)": 0.0,
                    "有効作業コマ": 0,
                    "除外コマ": standby_frames
                }

            st.session_state.history.insert(0, new_record)
            save_history(st.session_state.history)

        res = st.session_state.analysis_result
        if res == "no_tool_or_person":
            st.warning("映像内から対象機材を保持した作業姿勢が検出されませんでした。")
        elif res is not None:
            if not start_btn and res["last_frame"] is not None:
                st.image(res["last_frame"], caption=f"最終解析フレーム（{res['tool_type']}）", channels="RGB", use_container_width=True)

            st.markdown(f"""
            <div class="kpi-grid">
                <div class="kpi-box">
                    <div class="kpi-label">安全フォーム率</div>
                    <div class="kpi-val-safe">{res['safe_rate']:.1f}<span style="font-size:0.95rem;"> %</span></div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">危険フォーム率</div>
                    <div class="kpi-val-danger">{res['danger_rate']:.1f}<span style="font-size:0.95rem;"> %</span></div>
                </div>
                <div class="kpi-box">
                    <div class="kpi-label">非作業除外コマ</div>
                    <div class="kpi-val-neutral">{res['standby_frames']}<span style="font-size:0.95rem;"> コマ</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(res["safe_rate"]))

            if res["danger_rate"] >= 30.0:
                st.error(f"評価所見: 作業中の危険姿勢が {res['danger_rate']:.1f}% 検出されました。作業フォームの改善が必要です。")
            elif res["danger_rate"] > 0:
                st.warning(f"評価所見: 一部区間（{res['danger_rate']:.1f}%）で危険姿勢が確認されました。")
            else:
                st.success("評価所見: 全作業区間を通じて安全なフォームが維持されています。")
        else:
            st.markdown("""
            <div style="text-align:center; padding: 5.5rem 1rem; color: #C2D1CA; background-color: #131917; border-radius: 10px; border: 1px dashed #34463F;">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="1.5" style="margin-bottom: 0.75rem;">
                    <rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>
                </svg>
                <div style="font-size: 1rem; font-weight: 700; color: #FFFFFF; margin-bottom: 0.35rem;">解析スタンバイ</div>
                <div style="font-size: 0.88rem; color: #B4C4BC;">左側のパネルから対象機材と映像データを指定し、「解析を実行する」を選択してください。</div>
            </div>
            """, unsafe_allow_html=True)

if len(st.session_state.history) > 0:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("""
        <div class="sec-title">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#74C69D" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>
            </svg>
            <span>診断履歴アーカイブ</span>
        </div>
        """, unsafe_allow_html=True)
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        csv_data = df_history.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="CSV形式で履歴をエクスポート",
            data=csv_data,
            file_name="forest_safety_history.csv",
            mime="text/csv"
        )
