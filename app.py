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
    page_title="ForestSafe AI | 農林安全作業フォーム診断",
    page_icon="🌲",
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

# おしゃれなフォレスト地形図パターン背景 ＋ ガラス風カードデザイン
st.markdown("""
<style>
    /* 背景：森林グラデーション ＋ 山の等高線（トポグラフィック）パターン */
    .stApp {
        background-color: #e6f0ec;
        background-image: 
            radial-gradient(at 15% 15%, rgba(116, 198, 157, 0.25) 0px, transparent 50%),
            radial-gradient(at 85% 80%, rgba(45, 106, 79, 0.20) 0px, transparent 50%),
            url("data:image/svg+xml,%3Csvg width='120' height='120' viewBox='0 0 120 120' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M0 20 Q 30 5, 60 20 T 120 20 M0 50 Q 30 35, 60 50 T 120 50 M0 80 Q 30 65, 60 80 T 120 80 M0 110 Q 30 95, 60 110 T 120 110' fill='none' stroke='%232d6a4f' stroke-opacity='0.08' stroke-width='1.5'/%3E%3C/svg%3E");
    }

    /* ヘッダーの透明化 */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* サイドバー：すりガラス風デザイン */
    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.78) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(45, 106, 79, 0.15);
    }

    /* メインのヒーローバナー：木漏れ日をイメージした深緑グラデーション */
    .hero-banner {
        background: linear-gradient(120deg, #132a13 0%, #1b4332 45%, #2d6a4f 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px rgba(27, 67, 50, 0.22);
        border: 1px solid rgba(255, 255, 255, 0.15);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: "🌲";
        position: absolute;
        right: 25px;
        bottom: -15px;
        font-size: 7rem;
        opacity: 0.12;
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: 0.04em;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #d8f3dc;
        margin-top: 0.5rem;
    }

    /* コンテナ（border=True）を半透明のガラス風カードにする */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(10px);
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.9) !important;
        box-shadow: 0 6px 20px rgba(27, 67, 50, 0.07) !important;
        padding: 0.5rem !important;
    }

    /* メインボタンの立体グラデーション */
    div.stButton > button {
        background: linear-gradient(90deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 6px 15px rgba(45, 106, 79, 0.3);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(45, 106, 79, 0.45);
        color: #ffffff;
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
    st.markdown("## 🌲 ForestSafe AI")
    st.caption("YOLO11 骨格推定による農林作業安全診断システム")
    st.divider()
    st.markdown("### 📋 安全判定の基準")
    st.info("**🪚 チェンソーモード**\n\nチェンソー保持中、手首が肩より高い位置に上がると「危険」と判定します。")
    st.warning("**🌿 刈払機モード**\n\n刈払機保持中、右手首が腰より大きく右側へ振り抜かれると「危険」と判定します。")
    st.success("**⚡ 高速保持フィルタ搭載**\n\n骨格座標の構え方から作業中のみを瞬時に抽出し、非作業シーンを除外します。")
    st.divider()
    
    if st.session_state.analysis_result is not None or len(st.session_state.history) > 0:
        if st.button("🗑️ 保存された履歴をすべて削除"):
            st.session_state.analysis_result = None
            st.session_state.history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌲 農林安全作業 フォーム診断AI</div>
    <div class="hero-subtitle">道具保持フィルタ＆AI骨格検出で「作業中の姿勢」だけを自動抽出し、安全率を定量評価します</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.3], gap="large")

video_path = None
video_name = "未指定"

with col_left:
    with st.container(border=True):
        st.subheader("⚙️ 1. 診断設定")
        tool_type = st.radio(
            "使用する作業ツールを選択",
            ["チェンソー", "刈払機（草刈機）"],
            horizontal=True
        )

    with st.container(border=True):
        st.subheader("🎥 2. 動画の読み込み")
        tab1, tab2 = st.tabs(["📁 動画ファイルをアップロード", "🔗 YouTubeリンク"])

        with tab1:
            uploaded_file = st.file_uploader("MP4 / MOV / AVI 形式に対応", type=["mp4", "mov", "avi"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                video_name = uploaded_file.name
                st.success("✅ 動画のセットが完了しました")

        with tab2:
            youtube_url = st.text_input("YouTubeのURLを貼り付け", placeholder="https://www.youtube.com/watch?v=...")
            if youtube_url:
                clean_url = youtube_url.split("?si=")[0]
                with st.spinner("YouTube動画を準備中..."):
                    try:
                        temp_dir = tempfile.mkdtemp()
                        output_path = os.path.join(temp_dir, "youtube_video.mp4")
                        ydl_opts = {
                            'format': 'best[ext=mp4]/best',
                            'outtmpl': output_path,
                            'quiet': True,
                            'noplaylist': True,
                            'geo_bypass': True,
                            'nocheckcertificate': True,
                            'extractor_args': {
                                'youtube': {
                                    'player_client': ['tv', 'ios', 'mweb']
                                }
                            }
                        }
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([clean_url])
                        video_path = output_path
                        video_name = clean_url
                        st.success("✅ YouTube動画の読み込み完了")
                    except Exception as e:
                        st.error("YouTube側の制限により直接読み込めませんでした。「動画ファイルをアップロード」タブをご利用ください。")

    if video_path is not None:
        start_btn = st.button("🚀 AIフォーム診断を開始する")
    else:
        start_btn = False
        st.info("👈 動画をアップロードすると診断ボタンが表示されます")

with col_right:
    with st.container(border=True):
        st.subheader("🖥️ 3. AI解析モニター & 診断レポート")

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
                    progress_text.markdown(f"**⏳ AI骨格解析を実行中... {percent}% 完了**")

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
                                status_text = "DANGER! (HIGH / KICKBACK)"
                                color = (0, 0, 255)
                            else:
                                safe_frames += 1
                                status_text = "SAFE FORM"
                                color = (0, 255, 0)
                        else:
                            standby_frames += 1
                            status_text = "NO TOOL (STANDBY)"
                            color = (180, 180, 180)

                        cv2.putText(annotated_frame, status_text, (15, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                last_annotated_frame = annotated_frame_rgb
                st_frame.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

            cap.release()
            progress_bar.progress(100)
            progress_text.markdown("**✅ AI解析 100% 完了！**")

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
                    "動画名": video_name,
                    "作業ツール": tool_type,
                    "安全フォーム率 (%)": round(safe_rate, 1),
                    "危険フォーム率 (%)": round(danger_rate, 1),
                    "作業コマ数": total_detected,
                    "非作業コマ数": standby_frames
                }
            else:
                st.session_state.analysis_result = "no_tool_or_person"
                new_record = {
                    "診断日時": now_str,
                    "動画名": video_name,
                    "作業ツール": tool_type,
                    "安全フォーム率 (%)": 0.0,
                    "危険フォーム率 (%)": 0.0,
                    "作業コマ数": 0,
                    "非作業コマ数": standby_frames
                }

            st.session_state.history.insert(0, new_record)
            save_history(st.session_state.history)

        res = st.session_state.analysis_result
        if res == "no_tool_or_person":
            st.warning("動画内からチェンソー・刈払機を構えている作業姿勢を検出できませんでした。")
        elif res is not None:
            if not start_btn and res["last_frame"] is not None:
                st.image(res["last_frame"], caption=f"最終解析フレーム（{res['tool_type']}モード）", channels="RGB", use_container_width=True)

            st.divider()
            st.markdown(f"### 📊 フォーム安全診断レポート（{res['tool_type']}）")
            m1, m2, m3 = st.columns(3)
            m1.metric(label="🟢 安全フォーム率", value=f"{res['safe_rate']:.1f} %")
            m2.metric(label="🔴 危険フォーム率", value=f"{res['danger_rate']:.1f} %")
            m3.metric(label="⚪ 非作業（除外）コマ", value=f"{res['standby_frames']} コマ")

            st.progress(int(res["safe_rate"]))

            if res["danger_rate"] >= 30.0:
                st.error(f"⚠️ 作業中の危険姿勢が **{res['danger_rate']:.1f}%** 検出されました。作業姿勢の改善が必要です。")
            elif res["danger_rate"] > 0:
                st.warning(f"⚡ 作業中の一部（**{res['danger_rate']:.1f}%**）で危険な姿勢が見られました。注意して作業してください。")
            else:
                if start_btn:
                    st.balloons()
                st.success("✨ 素晴らしいフォームです！作業中を通して安全な姿勢が維持されています。")
        else:
            st.markdown("""
            <div style="text-align:center; padding: 4.5rem 1rem; color: #495057;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🌲🎥</div>
                <h3 style="color: #1b4332; margin-bottom: 0.5rem;">AI解析スタンバイ中</h3>
                <p style="font-size: 0.95rem;">左側のパネルから作業ツールと動画を選択し、<br>「AIフォーム診断を開始する」ボタンを押してください。</p>
            </div>
            """, unsafe_allow_html=True)

if len(st.session_state.history) > 0:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("📋 診断履歴一覧（新しい順・自動保存）")
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        csv_data = df_history.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="📥 診断履歴をCSVでダウンロード",
            data=csv_data,
            file_name="forest_safety_history.csv",
            mime="text/csv"
        )
