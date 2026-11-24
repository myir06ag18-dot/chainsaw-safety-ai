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
    page_title="ForestSafe | 農林作業安全フォーム診断システム",
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

# 洗練された業務用Webシステム風CSS（Noto Sans JP導入・絵文字排除・ミニマル配色）
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&family=Noto+Sans+JP:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', 'Noto Sans JP', sans-serif;
        color: #1f2937;
    }

    /* 背景：落ち着いたスレート・オフホワイト */
    .stApp {
        background-color: #f4f6f5;
    }

    header[data-testid="stHeader"] {
        background: transparent !important;
    }

    /* サイドバー */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e5e7eb;
    }

    /* トップヘッダー（シンプルで重厚感のある業務システム風） */
    .app-header {
        background-color: #18362a;
        padding: 1.75rem 2.25rem;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 1.5rem;
        border-left: 6px solid #40916c;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    .app-header-tag {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #95d5b2;
        margin-bottom: 0.35rem;
    }
    .app-header-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.02em;
    }
    .app-header-desc {
        font-size: 0.92rem;
        color: #d1fae5;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* カードコンテナ */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03) !important;
        padding: 0.5rem !important;
    }

    /* 見出しのスタイル調整 */
    h3 {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: #111827 !important;
        letter-spacing: 0.01em;
    }

    /* プライマリボタン（フラットで洗練された質感） */
    div.stButton > button {
        background-color: #1e4636;
        color: #ffffff;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.25rem;
        font-size: 0.98rem;
        width: 100%;
        transition: background-color 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    div.stButton > button:hover {
        background-color: #2d6a4f;
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
    st.markdown("### ForestSafe System")
    st.caption("農林作業 姿勢安全評価プラットフォーム")
    st.divider()
    st.markdown("**判定基準仕様**")
    st.markdown("""
    - **チェンソー作業**  
      機材保持中、手首位置が肩ラインを超過した場合に危険姿勢と判定します。
    - **刈払機（草刈機）作業**  
      機材保持中、右手首が腰ラインより大きく右側へ振り抜かれた場合に危険姿勢と判定します。
    - **非作業区間の自動除外**  
      骨格座標から機材保持状態を判別し、移動・待機中のフレームは集計から除外します。
    """)
    st.divider()
    
    if st.session_state.analysis_result is not None or len(st.session_state.history) > 0:
        if st.button("診断履歴データを初期化"):
            st.session_state.analysis_result = None
            st.session_state.history = []
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()

st.markdown("""
<div class="app-header">
    <div class="app-header-tag">FORESTRY SAFETY MANAGEMENT</div>
    <div class="app-header-title">農林作業 安全フォーム診断システム</div>
    <div class="app-header-desc">骨格推定モデルと作業姿勢フィルタにより、稼働中の安全率を定量評価します</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.35], gap="large")

video_path = None
video_name = "未指定"

with col_left:
    with st.container(border=True):
        st.subheader("1. 対象機材の選択")
        tool_type = st.radio(
            "診断対象とする作業機材を選択してください",
            ["チェンソー", "刈払機（草刈機）"],
            horizontal=True
        )

    with st.container(border=True):
        st.subheader("2. 映像データの入力")
        tab1, tab2 = st.tabs(["動画ファイル", "YouTube URL"])

        with tab1:
            uploaded_file = st.file_uploader("対応形式: MP4 / MOV / AVI", type=["mp4", "mov", "avi"])
            if uploaded_file is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                video_path = tfile.name
                video_name = uploaded_file.name
                st.caption(f"読み込み完了: {uploaded_file.name}")

        with tab2:
            youtube_url = st.text_input("動画URLを入力", placeholder="https://www.youtube.com/watch?v=...")
            if youtube_url:
                clean_url = youtube_url.split("?si=")[0]
                with st.spinner("映像データを取得しています..."):
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
                        st.caption("映像データの取得が完了しました。")
                    except Exception as e:
                        st.error("URLからの直接取得に失敗しました。動画ファイルタブからアップロードしてください。")

    if video_path is not None:
        start_btn = st.button("解析を実行する")
    else:
        start_btn = False
        st.caption("映像データをセットすると解析を実行できます。")

with col_right:
    with st.container(border=True):
        st.subheader("3. 解析モニター・診断結果")

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
                    progress_text.caption(f"処理状況: {percent}% 完了")

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
            progress_text.caption("処理状況: 100% 完了")

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

            st.divider()
            st.markdown(f"**評価サマリー（{res['tool_type']}）**")
            m1, m2, m3 = st.columns(3)
            m1.metric(label="安全フォーム率", value=f"{res['safe_rate']:.1f} %")
            m2.metric(label="危険フォーム率", value=f"{res['danger_rate']:.1f} %")
            m3.metric(label="非作業除外コマ", value=f"{res['standby_frames']} コマ")

            st.progress(int(res["safe_rate"]))

            if res["danger_rate"] >= 30.0:
                st.error(f"評価結果: 作業中の危険姿勢が {res['danger_rate']:.1f}% 検出されました。作業姿勢の是正が推奨されます。")
            elif res["danger_rate"] > 0:
                st.warning(f"評価結果: 一部区間（{res['danger_rate']:.1f}%）で危険姿勢が検出されました。")
            else:
                st.success("評価結果: 全作業区間を通じて適正な作業姿勢が維持されています。")
        else:
            st.markdown("""
            <div style="text-align:center; padding: 5rem 1rem; color: #6b7280; background-color: #f9fafb; border-radius: 8px; border: 1px dashed #d1d5db;">
                <div style="font-size: 1rem; font-weight: 600; color: #374151; margin-bottom: 0.4rem;">解析待機中</div>
                <div style="font-size: 0.88rem;">左側のパネルから対象機材と映像データを指定し、「解析を実行する」を選択してください。</div>
            </div>
            """, unsafe_allow_html=True)

if len(st.session_state.history) > 0:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container(border=True):
        st.subheader("診断履歴ログ")
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True, hide_index=True)

        csv_data = df_history.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="CSV形式で履歴を出力",
            data=csv_data,
            file_name="forest_safety_history.csv",
            mime="text/csv"
        )
