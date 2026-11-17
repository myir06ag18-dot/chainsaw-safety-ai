import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import yt_dlp
from ultralytics import YOLO

st.set_page_config(
    page_title="ForestSafe AI | 農林安全作業フォーム診断",
    page_icon="🌲",
    layout="wide"
)

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

st.markdown("""
<style>
    .stApp { background-color: #f4f7f5; }
    .hero-banner {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(27, 67, 50, 0.2);
    }
    .hero-title { font-size: 2.2rem; font-weight: 800; margin: 0; letter-spacing: 0.03em; }
    .hero-subtitle { font-size: 1.05rem; color: #d8f3dc; margin-top: 0.5rem; }
    .ui-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        border: 1px solid #e0e8e3;
        margin-bottom: 1rem;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #2d6a4f 0%, #40916c 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-size: 1.1rem;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 10px rgba(45, 106, 79, 0.25);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(45, 106, 79, 0.35);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return YOLO("yolo11n-pose.pt")

# 座標計算のみで高速に「道具を構えているか」を判定する関数
def is_holding_tool_fast(kpts):
    l_shoulder, r_shoulder = kpts[5], kpts[6]
    l_wrist, r_wrist = kpts[9], kpts[10]
    l_hip, r_hip = kpts[11], kpts[12]

    # 両手首が見えていない場合は非保持
    if l_wrist[0] <= 0 or r_wrist[0] <= 0:
        return False

    # 肩幅を基準にする
    shoulder_width = abs(l_shoulder[0] - r_shoulder[0])
    if shoulder_width < 10:
        return False

    # 両手首の距離チェック（両手でハンドルを握っている構えか）
    wrist_dist = np.hypot(l_wrist[0] - r_wrist[0], l_wrist[1] - r_wrist[1])
    if wrist_dist < shoulder_width * 0.25 or wrist_dist > shoulder_width * 3.0:
        return False

    # 腕を下にダランと垂らしている歩行状態を除外
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
    
    if st.session_state.analysis_result is not None:
        if st.button("🗑️ 診断結果をクリア"):
            st.session_state.analysis_result = None
            st.rerun()

st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌲 農林安全作業 フォーム診断AI</div>
    <div class="hero-subtitle">道具保持フィルタ＆AI骨格検出で「作業中の姿勢」だけを自動抽出し、安全率を定量評価します</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.3], gap="large")

video_path = None

with col_left:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.subheader("⚙️ 1. 診断設定")
    tool_type = st.radio(
        "使用する作業ツールを選択",
        ["チェンソー", "刈払機（草刈機）"],
        horizontal=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.subheader("🎥 2. 動画の読み込み")
    tab1, tab2 = st.tabs(["📁 動画ファイルをアップロード", "🔗 YouTubeリンク"])

    with tab1:
        uploaded_file = st.file_uploader("MP4 / MOV / AVI 形式に対応", type=["mp4", "mov", "avi"])
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            video_path = tfile.name
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
                    st.success("✅ YouTube動画の読み込み完了")
                except Exception as e:
                    st.error("YouTube側の制限により直接読み込めませんでした。「動画ファイルをアップロード」タブをご利用ください。")
    st.markdown("</div>", unsafe_allow_html=True)

    if video_path is not None:
        start_btn = st.button("🚀 AIフォーム診断を開始する")
    else:
        start_btn = False
        st.info("👈 動画をアップロードすると診断ボタンが表示されます")

with col_right:
    st.markdown('<div class="ui-card">', unsafe_allow_html=True)
    st.subheader("🖥️ 3. AI解析モニター & 診断レポート")

    if start_btn and video_path is not None:
        model = load_model()
        cap = cv2.VideoCapture(video_path)

        progress_text = st.empty()
        progress_bar = st.progress(0)
        st_frame = st.empty()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        skip_frames = 8  # 間引き幅を8にしてさらに高速化

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
            if w > 480:  # 解像度を480pxにしてAI推論を高速化
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
        else:
            st.session_state.analysis_result = "no_tool_or_person"

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
        <div style="text-align:center; padding: 4rem 1rem; color: #6c757d;">
            <h3>🎬 解析待機中</h3>
            <p>左側のパネルから作業ツールと動画を選択し、<br>「AIフォーム診断を開始する」ボタンを押してください。</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
