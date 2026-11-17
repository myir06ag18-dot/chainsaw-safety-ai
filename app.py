import streamlit as st
import cv2
import tempfile
import os
import yt_dlp
from ultralytics import YOLO

st.title("農林安全作業 フォーム診断AI")

@st.cache_resource
def load_model():
    return YOLO("yolo11n-pose.pt")

tool_type = st.radio("診断する作業ツールを選択してください:", ["チェンソー", "刈払機（草刈機）"])
input_method = st.radio("動画の入力方法を選択してください:", ["動画ファイルをアップロード", "YouTubeリンクを貼り付け"])

video_path = None

if input_method == "動画ファイルをアップロード":
    uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

else:
    youtube_url = st.text_input("YouTubeのURLを貼り付けてください（例: https://www.youtube.com/watch?v=...）")
    if youtube_url:
        clean_url = youtube_url.split("?si=")[0]
        st.info("YouTube動画を準備しています...")
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
            st.success("動画の読み込みが完了しました！")
        except Exception as e:
            st.error("YouTube側の制限により直接読み込めませんでした。「動画ファイルをアップロード」をご利用ください。")
            st.caption(f"詳細エラー: {e}")

if video_path is not None:
    if st.button("AI解析を開始"):
        model = load_model()
        cap = cv2.VideoCapture(video_path)
        
        # ① 解析中の進捗パーセント表示エリア
        progress_text = st.empty()
        progress_bar = st.progress(0)
        st_frame = st.empty()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        skip_frames = 5  # 5コマに1回解析して高速化

        safe_frames = 0
        danger_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # 進捗パーセントの計算と表示
            if total_frames > 0:
                percent = min(int((frame_count / total_frames) * 100), 100)
                progress_bar.progress(percent)
                progress_text.markdown(f"**⏳ AI解析中... {percent}% 完了**")

            h, w = frame.shape[:2]
            if w > 640:
                scale = 640 / w
                frame = cv2.resize(frame, (640, int(h * scale)))

            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            is_danger = False

            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                kpts = results[0].keypoints.xy[0]
                if len(kpts) >= 13:
                    l_shoulder, r_shoulder = kpts[5], kpts[6]
                    l_wrist, r_wrist = kpts[9], kpts[10]
                    l_hip, r_hip = kpts[11], kpts[12]

                    if tool_type == "チェンソー":
                        if (l_wrist[1] > 0 and l_shoulder[1] > 0 and l_wrist[1] < l_shoulder[1]) or \
                           (r_wrist[1] > 0 and r_shoulder[1] > 0 and r_wrist[1] < r_shoulder[1]):
                            is_danger = True
                    else:
                        if r_wrist[0] > 0 and r_hip[0] > 0 and r_wrist[0] < r_hip[0] - 30:
                            is_danger = True

                if is_danger:
                    danger_frames += 1
                    status_text = "DANGER! (HIGH / KICKBACK)"
                    color = (0, 0, 255)
                else:
                    safe_frames += 1
                    status_text = "SAFE FORM"
                    color = (0, 255, 0)

                cv2.putText(annotated_frame, status_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, channels="RGB")
            
        cap.release()
        progress_bar.progress(100)
        progress_text.markdown("**✅ AI解析 100% 完了！**")

        # ② 解析終了後の「何％安全か」レポート表示
        total_detected = safe_frames + danger_frames
        if total_detected > 0:
            safe_rate = (safe_frames / total_detected) * 100
            danger_rate = (danger_frames / total_detected) * 100

            st.subheader("📊 フォーム安全診断レポート")
            col1, col2 = st.columns(2)
            col1.metric(label="🟢 安全フォーム率", value=f"{safe_rate:.1f} %")
            col2.metric(label="🔴 危険フォーム率", value=f"{danger_rate:.1f} %")

            st.progress(int(safe_rate))

            if danger_rate >= 30.0:
                st.error(f"⚠️ 危険姿勢が全体の {danger_rate:.1f}% 検出されました。作業姿勢の見直しを推奨します。")
            elif danger_rate > 0:
                st.warning(f"⚡ 一部（{danger_rate:.1f}%）で危険な姿勢が見られました。注意して作業してください。")
            else:
                st.balloons()
                st.success("✨ 素晴らしいフォームです！全体を通して安全な姿勢が維持されています。")
        else:
            st.warning("動画内から作業者の骨格を十分に検出できませんでした。")
