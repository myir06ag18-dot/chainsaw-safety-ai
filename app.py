import streamlit as st
import cv2
import tempfile
import os
import yt_dlp
from ultralytics import YOLO

st.title("農林安全作業 フォーム診断AI")

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
            st.error("YouTube側の制限により直接読み込めませんでした。動画をPCに保存してから「動画ファイルをアップロード」をご利用ください。")
            st.caption(f"詳細エラー: {e}")

if video_path is not None:
    if st.button("AI解析を開始"):
        st.info("解析を開始しました...")
        model = YOLO("yolo11n-pose.pt")
        cap = cv2.VideoCapture(video_path)
        st_frame = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            
            if tool_type == "チェンソー":
                cv2.putText(annotated_frame, "CHAINSAW MODE: CUTTING ZONE", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(annotated_frame, "BRUSHCUTTER MODE: KICKBACK WARNING", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, channels="RGB")
            
        cap.release()
        st.success("解析が完了しました！")
