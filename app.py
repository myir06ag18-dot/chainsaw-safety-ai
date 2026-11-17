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
            st.error("YouTube側の制限により直接読み込めませんでした。「動画ファイルをアップロード」をご利用ください。")
            st.caption(f"詳細エラー: {e}")

if video_path is not None:
    if st.button("AI解析を開始"):
        st.info("解析を開始しました...")
        model = YOLO("yolo11n-pose.pt")
        cap = cv2.VideoCapture(video_path)
        st_frame = st.empty()

        # 安全・危険フレームのカウント用変数
        safe_frames = 0
        danger_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            is_danger = False

            # 骨格座標が検出された場合の判定ロジック
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                kpts = results[0].keypoints.xy[0]
                # YOLO11のキーポイント番号: 5=左肩, 6=右肩, 9=左手首, 10=右手首, 11=左腰, 12=右腰
                if len(kpts) >= 13:
                    l_shoulder, r_shoulder = kpts[5], kpts[6]
                    l_wrist, r_wrist = kpts[9], kpts[10]
                    l_hip, r_hip = kpts[11], kpts[12]

                    if tool_type == "チェンソー":
                        # 手首が肩より高い位置にある場合は危険（画像座標は上が0）
                        if (l_wrist[1] > 0 and l_shoulder[1] > 0 and l_wrist[1] < l_shoulder[1]) or \
                           (r_wrist[1] > 0 and r_shoulder[1] > 0 and r_wrist[1] < r_shoulder[1]):
                            is_danger = True
                    else:
                        # 刈払機：右手首が右腰より大きく右側（画面左方向）へ振られている場合
                        if r_wrist[0] > 0 and r_hip[0] > 0 and r_wrist[0] < r_hip[0] - 40:
                            is_danger = True

                # カウントと画面上のリアルタイム表示
                if is_danger:
                    danger_frames += 1
                    status_text = "DANGER! (KICKBACK / HIGH POSITION)"
                    color = (0, 0, 255)  # 赤色
                else:
                    safe_frames += 1
                    status_text = "SAFE FORM"
                    color = (0, 255, 0)  # 緑色

                cv2.putText(annotated_frame, status_text, (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, channels="RGB")
            
        cap.release()
        st.success("解析が完了しました！")

        # 診断結果のパーセンテージ表示
        total_detected = safe_frames + danger_frames
        if total_detected > 0:
            safe_rate = (safe_frames / total_detected) * 100
            danger_rate = (danger_frames / total_detected) * 100

            st.subheader("📊 フォーム安全診断レポート")
            col1, col2 = st.columns(2)
            col1.metric(label="🟢 安全フォーム率", value=f"{safe_rate:.1f} %")
            col2.metric(label="🔴 危険フォーム率", value=f"{danger_rate:.1f} %")

            # 視覚的な割合バーの表示
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
