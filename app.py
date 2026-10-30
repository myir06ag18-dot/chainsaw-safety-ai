import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import math
import os

st.title("チェンソー作業 フォーム安全度診断AI")

pose_model = YOLO('yolo11n-pose.pt')
detect_model = YOLO('yolo11n.pt')

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

youtube_url = st.text_input("YouTubeのURLを入力", placeholder="https://www.youtube.com/watch?v=...")
uploaded_file = st.file_uploader("または動画ファイルをアップロード", type=["mp4", "mov", "avi"])

if st.button("AI解析を開始"):
    video_path = None
    if youtube_url and youtube_url.strip() != "":
        download_path = "yt_download.mp4"
        if os.path.exists(download_path): os.remove(download_path)
        cmd = f'yt-dlp -f "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best" --merge-output-format mp4 "{youtube_url.strip()}" -o "{download_path}"'
        os.system(cmd)
        if os.path.exists(download_path): video_path = download_path
    elif uploaded_file is not None:
        video_path = "temp_uploaded_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    if not video_path:
        st.error("動画またはYouTubeのURLを指定してください。")
    else:
        st.info("解析を開始しました...")
        cap = cv2.VideoCapture(video_path)
        temp_output_path = "temp_output.mp4"
        final_output_path = "app_playable_output.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_w = 1280
        if w > max_w:
            h = int(h * (max_w / w))
            w = max_w

        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (w, h))

        total, pro, warning, danger = 0, 0, 0, 0
        frame_count = 0
        SKIP_FRAMES = 2
        cached_kpts = None
        cached_status = ("切断作業を探索中...", (255, 255, 0))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1
            if (w, h) != (frame.shape[1], frame.shape[0]):
                frame = cv2.resize(frame, (w, h))

            if frame_count % (SKIP_FRAMES + 1) == 1:
                total += 1
                pose_results = pose_model(frame, verbose=False, imgsz=480)
                det_results = detect_model(frame, verbose=False, imgsz=480)
                det_boxes = []
                for det in det_results:
                    if det.boxes is not None:
                        for box in det.boxes.xyxy.cpu().numpy(): det_boxes.append(box)

                found_cutting_worker = False
                for result in pose_results:
                    if result.keypoints is not None and len(result.keypoints.xy) > 0:
                        for kpts in result.keypoints.xy.cpu().numpy():
                            if len(kpts) < 15: continue
                            nose, r_shoulder, l_wrist, r_wrist, r_hip, r_knee = kpts[0], kpts[6], kpts[9], kpts[10], kpts[12], kpts[14]

                            is_holding_chainsaw = False
                            wrist_pos = None
                            for wrist in [r_wrist, l_wrist]:
                                if all(wrist):
                                    wx, wy = wrist[0], wrist[1]
                                    for box in det_boxes:
                                        if (box[0] - 50 <= wx <= box[2] + 50) and (box[1] - 50 <= wy <= box[3] + 50):
                                            is_holding_chainsaw = True
                                            wrist_pos = wrist
                                            break
                                if is_holding_chainsaw: break

                            is_actively_cutting = False
                            if is_holding_chainsaw and all(r_shoulder) and all(r_hip) and all(r_knee):
                                hip_angle = calculate_angle(r_shoulder, r_hip, r_knee)
                                is_bent_down = hip_angle < 160
                                is_hand_lowered = wrist_pos[1] >= (r_hip[1] - 30) if wrist_pos is not None else False
                                if is_bent_down and is_hand_lowered: is_actively_cutting = True

                            if is_actively_cutting:
                                found_cutting_worker = True
                                cached_kpts = kpts
                                if all(r_shoulder) and all(r_wrist) and all(nose):
                                    head_offset = abs(nose[0] - r_wrist[0])
                                    knee_angle = calculate_angle(r_shoulder, r_hip, r_knee)
                                    if head_offset < 25:
                                        cached_status = ("【危険】キックバック直撃ゾーン", (0, 0, 255))
                                        danger += 1
                                    elif knee_angle > 170:
                                        cached_status = ("【注意】やや姿勢が不安定", (0, 255, 255))
                                        warning += 1
                                    else:
                                        cached_status = ("【安全】プロの基本姿勢", (0, 255, 0))
                                        pro += 1
                                break

                if not found_cutting_worker:
                    cached_kpts = None
                    cached_status = ("切断作業を探索中...", (255, 255, 0))

            status_text, color = cached_status
            if cached_kpts is not None:
                kpts = cached_kpts
                for start_idx, end_idx in SKELETON_CONNECTIONS:
                    if start_idx < len(kpts) and end_idx < len(kpts):
                        pt1, pt2 = kpts[start_idx], kpts[end_idx]
                        if all(pt1) and all(pt2):
                            cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 3, cv2.LINE_AA)

                for kp in kpts:
                    if all(kp): cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (255, 255, 255), -1)

                if all(kpts[0]):
                    cv2.putText(frame, "CUTTING ACTION", (int(kpts[0][0]) - 50, int(kpts[0][1]) - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, status_text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3, cv2.LINE_AA)
            out.write(frame)

        cap.release()
        out.release()

        os.system(f'ffmpeg -y -i "{temp_output_path}" -vcodec libx264 -preset ultrafast -pix_fmt yuv420p "{final_output_path}"')

        pro_ratio = (pro / total * 100) if total > 0 else 0
        warn_ratio = (warning / total * 100) if total > 0 else 0
        danger_ratio = (danger / total * 100) if total > 0 else 0

        st.success("解析が完了しました！")
        st.video(final_output_path)
        st.text(f"■ 安全（プロ姿勢）: {pro_ratio:.1f}%\n■ 注意（不安定）  : {warn_ratio:.1f}%\n■ 危険（直撃線）  : {danger_ratio:.1f}%")
