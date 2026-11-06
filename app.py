import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO

# 画面のタイトル設定
st.title("農林安全作業 フォーム診断AI")

# 1. 作業ツールの選択
tool_type = st.radio("診断する作業ツールを選択してください:", ["チェンソー", "刈払機（草刈機）"])

# 動画のアップロード機能
uploaded_file = st.file_uploader("動画ファイルをアップロード", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # 一時ファイルとして保存
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    if st.button("AI解析を開始"):
        st.info("解析を開始しました...")
        
        # YOLO11 姿勢検出モデルの読み込み
        model = YOLO("yolo11n-pose.pt")
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # AI解析の実行
            results = model(frame)
            annotated_frame = results[0].plot()
            
            # ツールごとの判定メッセージを追加
            if tool_type == "チェンソー":
                cv2.putText(annotated_frame, "CHAINSAW MODE: CUTTING ZONE", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # 刈払機モード（キックバック危険エリアの警告表示）
                cv2.putText(annotated_frame, "BRUSHCUTTER MODE: KICKBACK WARNING", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # 画面表示用の変換
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, channels="RGB")
            
        cap.release()
        st.success("解析が完了しました！")
