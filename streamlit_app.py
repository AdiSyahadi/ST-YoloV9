from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2

# Custom transformer class for YOLOv9 object detection
class YOLOv9Transformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model.predict(frame)
        annotator = Annotator(frame)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # Box coordinates
                c = box.cls      # Class ID
                annotator.box_label(b, self.model.names[int(c)])
        return annotator.result()

# Load the YOLOv9 model
model = YOLO('yolov9c.pt')

# Streamlit UI setup
st.set_page_config(page_title="Ruang Belajar", layout="wide", initial_sidebar_state="expanded")
st.title('Ruang Belajar Deployment YoloV9')

# Sidebar
with st.sidebar:
    video_source = st.radio('Pilih video source', ['Local', 'YouTube'])
    youtube_link = ""
    if video_source == 'YouTube':
        youtube_link = st.text_input('YouTube video link', '')
    elif video_source == 'Local':
        webrtc_streamer(
            key="example",
            video_transformer_factory=lambda: YOLOv9Transformer(model),
            async_transform=True,
        )

# Process video
if st.button('Start'):
    if video_source == 'YouTube' and youtube_link:
        st.write("Streaming dari YouTube saat ini tidak didukung dalam contoh ini.")
    elif video_source == 'Local':
        webrtc_streamer(
            key="example",
            video_transformer_factory=lambda: YOLOv9Transformer(model),
            async_transform=True,
        )

# Note on closing resources
st.sidebar.markdown("### Note")
st.sidebar.markdown("Tutup tab browser lain yang tidak anda gunakan untuk mendapat hasil webcam yang baik saat menggunakan sumber video lokal.")
