from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import streamlit as st
import cv2
from vidgear.gears import CamGear

# Membuat Streamlit UI
st.set_page_config(page_title="Ruang Belajar", layout="wide", initial_sidebar_state="expanded")

st.title('Ruang Belajar Deployment YoloV9')

# Membuat Sidebar
with st.sidebar:
    video = st.radio('Pilih data video', ['Webcam', 'YouTube'])
    link_youtube = ""
    if video == 'YouTube':
        link_youtube = st.text_input('Masukan Link Youtube', '')

# Process video
if st.button('Start'):
    if data_video == 'YouTube' and youtube_link:
        process_video(youtube_link, model)
    elif data_video == 'Webcam':
        process_video('Webcam', model)

# Load Model YOLOv9 
model = YOLO('yolov9c.pt')

# Membuat Function untuk plot bounding boxes pada frames
def plot_boxes(frame, model):
    results = model.predict(frame)
    annotasi = Annotator(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # Box coordinat
            c = box.cls      # ID Class
            annotasi.box_label(b, model.names[int(c)])
    return annotasi.result()

# Membuat Function untuk process dan display video
def process_video(data, model):
    if data == 'Webcam':
        camera = cv2.VideoCapture(0)  # kode webcam
    else:  # YouTube link
        camera = CamGear(data=data, stream_mode=True, logging=True).start()

    FRAME_WINDOW = st.empty()

    while True:
        if data == 'Webcam':
            ret, frame = camera.read()
            if not ret:
                break
        else:  # YouTube link
            frame = camera.read()
            if frame is None:
                break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = plot_boxes(frame, model)
        FRAME_WINDOW.image(frame)

    if data == 'Webcam':
        camera.release()
