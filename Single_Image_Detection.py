import streamlit as st
from pathlib import Path
import PIL
import pandas as pd
from collections import Counter
import cv2
import numpy as np
import settings
import os
import zipfile
from ultralytics import YOLO


def plot_boxes(image, boxes, class_names, selected_class=None):
    """
    Draws bounding boxes on the image. If selected_class is provided,
    only draws boxes of that class.
    """
    color_map = {
        0: (255, 220, 180), 1: (180, 255, 180), 2: (255, 180, 180),
        3: (180, 255, 255), 4: (255, 180, 255), 5: (220, 220, 220),
        6: (255, 240, 180), 7: (180, 255, 220), 8: (220, 180, 255),
        9: (180, 220, 255), 10: (200, 180, 255), 11: (255, 180, 200),
        12: (180, 255, 200), 13: (255, 255, 180), 14: (255, 180, 240),
        15: (180, 255, 255), 16: (255, 220, 180), 17: (180, 255, 240),
        18: (200, 255, 180), 19: (200, 200, 255), 20: (180, 200, 255),
        21: (255, 180, 240), 22: (255, 200, 220), 23: (220, 255, 180),
        24: (240, 255, 180), 25: (180, 220, 240), 26: (255, 220, 240),
        27: (240, 220, 180), 28: (240, 180, 220), 29: (180, 240, 255),
        30: (180, 220, 220), 31: (255, 180, 180), 32: (240, 200, 255),
        33: (200, 200, 220)
    }

    for box in boxes:
        cls = int(box.cls[0])
        class_name = class_names.get(cls, "Unknown")
        
        if selected_class and class_name != selected_class:
            continue

        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        label = ' '.join(class_name.split()[:2])
        color = color_map.get(cls, (255, 255, 255))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return image

def load_model(model_path):
    """
    Load YOLO model from specified path
    """
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

@st.cache_resource
def load_model_cached(model_path):
    return load_model(model_path)

# Create output directory
output_dir = "annotated_outputs"
os.makedirs(output_dir, exist_ok=True)

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Service Key Detection")

# Sidebar

with st.sidebar.expander("Additional Settings"):
    confidence = float(st.slider(
        "Select Model Confidence (%)", 25, 100, 40)) / 100

st.sidebar.header("Upload Image")
source_img = st.sidebar.file_uploader(
    "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

# Set model path
model_path = Path(settings.DETECTION_MODEL)

# Load Model
try:
    model = load_model_cached(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    st.stop()

# Initialize session state
if 'detections' not in st.session_state:
    st.session_state['detections'] = None
    st.session_state['cls_names'] = []
    st.session_state['image_np'] = None
    st.session_state['processed_image'] = None
    st.session_state['detection_summary'] = None

# Layout for Original and Detected Images
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Image")
    try:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            default_image = PIL.Image.open(default_image_path).convert("RGB")
            st.image(default_image, caption="Default Image", use_column_width=True)
        else:
            uploaded_image = PIL.Image.open(source_img).convert("RGB")
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    except Exception as ex:
        st.error("Error occurred while opening the image.")
        st.error(ex)

with col2:
    st.subheader("Detected Image")
    try:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path).convert("RGB")
            st.image(default_detected_image, caption='Detected Image', use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image, conf=0.15, iou = 0.2)
                boxes = res[0].boxes
                
                if not boxes:
                    st.warning("No objects detected.")
                    st.session_state['detections'] = None
                    st.session_state['cls_names'] = []
                    st.session_state['image_np'] = None
                    st.session_state['processed_image'] = None
                    st.session_state['detection_summary'] = None
                else:
                    image_np = np.array(uploaded_image)
                    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    
                    st.session_state['detections'] = boxes
                    st.session_state['cls_names'] = [model.names[int(box.cls[0])] for box in boxes]
                    st.session_state['image_np'] = image_np.copy()
                    
                    image_np_drawn = plot_boxes(image_np.copy(), boxes, model.names)
                    image_np_drawn = cv2.cvtColor(image_np_drawn, cv2.COLOR_BGR2RGB)
                    st.session_state['processed_image'] = image_np_drawn
                    
                    st.image(image_np_drawn, caption='Detected Image', use_column_width=True)
                    
                    # Create detection summary
                    cls_counts = Counter(st.session_state['cls_names'])
                    df = pd.DataFrame(cls_counts.items(), columns=['Class', 'Quantity'])
                    total_quantity = df['Quantity'].sum()
                    total_row = pd.DataFrame([{'Class': 'Total', 'Quantity': total_quantity}])
                    df = pd.concat([df, total_row], ignore_index=True)
                    st.session_state['detection_summary'] = df
                    
                    # Save files for download
                    if st.session_state['processed_image'] is not None:
                        output_image_path = os.path.join(output_dir, "annotated_image.png")
                        PIL.Image.fromarray(st.session_state['processed_image']).save(output_image_path)
                        
                    if st.session_state['detection_summary'] is not None:
                        csv_path = os.path.join(output_dir, "detection_summary.csv")
                        st.session_state['detection_summary'].to_csv(csv_path, index=False)
                        
                    # Create ZIP file
                    zip_path = os.path.join(output_dir, "detection_results.zip")
                    with zipfile.ZipFile(zip_path, 'w') as zipf:
                        if os.path.exists(output_image_path):
                            zipf.write(output_image_path, "annotated_image.png")
                        if os.path.exists(csv_path):
                            zipf.write(csv_path, "detection_summary.csv")
    except Exception as ex:
        st.error("Error occurred during object detection.")
        st.error(ex)

# Interactive Filtering (moved above Detection Summary)
if st.session_state['detections']:
    st.subheader("Filter Detections by Class")
    unique_classes = sorted(list(set(st.session_state['cls_names'])))
    selected_class = st.selectbox("Select a class to filter", ["All"] + unique_classes)

    image_np_filtered = st.session_state['image_np'].copy()
    boxes = st.session_state['detections']
    class_names = model.names

    if selected_class != "All":
        filtered_boxes = [box for box, cls in zip(boxes, st.session_state['cls_names']) if cls == selected_class]
    else:
        filtered_boxes = boxes

    if filtered_boxes:
        image_np_filtered = plot_boxes(image_np_filtered.copy(), filtered_boxes, class_names, 
                                     selected_class if selected_class != "All" else None)
        image_np_filtered = cv2.cvtColor(image_np_filtered, cv2.COLOR_BGR2RGB)
        st.image(image_np_filtered, caption=f'Detected Image - {"All Classes" if selected_class == "All" else selected_class}', 
                use_column_width=True)
    else:
        st.warning(f"No detections found for class: {selected_class}")

# Detection Summary
if st.session_state['detection_summary'] is not None:
    st.subheader("Detection Summary")
    # Wrap the table in a container with a max width
    with st.container():
        col1, col2 = st.columns([1, 2])  # Creates two columns with the first taking 1/3 of the space
        with col1:
            st.table(st.session_state['detection_summary'])
        
        # Move download button to align with the table
        zip_path = os.path.join(output_dir, "detection_results.zip")
        if os.path.exists(zip_path):
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="Download Results (Images & Summary)",
                    data=f,
                    file_name="detection_results.zip",
                    mime="application/zip"
                )

# Cleanup function
def cleanup_files():
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir, topdown=False):
            for file in files:
                try:
                    os.remove(os.path.join(root, file))
                except Exception as e:
                    st.error(f"Error removing file {file}: {e}")
            for dir in dirs:
                try:
                    os.rmdir(os.path.join(root, dir))
                except Exception as e:
                    st.error(f"Error removing directory {dir}: {e}")

# Register cleanup function to run when the script reruns
st.session_state['cleanup'] = cleanup_files