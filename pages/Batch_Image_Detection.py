import streamlit as st
from pathlib import Path
import PIL
import pandas as pd
from collections import Counter
import zipfile
import os
import logging
import cv2
import numpy as np
import settings
import helper

# Initialize all session state variables at the start
if 'detections' not in st.session_state:
    st.session_state['detections'] = []

if 'detection_summary' not in st.session_state:
    st.session_state['detection_summary'] = pd.DataFrame(columns=['Class Label', 'Class Number', 'Quantity'])

if 'processed_files' not in st.session_state:
    st.session_state['processed_files'] = set()

if 'zip_path' not in st.session_state:
    st.session_state['zip_path'] = None



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
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        label = str(cls)
        color = color_map.get(cls, (255, 255, 255))
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return image

@st.cache_resource
def load_model_cached(model_path):
    return helper.load_model(model_path)

st.set_page_config(
    page_title="Multiple Image Object Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center;'>Multiple Image Service Key Detection</h1>", unsafe_allow_html=True)

# Create container for settings
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        confidence = float(st.slider(
            "Select Model Confidence (%)", 25, 100, 40)) / 100
        
        uploaded_images = st.file_uploader(
            "Choose images...", type=("jpg", "jpeg", "png", 'bmp', 'webp'), 
            accept_multiple_files=True)

# Load model
model_path = Path(settings.DETECTION_MODEL)

try:
    model = load_model_cached(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)
    st.stop()

# Create output directory
output_dir = "annotated_outputs"
os.makedirs(output_dir, exist_ok=True)
# Cleanup temporary files
def cleanup_files():
    """Clean up all files in the output directory"""
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logging.error(f"Error removing file {file_path}: {e}")


# Detection button and processing
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button('Detect Objects') and uploaded_images:
        # Clear previous detection summary when processing new images
        st.session_state['detection_summary'] = pd.DataFrame(columns=['Class Label', 'Class Number', 'Quantity'])
        
        # Clean up previous files
        cleanup_files()
        
        # Keep track of files created in this batch
        current_batch_files = []
        
        progress_bar = st.progress(0)
        for idx, uploaded_image in enumerate(uploaded_images):
            try:
                image = PIL.Image.open(uploaded_image).convert("RGB")
                image_np = np.array(image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                res = model.predict(image, conf=0.15,iou=0.2)
                boxes = res[0].boxes

                if boxes:
                    st.session_state['detections'] = boxes
                    image_np_drawn = plot_boxes(image_np.copy(), boxes, model.names)
                    image_np_drawn = cv2.cvtColor(image_np_drawn, cv2.COLOR_BGR2RGB)
                    
                    # Save annotated image
                    output_image_path = os.path.join(output_dir, f"{uploaded_image.name.split('.')[0]}_annotated.png")
                    PIL.Image.fromarray(image_np_drawn).save(output_image_path)
                    current_batch_files.append(output_image_path)

                    cls_counts = Counter([int(box.cls[0]) for box in boxes])
                    cls_data = [
                        {"Class Label": model.names[cls], "Class Number": cls, "Quantity": count}
                        for cls, count in cls_counts.items()
                    ]
                    
                    total_quantity = sum([item['Quantity'] for item in cls_data])
                    cls_data.append({"Class Label": "Total", "Class Number": "", "Quantity": total_quantity})

                    df = pd.DataFrame(cls_data)
                    csv_path = os.path.join(output_dir, f"{uploaded_image.name.split('.')[0]}_detection_summary.csv")
                    df.to_csv(csv_path, index=False)
                    current_batch_files.append(csv_path)

                    st.session_state['detection_summary'] = pd.concat(
                        [st.session_state['detection_summary'], df], 
                        ignore_index=True
                    )
                
                # Update progress bar
                progress_bar.progress((idx + 1) / len(uploaded_images))

            except Exception as ex:
                logging.error(f"Error processing image {uploaded_image.name}: {ex}")
                st.error(f"Error processing {uploaded_image.name}: {str(ex)}")

        # Create ZIP file only with current batch files
        if current_batch_files:
            zip_filename = "annotated_images_and_summaries.zip"
            st.session_state['zip_path'] = os.path.join(output_dir, zip_filename)
            with zipfile.ZipFile(st.session_state['zip_path'], 'w') as zipf:
                for file_path in current_batch_files:
                    # Get just the filename without the directory path
                    arcname = os.path.basename(file_path)
                    zipf.write(file_path, arcname=arcname)
        
        st.success("Processing complete!")

# Price calculation section
if isinstance(st.session_state['detection_summary'], pd.DataFrame) and not st.session_state['detection_summary'].empty:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.subheader("Price Calculator")
        
        try:
            # Filter out the "Total" rows for the dropdown
            filtered_df = st.session_state['detection_summary'][
                st.session_state['detection_summary']['Class Label'] != "Total"
            ]
            
            if not filtered_df.empty:
                # Add "All" option to class labels
                class_labels = ["All"] + list(filtered_df['Class Label'].unique())
                
                selected_label = st.selectbox(
                    "Select a class label to calculate price",
                    class_labels
                )

                if selected_label == "All":
                    # Show individual price inputs for each class
                    st.write("Enter unit prices for each class:")
                    
                    class_prices = {}
                    total_price = 0
                    
                    # Create a table for the results
                    results_data = []
                    
                    for label in filtered_df['Class Label'].unique():
                        col_price, col_count = st.columns([2, 1])
                        with col_price:
                            unit_price = st.number_input(
                                f"Unit price for {label}:",
                                min_value=0.0,
                                value=0.0,
                                step=0.01,
                                key=f"price_{label}"
                            )
                        with col_count:
                            count = filtered_df[filtered_df['Class Label'] == label]['Quantity'].sum()
                            st.write(f"Count: {count}")
                        
                        class_prices[label] = unit_price
                        subtotal = count * unit_price
                        total_price += subtotal
                        
                        results_data.append({
                            "Class": label,
                            "Count": count,
                            "Unit Price": f"${unit_price:.2f}",
                            "Subtotal": f"${subtotal:.2f}"
                        })
                    
                    if st.button("Calculate Total Price"):
                        # Display results in a formatted table
                        st.write("### Price Breakdown")
                        results_df = pd.DataFrame(results_data)
                        st.table(results_df)
                        
                        # Display grand total
                        st.markdown(f"### Grand Total: **${total_price:.2f}**")
                
                else:
                    # Single class price calculation
                    selected_class_count = filtered_df[
                        filtered_df['Class Label'] == selected_label
                    ]['Quantity'].sum()
                    
                    st.write(f"Total count for {selected_label}: {selected_class_count}")

                    unit_price = st.number_input(
                        "Enter unit price:",
                        min_value=0.0,
                        value=0.0,
                        step=0.01,
                        key="single_price"
                    )

                    if st.button("Calculate Total Price"):
                        total_price = selected_class_count * unit_price
                        st.write(f"Total price for {selected_label}: ${total_price:.2f}")
            
            else:
                st.warning("No detection data available for price calculation. Please run detection first.")

        except Exception as e:
            st.error(f"An error occurred during price calculation: {str(e)}")
            logging.error(f"Price calculation error: {str(e)}")

        # Download section
        if st.session_state['zip_path'] and os.path.exists(st.session_state['zip_path']):
            with open(st.session_state['zip_path'], "rb") as f:
                st.download_button(
                    label="Download Annotated Images and CSVs",
                    data=f,
                    file_name="annotated_images_and_summaries.zip",
                    mime="application/zip"
                )

# Register cleanup function to run when the script reruns
st.session_state['cleanup'] = cleanup_files