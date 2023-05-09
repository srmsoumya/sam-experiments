import hashlib

import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torchvision
from segment_anything import sam_model_registry, SamPredictor


print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("CUDA is available:", torch.cuda.is_available())

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
checkpoint = "sam_vit_b_01ec64.pth"
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=checkpoint)
predictor = SamPredictor(sam)

if "saved_masks" not in st.session_state:
    st.session_state.saved_masks = []

def show_mask(mask, ax, color=None):
    if color is None:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array(color)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=30):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))


def string_to_color(text: str) -> tuple:
    """Convert a string to a color using its hash."""
    # Generate a 32-bit hash of the input text
    hash_value = int(hashlib.md5(text.encode('utf-8')).hexdigest()[:8], 16)

    # Convert the hash value to RGB values
    red = ((hash_value >> 16) & 0xff) / 255
    green = ((hash_value >> 8) & 0xff) / 255
    blue = (hash_value & 0xff) / 255
    opacity = 0.7

    return red, green, blue, opacity

def string_to_css_color(text: str) -> str:
    """Convert a string to a CSS color using its hash."""
    red, green, blue, opacity = string_to_color(text)

    # Convert the RGB values to a CSS color string
    css_color = f"rgba({red*255}, {green*255}, {blue*255}, {opacity})"

    return css_color

st.set_page_config(page_title="Label images with Segment Anything")

st.title("Label images with Segment Anything")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Define a Streamlit app that takes a comma-separated list of strings as input
input_str = st.text_input("Enter a comma-separated list of possible labels:")
if input_str:
    items = [s.strip() for s in input_str.split(",")]
    # Create the legend as a formatted Markdown string
    legend_str = ""
    for label in items:
        legend_str += f"{label}<span style='display:inline-block;width:12px;height:12px;background-color:{string_to_css_color(label)};margin-left:4px;'></span><br>"
    # Create a dropdown menu with the items and their assigned colors
    selected_item = st.sidebar.selectbox("Selected label:", items)
    # Display the legend in the sidebar using st.sidebar.markdown()
    st.sidebar.markdown(f"Label color codes:\n\n{legend_str}", unsafe_allow_html=True)


# Specify canvas parameters in application
prompt_tools = {"include dot": "green", "exclude dot": "red"}
# Create the legend as a formatted Markdown string
legend_str = ""
for label, color in prompt_tools.items():
    legend_str += f"<span style='color:{color};'>\u25CF</span> {label}<br>"
drawing_mode = st.sidebar.selectbox(
    "Selected promt tool:", prompt_tools.keys()
)
# Display the legend in the sidebar using st.sidebar.markdown()
st.sidebar.markdown(f"Prompt color codes:\n\n{legend_str}", unsafe_allow_html=True)

# stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'include dot':
    stroke_color = "green"
elif drawing_mode == 'exclude dot':
    stroke_color = "red"

realtime_update = st.sidebar.checkbox("Update in realtime", True)

    
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    report_image_width = 500
    report_image_height = (report_image_width/image.width)*image.height

    # Create a canvas component
    canvas_slot = st.empty()
    with canvas_slot.container():
        st.header("Canvas to label image")
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=3,
            stroke_color=stroke_color,
            background_color="#eee",
            background_image=image,
            update_streamlit=realtime_update,
            height=report_image_height,
            width=report_image_width,
            drawing_mode="point",
            point_display_radius=1,
            key="canvas",
        )


    accepted_masks = st.session_state.saved_masks
    if accepted_masks:
        st.header("Preview of accepted masks")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image)
        patches = []
        labels = set()
        for label, mask in accepted_masks:
            color = string_to_color(label)
            show_mask(mask, ax, color=color)
            if label not in labels:
                patches.append(mpatches.Patch(color=color, label=label))
                labels.add(label)
        ax.set_title(f"Preview of accepted masks", fontsize=10)
        ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        ax.axis('off')
        st.pyplot(fig)


    input_points = []
    input_labels = []

    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    predictor.set_image(cv_image)
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    objs = canvas_result.json_data["objects"] if canvas_result.json_data else []
    if objs:
        for obj in objs:
            if obj["type"] == "circle":
                input_points.append((obj["left"]*(image.width/report_image_width), obj["top"]*(image.height/report_image_height)))
                if obj["stroke"] == "green":
                    input_labels.append(1)
                else:
                    input_labels.append(0)

        input_points = np.array(input_points)
        input_labels = np.array(input_labels)

        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True,
        )

        masks_slot = st.empty()
        with masks_slot.container():
            st.header("Predicted masks for the image")
            columns = st.columns(3)
            for i, (mask, score) in enumerate(zip(masks, scores)):
                column = columns[i]
                with column:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    # Create the plot
                    ax.imshow(image)
                    show_mask(mask, ax, color=(30/255, 144/255, 255/255, 0.6))
                    show_points(input_points, input_labels, ax)
                    ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=10)
                    ax.axis('off')

                    # Display the plot in Streamlit using st.pyplot()
                    st.pyplot(fig)
                    if st.button(f"Accept {i}?"):
                        st.session_state.saved_masks.append((selected_item, mask))
                        masks_slot.empty()
                        canvas_slot.empty()
                        del st.session_state["canvas"]["raw"]["objects"]

                        st.experimental_rerun()
