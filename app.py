from io import StringIO
from pathlib import Path
import streamlit as st
import time
from detect import *
import os
import sys
import argparse
from PIL import Image
import cv2
import time

# Set up the pathImg variables in a higher scope
import os

# Define the paths for original and annotated images
pathImg_original = os.path.join("dataset", "val", "images")  # Path for the original image without annotations
pathImg_annotated = os.path.join("dataset", "train", "images")  # Path for the annotated image after detection


# st.set_page_config(layout = "wide")
st.set_page_config(page_title="Yolo V5 Waste Segmentation Model", page_icon="🤖")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 340px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 340px;
        margin-left: -340px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

#################### Title #####################################################
st.markdown("<h3 style='text-align: center; color: red; font-family: font of choice, fallback font no1, sans-serif;'>Capstone Project</h3>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-family: font of choice, fallback font no1, sans-serif;'>Yolo V8-Waste Detection Model</h2>", unsafe_allow_html=True)
st.markdown('#')  # inserts empty space

# --------------------------------------------------------------------------------

DEMO_PIC = os.path.join('data', 'images', 'waste.jpg')

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


def remove_annotations(image_path):
    '''
        Removes annotations from an annotated image.
    '''
    image = cv2.imread(image_path)
    # Remove annotations (you need to implement this part based on your annotation format)
    # For example, if annotations are bounding boxes, you would remove the bounding boxes from the image.
    return image


def is_annotated_image(image_path):
    '''
        Placeholder function to check if an image is annotated.
        Replace this function with your actual logic to determine if an image is annotated.
    '''
    return False  # Placeholder logic, replace with your actual annotation checking logic


# ---------------------------Main Function for Execution--------------------------

def main():
    global pathImg_annotated  # Define pathImg_annotated as global here

    source_index = 0  # Set default source index for image upload

    cocoClassesLst = ["Cig_bud", "cig_pack", "Disposable", "Garbage", "Metal", "Plastic", "Plastic_Bag",
                      "Plastic_Container", "Plastic_bottle", "Plastic_wrapper", "Tetrapack", "Thermocol", "Tin_Box", "Can", "All classes"]

    classes_index = st.sidebar.multiselect("Select Classes", range(
        len(cocoClassesLst)), format_func=lambda x: cocoClassesLst[x])

    isAllinList = 80 in classes_index
    if isAllinList == True:
        classes_index = classes_index.clear()

    print("Selected Classes: ", classes_index)

    #################### Parameters to setup ########################################
    deviceLst = ['cpu', '0', '1', '2', '3']
    DEVICES = st.sidebar.selectbox(
        "Select Devices", deviceLst, index=0)
    print("Devices: ", DEVICES)
    MIN_SCORE_THRES = st.sidebar.slider(
        'Min Confidence Score Threshold', min_value=0.0, max_value=1.0, value=0.4)
    #################### /Parameters to setup ########################################

    weights = os.path.join("weights", "best.pt")

    if source_index == 0:

        uploaded_file = st.sidebar.file_uploader(
            "Upload Image", type=['png', 'jpeg', 'jpg'])

        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Resource Loading...'):
                st.sidebar.text("Uploaded Pic")
                st.sidebar.image(uploaded_file)
                # Save the uploaded image to the pathImg_original folder
                temp_image_path = pathImg_original + "/" + uploaded_file.name
                print(temp_image_path)
                with open(temp_image_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())

        elif uploaded_file is None:
            is_valid = True
            st.sidebar.text("DEMO Pic")
            st.sidebar.image(DEMO_PIC)
            data_source = DEMO_PIC

        else:
            is_valid = False

    if is_valid:
        print('valid')
        if st.button('Detect'):
            data_source = pathImg_annotated + "/" + uploaded_file.name
            print(data_source)

            if classes_index:
                with st.spinner(text='Inferencing, Please Wait.....'):
                    run(weights=weights,
                        source=data_source,
                        conf_thres=MIN_SCORE_THRES,
                        device=DEVICES,
                        save_txt=True,
                        save_conf=True,
                        classes=classes_index,
                        nosave=False,
                        )

            else:
                with st.spinner(text='Inferencing, Please Wait.....'):
                    run(weights=weights,
                        source=data_source,
                        conf_thres=MIN_SCORE_THRES,
                        device=DEVICES,
                        save_txt=True,
                        save_conf=True,
                        nosave=False,
                        )

            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith(".png"):
                            pathImg_annotated = os.path.join(get_detection_folder(), img)
                            st.image(pathImg_annotated)

                    st.markdown("### Output")
                    st.write("Path of Annotated Images: ", pathImg_annotated)
                    st.write("Path of TXT File: ",
                             os.path.join(get_detection_folder(), 'labels'))
                    st.balloons()

# --------------------MAIN FUNCTION CODE------------------------
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
# ------------------------------------------------------------------
