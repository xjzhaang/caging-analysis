import torch


import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tempfile import NamedTemporaryFile
from skimage.io import imread
from cellpose import models as cellpose_models

from src.preprocessing_utils import preprocess_image
from src.classify_caging import classify_image


st.set_page_config(
    page_title="Caging Analyzer",
    page_icon="🔬",
)


def expand_channels(image, channel_index):
    # Create zeros channels
    zero_channels = np.zeros_like(image)

    # Insert zero channels based on channel index
    if channel_index == 0:
        expanded_image = np.concatenate((image, zero_channels, zero_channels), axis=2)
    elif channel_index == 1:
        expanded_image = np.concatenate((zero_channels, image, zero_channels), axis=2)
    elif channel_index == 2:
        expanded_image = np.concatenate((zero_channels, zero_channels, image), axis=2)
    else:
        raise ValueError("Invalid channel index")
    return expanded_image


def click_button():
    st.session_state.clicked = True


@st.cache_data()
def process_images(uploaded_file, channel_number):
    st.write('Begin analysis of your images...')
    results = {}
    model_cp = cellpose_models.CellposeModel(
        gpu=True if torch.cuda.is_available() else False,
        pretrained_model="./src/cellpose_model/CP_myo",
    )
    progress_text = "Processing file in progress... Please wait."
    my_bar = st.progress(0, text=progress_text)

    for idx, files in enumerate(uploaded_file):
        with NamedTemporaryFile("wb", suffix=".tif") as f:
            f.write(files.getvalue())
            image_to_process = np.expand_dims(imread(f.name)[:, :, channel_number], axis=0)
            p_image = preprocess_image(image_to_process)

            # st.image(expand_channels(p_image, channel_number))
            with torch.inference_mode():
                pred_mask, _, _ = model_cp.eval(p_image, channels=[0, 0], diameter=48.11, normalize=True,
                                                net_avg=False)
                pred_mask = np.expand_dims(pred_mask, axis=0)
            caged_or_not = classify_image(pred_mask)

        results[idx] = {'p_image': p_image, 'pred_mask': pred_mask, 'caged_or_not': caged_or_not}
        my_bar.progress((idx + 1) / len(uploaded_file), text=f"Processed file {idx + 1} / {len(uploaded_file)}")
    return results


def main():
    st.write("## Caging analyzer for microgroove images")

    #Upload SIDEBAR
    if "file_uploader_key" not in st.session_state:
        st.session_state["file_uploader_key"] = 0

    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []

    st.sidebar.title("Upload Images")
    uploaded_file = st.sidebar.file_uploader("uploade",type=["jpg", "jpeg", "png", "tif"],
                                             accept_multiple_files=True, key=st.session_state["file_uploader_key"],label_visibility="hidden")



    if uploaded_file:
        st.sidebar.write(f"{len(uploaded_file)} images uploaded")
        st.session_state["uploaded_files"] = uploaded_file

    if not uploaded_file:
        st.write("### :arrow_left: Upload an image on the left to begin!")

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    if st.sidebar.button("Clear all"):
        st.session_state["file_uploader_key"] += 1
        st.session_state.clicked = False
        st.cache_data.clear()
        st.rerun()


    #CHANNEL SIDEBAR
    if uploaded_file:

        st.sidebar.title("Channel to process")
        max_channel = 0
        if uploaded_file:
            with NamedTemporaryFile("wb", suffix=".tif") as f:
                f.write(uploaded_file[0].getvalue())
                image = imread(f.name)
                max_channel = image.shape[2] - 1

        channel_number = st.sidebar.number_input("Channel", min_value=0, max_value=max_channel, value=0, step=1)
        st.sidebar.button("Analyze", on_click=click_button)

        if st.session_state.clicked:

            results = process_images(uploaded_file, channel_number)
            image_names = [img.name for img in uploaded_file]
            view_images = st.selectbox("#### Select an image to view:", image_names)
            selected_index = image_names.index(view_images)
            dict_entry = results[selected_index]

            processed_image = dict_entry["p_image"].transpose(1,2,0)

            st.write("##### Preprocessed image")
            st.image(expand_channels(processed_image, channel_number))

            st.write("##### Nuclei segmentation")
            fig, ax = plt.subplots()
            cmap_v = sns.color_palette("viridis", 24, as_cmap=True)
            cmap_ = cmap_v.copy()
            cmap_.colors[0] = [1, 1, 1, 1]
            ax.imshow(dict_entry["pred_mask"][0], cmap=cmap_)
            ax.axis('off')
            st.pyplot(fig)

            st.write("##### Caging analysis")
            fig, ax = plt.subplots()
            ax.imshow(dict_entry["caged_or_not"][0], cmap=cmap_)
            ax.axis('off')
            st.pyplot(fig)


   #status_text.write(f"Finished preprocessing file {idx + 1} / {len(uploaded_file)}")

                    # fig, ax = plt.subplots()
                    # cmap_v = sns.color_palette("viridis", 24, as_cmap=True)
                    # cmap_ = cmap_v.copy()
                    # cmap_.colors[0] = [1, 1, 1, 1]
                    # ax.imshow(pred_mask, cmap=cmap_)
                    # ax.axis('off')
                    # st.pyplot(fig)



                    # fig, ax = plt.subplots()
                    # cmap_v = sns.color_palette("viridis", 24, as_cmap=True)
                    # cmap_ = cmap_v.copy()
                    # cmap_.colors[0] = [1, 1, 1, 1]
                    # ax.imshow(caged_or_not[0], cmap=cmap_)
                    # ax.axis('off')
                    # st.pyplot(fig)

if __name__ == '__main__':
    main()

