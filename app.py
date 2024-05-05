import torch
import time
import warnings

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
from cellpose import models as cellpose_models

from src.preprocessing_utils import preprocess_image
from src.classify_caging import classify_image

import io
import zipfile

warnings.filterwarnings("ignore")
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


# def click_button():
#     st.session_state.clicked = True

@st.cache_resource()
def load_model():
    model_cp = cellpose_models.CellposeModel(
        gpu=True if torch.cuda.is_available() else False,
        pretrained_model="./src/cellpose_model/CP_myo",
    )
    return model_cp

@st.cache_data()
def process_images(uploaded_file, nuclei_channel_number, grooves_channel_number, _model_cp):
    st.write('Begin analysis of your images...')
    results = {}
    progress_text = "Processing file in progress... Please wait."
    my_bar = st.progress(0, text=progress_text)

    start_time = time.time()
    for idx, files in enumerate(uploaded_file):
        io_read = io.BytesIO(files.read())
        image = imread(io_read, plugin='tifffile')
        image_to_process = np.expand_dims(image[:, :, nuclei_channel_number], axis=0)
        if grooves_channel_number > -1:
            grooves_ch = np.expand_dims(image[:, :, grooves_channel_number], axis=0)
        else:
            grooves_ch = None
        p_image = preprocess_image(image_to_process, grooves_ch)

        # st.image(expand_channels(p_image, channel_number))
        with torch.inference_mode():
            pred_mask, _, _ = _model_cp.eval(p_image, channels=[0, 0], diameter=48.11, normalize=True,
                                            net_avg=False)
            pred_mask = np.expand_dims(pred_mask, axis=0)
        caged_or_not = classify_image(pred_mask)

        results[idx] = {'p_image': p_image, 'pred_mask': pred_mask, 'caged_or_not': caged_or_not}
        elapsed_time = time.time() - start_time
        my_bar.progress((idx + 1) / len(uploaded_file), text=f"Processed file(s) {idx + 1} / {len(uploaded_file)}, Elapsed time: {elapsed_time:.2f} seconds")
    return results


def main():
    model_cp = load_model()
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

    # if 'clicked' not in st.session_state:
    #     st.session_state.clicked = False

    if st.sidebar.button("Clear all"):
        st.session_state["file_uploader_key"] += 1
        # st.session_state.clicked = False
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


    #CHANNEL SIDEBAR
    if uploaded_file:

        st.sidebar.title("Select channels")
        max_channel = 10
        # if uploaded_file:
        #     io_test = io.BytesIO(uploaded_file[0].read())
        #     image_test = imread(io_test, plugin='tifffile')
        #     max_channel = image_test.shape[2] - 1
        

        nuclei_channel_number = st.sidebar.number_input("Nuclei channel", min_value=0, max_value=max_channel, value=0, step=1)
        grooves_channel_number = st.sidebar.number_input("Grooves channel for rotation calculation (if image already horizontal, set to -1)", min_value=-1, max_value=max_channel, value=3, step=1)
        #st.sidebar.button("Analyze", on_click=click_button)

        #if st.session_state.clicked:
        if st.sidebar.button("Analyze"):

            results = process_images(uploaded_file, nuclei_channel_number, grooves_channel_number, model_cp)

            image_names = [img.name for img in uploaded_file]

            zip_data = io.BytesIO()
            with zipfile.ZipFile(zip_data, mode="w") as z:
                for idx in range(len(image_names)):
                    result = results[image_names.index(image_names[idx])]
                    folder_name = f"{image_names[idx]}/"
                    p_image_data = io.BytesIO()
                    imsave(p_image_data, result["p_image"], plugin='tifffile')
                    z.writestr(f"{folder_name}preprocessed_image.tif", p_image_data.getvalue())

                    pred_mask_data = io.BytesIO()
                    imsave(pred_mask_data, result["pred_mask"], plugin='tifffile')
                    z.writestr(f"{folder_name}segmentation_image.tif", pred_mask_data.getvalue())

                    caged_or_not_data = io.BytesIO()
                    caged_or_not_image = np.uint8(result["caged_or_not"])
                    imsave(caged_or_not_data, caged_or_not_image, plugin='tifffile')
                    z.writestr(f"{folder_name}caging_image.tif", caged_or_not_data.getvalue())

                # Set the position to the beginning of the stream
                zip_data.seek(0)

                # Provide the download link to the zip file
                st.download_button(
                    label="Download Zip Outputs",
                    data=zip_data,
                    file_name="results.zip",
                    mime="application/zip",
                    type="primary",
                )

            view_images = st.selectbox("#### Select an image to view:", image_names)
            selected_index = image_names.index(view_images)

            dict_entry = results[selected_index]
            processed_image = dict_entry["p_image"].transpose(1,2,0)

            st.write("##### Preprocessed image")
            st.image(expand_channels(processed_image, nuclei_channel_number))

            st.write("##### Nuclei segmentation")
            fig1, ax1 = plt.subplots()
            cmap_v = sns.color_palette("viridis", 24, as_cmap=True)
            cmap_v1 = cmap_v.copy()
            cmap_v1.colors[0] = [1, 1, 1, 1]
            ax1.imshow(dict_entry["pred_mask"][0], cmap=cmap_v1)
            ax1.axis('off')
            st.pyplot(fig1)

            st.write("##### Caging analysis")
            fig2, ax2 = plt.subplots()
            cmap_2 = sns.color_palette("hls", 3, as_cmap=True)
            cmap_v2 = cmap_2.copy()
            cmap_v2.colors[0] = [1, 1, 1, 1]
            ax2.imshow(dict_entry["caged_or_not"][0], cmap=cmap_v2)
            ax2.axis('off')
            st.pyplot(fig2)




if __name__ == '__main__':
    main()

