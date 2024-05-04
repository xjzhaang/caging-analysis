import numpy as np
import pandas as pd
from skimage.measure import regionprops
from tqdm import tqdm


def classify_image(image):
    df_all = create_object_labels(image)
    final_label_image = find_caged_nucleus(df_all, image)
    channel_0_2_normalized = final_label_image.astype(np.uint8)
    return channel_0_2_normalized


def is_near_border(x, y, distance, image_shape):
    """
    Check if a point (x, y) is near the image border.

    Parameters:
    - x (int): x-coordinate of the point.
    - y (int): y-coordinate of the point.
    - distance (int): Minimum distance from the border to consider the point not near the border.
    - image_shape (tuple): Shape of the image (height, width).

    Returns:
    - bool: True if the point is near the border, False otherwise.
    """
    border_distance_x = min(x, image_shape[1] - x)
    border_distance_y = min(y, image_shape[0] - y)
    return border_distance_x < distance or border_distance_y < distance


def create_object_labels(video_data):
    """
    Create labels for objects in every frame of a video using scikit-image's regionprops.

    Parameters:
    - video_data (np.ndarray): Video data of shape (t, x, y).

    Returns:
    - frame_labels (list): List of labels for each frame.
        Each label is a list of dictionaries with 'label' and 'regionprops' fields.
    """
    t, x, y = video_data.shape
    all_dataframe = []
    unique_label_counter = 1

    for frame_index in range(t):
        frame = video_data[frame_index]

        # Assuming your images are already labeled
        labeled_frame = frame.copy()

        # Calculate region properties using regionprops
        props = regionprops(labeled_frame)

        # frame_properties = []
        dataframe_properties = []

        for prop in props:
            region_label = prop.label

            # Create a unique label across frames
            unique_label = f"{region_label}_{unique_label_counter}"
            unique_label_counter += 1

            properties_dict = {
                'label': unique_label,
                'frame_index': frame_index,
                'major_axis_length': prop.major_axis_length,
                'minor_axis_length': prop.minor_axis_length,
                'bbox': prop.bbox,
                'area_bbox': prop.area_bbox,
                'coords': prop.coords,
                'extent': prop.extent,
                'orientation': prop.orientation,
            }
            if properties_dict['minor_axis_length'] > 1:
                dataframe_properties.append(properties_dict)

        all_dataframe.extend(dataframe_properties)

    df = pd.DataFrame(all_dataframe)

    return df

def find_caged_nucleus(dataframe, video):

    caged_video = np.zeros_like(video)

    for frame_id in tqdm(range(dataframe['frame_index'].max() + 1)):
        frame_0_data = dataframe[dataframe['frame_index'] == frame_id]

        # Create a blank labeled image for frame 0
        labeled_image = np.zeros_like(video[frame_id], dtype=np.uint8)

        # Assign cluster labels to each object in the labeled image
        for index, row in frame_0_data.iterrows():
            coords = row["coords"]
            conditions = ((row["minor_axis_length"] <= 26.5 and
                          np.abs(row["orientation"]) >= 1.47 and
                          row["extent"] >= 0.745 and
                          row["major_axis_length"] / row["minor_axis_length"] >= 2.0)
                          )

            label_value = 1 + int(conditions)

            original_label = int(row['label'].split("_")[0])

            labeled_image[video[frame_id] == original_label] = label_value

        caged_video[frame_id, :, :] = labeled_image

    return caged_video