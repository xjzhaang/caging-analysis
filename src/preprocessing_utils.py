import numpy as np
from skimage import transform, exposure, restoration, feature, util
import streamlit as st
def preprocess_image(image, grooves_ch):
    image = per_channel_scaling(image)
    image = apply_intensity_clipping(image)
    image = apply_denoising(image)
    if grooves_ch is not None:
        grooves_ch = exposure.equalize_hist(grooves_ch)
        image = detect_and_rotate_angle(image, grooves_ch)
    if image.ndim == 3:
        return util.img_as_ubyte(image)
    return image


def detect_and_rotate_angle(video_frames, grooves_ch, num_frames_for_angle=5):
    """
    Detects the rotation angle of a video based on Hough line transform
    from a subset of frames and rotates all frames accordingly.

    Parameters:
    - video_frames: NumPy array representing the video frames with shape (z, x, y) or (z, c, x, y).
    - num_frames_for_angle: Number of frames to use for angle detection.

    Returns:
    - rotated_frames: NumPy array representing the rotated video frames.
    """
    video_frames = np.array(video_frames)

    if video_frames.ndim == 4:
        frames_for_angle = grooves_ch[:num_frames_for_angle, 0, :, :]
    else:
        frames_for_angle = grooves_ch[:num_frames_for_angle]
    average_angle = compute_average_angle(frames_for_angle)
    rotated_frames = np.zeros_like(video_frames)

    for idx, frame in enumerate(video_frames):
        if video_frames.ndim == 4:
            rotated_frame = np.array(
                [transform.rotate(channel, angle=average_angle, preserve_range=True) for channel in frame])
        else:
            rotated_frame = transform.rotate(frame, angle=average_angle, preserve_range=True)

        rotated_frames[idx] = rotated_frame
    rotated_frames = rotated_frames.astype(np.float32)
    return rotated_frames


def angle_from_orientation(orientation, use_structure_tensor=False):
    """
    Calculate the angle considering special cases.

    Parameters:
    - orientation (float): The orientation angle.

    Returns:
    - float: The angle.
    """
    # Ensure orientation is in the range [0, 180)
    orientation = orientation % 180

    angle = 0
    if use_structure_tensor:
        threshold = 0.5  # You can adjust this threshold as needed
        if abs(orientation) < threshold or abs(orientation - 180) < threshold:
            angle = 90
        elif abs(orientation - 90) < threshold:
            angle = 0
        else:
            if 0 <= orientation < 90:
                if orientation >= 45:
                    angle = 90 - orientation
                else:
                    angle = -orientation + 90
            elif 90 <= orientation < 180:
                if orientation >= 135:
                    angle = 270 - orientation
                else:
                    angle = 90 - orientation
    else:
        if 0 <= orientation < 90:
            if orientation >= 45:
                angle = orientation - 90
            else:
                angle = orientation + 90
        elif 90 <= orientation < 180:
            if orientation >= 135:
                angle = orientation + 90
            else:
                angle = orientation - 90
    return angle


def compute_average_angle(grooves_ch):
    """
    Computes the average rotation angle based on Hough line transform.
    Then rotates the image to have horizontal grooves based on structure tensor orientation.

    Parameters:
    - frames: NumPy array representing the frames with shape (num_frames, x, y).

    Returns:
    - final_angle: The average rotation angle.
    """

    angles = []

    for frame in grooves_ch:
        edges = feature.canny(frame, sigma=5)
        h, theta, d = transform.hough_line(edges)
        hspace, hu_angle, dists = transform.hough_line_peaks(h, theta, d)

        orientation_rad = np.median(hu_angle)
        orientation_deg = np.rad2deg(orientation_rad)
        angles.append(orientation_deg)

    orientation = np.mean(angles)
    final_angle = angle_from_orientation(orientation)

    print(orientation, final_angle)
    return final_angle


def per_channel_scaling(image):
    """
    Perform per-channel scaling to the range [0, 1].

    Parameters:
    - image: numpy array with shape (t, c, x, y)

    Returns:
    - scaled_image: numpy array with the same shape, but values scaled to [0, 1]
    """

    image = image.astype(np.float32)

    if image.ndim == 3:
        # 3D Image (t, x, y)
        min_vals = np.min(image, axis=(1, 2), keepdims=True)
        max_vals = np.max(image, axis=(1, 2), keepdims=True)

    elif image.ndim == 4:
        min_vals = np.min(image, axis=(2, 3), keepdims=True)
        max_vals = np.max(image, axis=(2, 3), keepdims=True)

    else:
        raise ValueError("Input image must be 3D (t, x, y) or 4D (t, c, x, y).")

    scaled_image = (image - min_vals) / (max_vals - min_vals + 1e-8)
    return scaled_image


def apply_intensity_clipping(image, clip_percentile=1, channel_to_process=1):
    """
    Apply intensity clipping and total variation denoising to a 3D or 4D image.

    Parameters:
    - image (numpy.ndarray): Input image (3D or 4D).
    - clip_percentile (float): Percentile value for intensity clipping.
    - weight (float): Weight parameter for total variation denoising.
    - channel_to_process (int): If the input is 4D, specify the channel to process.

    Returns:
    - numpy.ndarray: Processed image.
    """
    if image.ndim == 3:
        # 3D Image (t, x, y)
        processed_image = np.zeros_like(image)

        for t in range(image.shape[0]):
            # Intensity clipping
            clip_max = np.percentile(image[t], 100 - clip_percentile)
            clipped_image = np.clip(image[t], 0, clip_max)
            processed_image[t] = clipped_image
            del clipped_image

    elif image.ndim == 4:
        # 4D Image (t, c, x, y)
        if channel_to_process is None:
            raise ValueError("For 4D images, specify the channel to process.")

        processed_image = np.copy(image)

        for t in range(image.shape[0]):
            # Intensity clipping for the specified channel
            clip_max = np.percentile(image[t, channel_to_process], 100 - clip_percentile)
            clipped_image = np.clip(image[t, channel_to_process], 0, clip_max)
            processed_image[t, channel_to_process] = clipped_image
            del clipped_image
    else:
        raise ValueError("Input image must be 3D (t, x, y) or 4D (t, c, x, y).")

    return processed_image


def apply_denoising(image, weight=0.02, channel_to_process=1):
    """
    Apply intensity clipping and total variation denoising to a 3D or 4D image.

    Parameters:
    - image (numpy.ndarray): Input image (3D or 4D).
    - clip_percentile (float): Percentile value for intensity clipping.
    - weight (float): Weight parameter for total variation denoising.
    - channel_to_process (int): If the input is 4D, specify the channel to process.

    Returns:
    - numpy.ndarray: Processed image.
    """
    if image.ndim == 3:
        # 3D Image (t, x, y)
        processed_image = np.zeros_like(image)

        for t in range(image.shape[0]):
            # Total variation denoising
            denoised_image = restoration.denoise_tv_chambolle(image[t], weight=weight)
            denoised_image = exposure.rescale_intensity(denoised_image, in_range='image', out_range='dtype')
            processed_image[t] = denoised_image
            del denoised_image

    elif image.ndim == 4:
        # 4D Image (t, c, x, y)
        if channel_to_process is None:
            raise ValueError("For 4D images, specify the channel to process.")

        processed_image = np.copy(image)

        for t in range(image.shape[0]):
            # Total variation denoising for the specified channel
            denoised_image = restoration.denoise_tv_chambolle(image[t, channel_to_process], weight=weight)
            denoised_image = exposure.rescale_intensity(denoised_image, in_range='image', out_range='dtype')
            processed_image[t, channel_to_process] = denoised_image
            del denoised_image
    else:
        raise ValueError("Input image must be 3D (t, x, y) or 4D (t, c, x, y).")

    return processed_image



