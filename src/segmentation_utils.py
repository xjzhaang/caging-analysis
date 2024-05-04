import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io, util
from skimage.measure import regionprops
from cellpose import models as cellpose_models
from tqdm import tqdm


class MyoblastDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = Path(file_path)
        self.volume = io.imread(self.file_path)

    def __len__(self):
        return len(self.volume)

    def __getitem__(self, idx):
        img = self.volume[idx]
        data = {}
        if img.ndim == 2:
            data["image"] = util.img_as_float32(img)
        else:
            data["image"] = util.img_as_float32(img[1])

        return data


def inference_loop(dataloader, model):
    reconstructed_volume = []
    with torch.inference_mode():
        for batch, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            image = data['image'].numpy()
            pred_mask, _, _ = model.eval(image, channels=[0,0], diameter=48.11, normalize=True, net_avg=False)

            # regions = regionprops(pred_mask)
            # filtered_mask = np.zeros_like(pred_mask)
            # for region in regions:
            #     if region.axis_minor_length > 1:
            #         filtered_mask[pred_mask == region.label] = region.label
            reconstructed_volume.append(np.stack([pred_mask], axis=0))

    # Stack the masks along a new dimension to get the 3D volume
    reconstructed_volume = np.stack(reconstructed_volume, axis=0)

    return reconstructed_volume


def segment_cells(volumes, model_path):
    model_cp = cellpose_models.CellposeModel(
        gpu=True if torch.cuda.is_available() else False,
        pretrained_model=model_path,
    )
    dataset = MyoblastDataset(volumes)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    res_image = inference_loop(dataloader, model_cp)
    return res_image