"""
Script for reading in the hdf5 file of images created by the CLEVR dataloader,
and using it to produce the Faster-RCNN features used by LXMERT.
"""

from vqa_framework.data_modules.clevr_loader import CLEVRDataModule, ClevrFeature
from vqa_framework.global_settings import comp_env_vars
import h5py
import numpy as np
import os
import math
import torch
from vislang.vislang.models.lxmert.processing_image import Preprocess
from vislang.vislang.models.lxmert.modeling_frcnn import GeneralizedRCNN
import vislang.vislang.models.lxmert.utils as utils
from tqdm import tqdm
import logging


def init_feats(output_dict, outfile, N):
    feat_dsets = {}
    for key in output_dict:
        shape = list(output_dict[key].shape)
        shape[0] = N

        if output_dict[key].dtype == torch.int64:
            target_dtype = np.int64
        elif output_dict[key].dtype == torch.float32:
            target_dtype = np.float32
        else:
            raise NotImplementedError("Should just be one of int64 or float 32?")

        feat_dsets[key] = outfile.create_dataset(key, tuple(shape), dtype=target_dtype)
    return feat_dsets


def extract_feats(
    outfile_path: str,
    batchsize: int,
    device: str,
    image_preprocess: Preprocess,
    frcnn: GeneralizedRCNN,
    frcnn_cfg,
    image_hdf5_path: str,
    skip_x_images: int = 0,
):
    # Note: the output hdf5 file will be the same length as the input hdf5 file.
    # if skip_x_images is not zero, then the first skip_x_images of the input
    # hdf5 file will be skipped (the output hdf5 will contain all zero entries)
    # for these images.

    # Create new hdf5 file for extracted values.
    feat_dsets = None

    # Create new hdf5 file where we'll be storing features.
    with h5py.File(outfile_path, "w") as outfile:
        # Open existing hdf5 file of images:
        with h5py.File(image_hdf5_path, "r") as infile:
            N = len(infile["features"]) - skip_x_images
            for i in tqdm(
                range(math.ceil(N / batchsize)), total=math.ceil(N / batchsize)
            ):
                batch = infile["features"][
                    skip_x_images + i * batchsize : skip_x_images + (i + 1) * batchsize
                ]
                batch = torch.FloatTensor(np.asarray(batch, dtype=np.float32)).to(
                    device
                )  # NxCxHxW in RGB order.
                batch = batch.permute(0, 2, 3, 1)  # NxHxWxC, RGB

                input_images = (
                    []
                )  # List of N tensors, each HxWxC and BGR colour channel order
                for j in range(len(batch)):
                    input_images.append(
                        batch[j, :, :, [2, 1, 0]]
                    )  # 1xHxWxC, RGB -> HxWxC, BGR

                images, sizes, scales_yx = image_preprocess(input_images)

                output_dict = frcnn(
                    images,
                    sizes,
                    scales_yx=scales_yx,
                    padding="max_detections",
                    max_detections=frcnn_cfg.max_detections,
                    return_tensors="pt",
                )

                if feat_dsets is None:
                    feat_dsets = init_feats(
                        output_dict=output_dict, outfile=outfile, N=N + skip_x_images
                    )

                for key in output_dict:
                    feat_dsets[key][
                        skip_x_images
                        + i * batchsize : skip_x_images
                        + i * batchsize
                        + len(batch)
                    ] = (output_dict[key].cpu().detach().numpy())


def main(data_dir=None, tmp_dir=None, device="cuda:0", batchsize=10):
    FRCNN_ID = "unc-nlp/frcnn-vg-finetuned"

    if not data_dir:
        data_dir = comp_env_vars.DATA_DIR
    if not tmp_dir:
        tmp_dir = comp_env_vars.TMP_DIR

    # Make sure that the internal hdf5 file for raw images exists; create if needed.
    dm = CLEVRDataModule(
        data_dir=data_dir, tmp_dir=tmp_dir, image_features=ClevrFeature.IMAGES
    )
    dm.prepare_data()
    del dm

    # load models and model components
    frcnn_cfg = utils.Config.from_pretrained(FRCNN_ID)
    # print(frcnn_cfg)
    frcnn_cfg.model.DEVICE = device  # Overwrite the device
    frcnn = GeneralizedRCNN.from_pretrained(FRCNN_ID, config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)

    with torch.no_grad():
        for split in [
            "val",
        ]:
            outfile_path = os.path.join(
                data_dir, "hdf5_clevr", f"user_{split}_vg_frcnn.h5"
            )

            if os.path.exists(outfile_path):
                logging.warning(
                    f"Skipping generating {outfile_path} because already exists."
                )
                continue

            image_hdf5_path = os.path.join(data_dir, "hdf5_clevr", f"{split}_ims.h5")
            extract_feats(
                outfile_path=outfile_path,
                batchsize=batchsize,
                device=device,
                image_preprocess=image_preprocess,
                frcnn=frcnn,
                frcnn_cfg=frcnn_cfg,
                image_hdf5_path=image_hdf5_path,
            )


if __name__ == "__main__":
    main(batchsize=8)
