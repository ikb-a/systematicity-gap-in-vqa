"""
Script for reading in the hdf5 file of images in the CLEVR format
(e.g., as created by the CLEVR dataloader, or by a GenericCLEVRDataModule)
and using it to produce the Faster-RCNN features used by LXMERT.
"""

import os
import torch
from vislang.vislang.models.lxmert.processing_image import Preprocess
from vislang.vislang.models.lxmert.modeling_frcnn import GeneralizedRCNN
import vislang.vislang.models.lxmert.utils as utils
import logging

from vislang.vislang.models.lxmert.feat_extraction.clevr_frcnn_extr import extract_feats


def main(
    image_hdf5_path: str,
    outfile_path: str,
    device="cuda:0",
    batchsize=6,
    skip_x_images=0,
):
    """

    :param input_path: Path to hdf5 file of *images* (NOT resnet features)
    :param outfile_path: Where to save the created hdf5 file of Faster RCNN feaures.
    :param device:
    :param batchsize:
    :param skip_x_images: If non-zero,  then the first skip_x_images of the input
                          hdf5 file will be skipped (the output hdf5 will
                          contain all zero entries) for these images.
    :return:
    """
    FRCNN_ID = "unc-nlp/frcnn-vg-finetuned"

    # load models and model components
    frcnn_cfg = utils.Config.from_pretrained(FRCNN_ID)
    frcnn_cfg.model.DEVICE = device  # Overwrite the device
    frcnn = GeneralizedRCNN.from_pretrained(FRCNN_ID, config=frcnn_cfg)
    image_preprocess = Preprocess(frcnn_cfg)

    with torch.no_grad():
        if os.path.exists(outfile_path):
            logging.warning(f"Not generating {outfile_path} because already exists.")
            return

        extract_feats(
            outfile_path=outfile_path,
            batchsize=batchsize,
            device=device,
            image_preprocess=image_preprocess,
            frcnn=frcnn,
            frcnn_cfg=frcnn_cfg,
            image_hdf5_path=image_hdf5_path,
            skip_x_images=skip_x_images,
        )


if __name__ == "__main__":
    image_hdf5_path = "/media/administrator/extdrive/vqa-frame/hdf5_atom_CLEVR_val_v1.0/val_v1.0_img_dir/val_ims.h5"
    outfile_path = "/media/administrator/extdrive/vqa-frame/hdf5_atom_CLEVR_val_v1.0/val_v1.0_img_dir/user_val_vg_frcnn.h5"
    main(image_hdf5_path=image_hdf5_path, outfile_path=outfile_path, batchsize=6)
