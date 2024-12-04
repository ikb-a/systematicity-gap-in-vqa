"""
Quick script to check some of the generated Stage 3 train & val_iid
questions.
"""
import random

import matplotlib.pyplot as plt
from vqa_framework.data_modules.clevr_scripts.deprecated_scipy import imread
import json
import os
import numpy as np
from collections import Counter
from utils import (
    load_ho_config,
    CLEVR_COLORS,
    CLEVR_MATERIALS,
    CLEVR_SHAPES,
    CLEVR_SIZES,
)
from vqa_framework.data_modules.generic_clevr_loader import (
    GenericCLEVRDataModule,
    ClevrFeature,
)

DISP_NUM = 1
FIRST_ONLY = True

# NOTE: This'll use the default ho's, you'll need to change this otherwise.
# HO_choices = load_ho_config("ho_tuples_config.json")
HO_choices = [
    (None, None, "sphere", "rubber"),
    (None, "brown", None, "rubber"),
    ("small", None, None, "rubber"),
    (None, "brown", "sphere", None),
    ("small", None, "sphere", None),
    ("small", "brown", None, None),
    (None, None, "cylinder", "metal"),
    (None, "red", None, "metal"),
    ("small", None, None, "metal"),
    (None, "red", "cylinder", None),
    ("small", None, "cylinder", None),
    ("small", "red", None, None),
    (None, None, "cube", "metal"),
    (None, "gray", None, "metal"),
    ("large", None, None, "metal"),
    (None, "gray", "cube", None),
    ("large", None, "cube", None),
    ("large", "gray", None, None),
    (None, None, "cube", "rubber"),
    (None, "purple", None, "rubber"),
    (None, "purple", "sphere", None),
    ("small", None, "cube", None),
    ("small", "purple", None, None),
]


# Path to held_out_CLEVR_add/
# (need at least `hdf5_held_out_CLEVR` and the 3 topmost .json files in `scenes`)
DATASET_PATH = "/media/administrator/extdrive/vqa-frame/held_out_CLEVR_add/"


def create_datamodule(ho: tuple):
    """
    create data module for ho, with no images
    :param ho:
    :return:
    """
    e_type = "no_text_no_vis"

    return GenericCLEVRDataModule(
        dm_train_scenes=os.path.join(
            DATASET_PATH, "scenes", "CLEVR_held_out_train.json"
        ),
        dm_val_scenes=os.path.join(
            DATASET_PATH, "scenes", "CLEVR_held_out_val_iid.json"
        ),
        dm_test_scenes=os.path.join(DATASET_PATH, "scenes", "CLEVR_held_out_test.json"),
        # These would normally be the paths to question .json files.
        # Once the .json files are converted to .hdf5 files, the .json
        # files are no longer required. We still however need to pass in the
        # names of the .json files, as that determines the names of the .hdf5
        # files we should be reading in & using.
        # Do not change these values.
        dm_train_questions=f"CLEVR_ho_train_{str(ho)}_{e_type}_questions.json",
        dm_val_questions=[f"CLEVR_ho_val_iid_{str(ho)}_{e_type}_questions.json"],
        dm_test_questions=[f"CLEVR_held_out_questions_test_{str(ho)}.json"],
        data_dir=os.path.join(DATASET_PATH, "hdf5_held_out_CLEVR"),
        image_features=ClevrFeature.USER_PROVIDED,
        user_image_features_val=os.path.join(
            DATASET_PATH, "hdf5_held_out_CLEVR_ims", "val_iid_v1.0_img", "val_ims.h5"
        ),
        user_image_features_train=os.path.join(
            DATASET_PATH, "hdf5_held_out_CLEVR_ims", "train_v1.0_img", "train_ims.h5"
        ),
        user_image_features_test=os.path.join(
            DATASET_PATH, "hdf5_held_out_CLEVR_ims", "test_v1.0_img", "test_ims.h5"
        ),
        dm_shuffle_train_data=False,
        dm_batch_size=1,
        val_batch_size=1,
        dm_questions_name=f"{e_type}_{str(ho)}_v1.0_ques",
        as_tokens=True,
        # Not needed b/c we used user-provided features
        dm_images_name=None,
    )


def display_q_idx(q_idx, dl):
    batch = dl.dataset[q_idx]
    features = batch[2]
    answers = batch[4]
    # programs = batch[5]
    questions = batch[0]

    print(f"Question Index: {q_idx}", flush=True)

    print("************************")
    print(f"Question: {questions}")
    print(f"Answer: {answers}")
    print()

    plt.imshow(features.cpu().numpy().transpose(1, 2, 0).astype(np.int32))
    plt.show(block=True)


if __name__ == "__main__":
    first = True
    for ho in HO_choices:
        if not first:
            continue
        first = not FIRST_ONLY

        print(ho)
        dm = create_datamodule(ho)
        dm.prepare_data()
        dm.setup()
        vocab = dm.get_vocab()

        print("==============================================")
        print("Stage 1: No text, no vis")
        for split, dl in [
            ("val", dm.val_dataloader()[0]),
            ("train", dm.train_dataloader()),
        ]:
            print(split)

            print("------------------------------------------")
            print(ho)

            print("Displaying matching + Pair:")
            # NOTE: First DISP_NUM, last DISP_NUM, first 1 for each QFI, + DISP_NUM random
            # NOTE: 1 per QFI is too many; disabled

            question_indices = (
                list(range(DISP_NUM))
                + list(range(len(dl) - DISP_NUM, len(dl)))
                + random.sample(
                    population=list(range(DISP_NUM, len(dl) - DISP_NUM)), k=DISP_NUM
                )
            )

            for q_idx in question_indices:
                display_q_idx(q_idx, dl)

        split = "test"
        dl = dm.test_dataloader()[0]

        print("==============================================")
        print("Test: Yes text, yes vis")
        print(split)

        print("Displaying matching + Pair:")
        # NOTE: First DISP_NUM, last DISP_NUM, first 1 for each QFI, + DISP_NUM random
        # NOTE: 1 per QFI is too many; disabled

        question_indices = (
            list(range(DISP_NUM))
            + list(range(len(dl) - DISP_NUM, len(dl)))
            + random.sample(
                population=list(range(DISP_NUM, len(dl) - DISP_NUM)), k=DISP_NUM
            )
        )

        for q_idx in question_indices:
            display_q_idx(q_idx, dl)
