"""
Code to filter already created CLEVR-ATOM-HO question files.

Specifically, we need to filter questions to remove any instances where the image matches the HOP.

Also, reports base accuracy for always predicting most common class.
"""

import json
import h5py
import os
import numpy as np
from time import sleep

# Path to val_v1.0_ques_dir directory
# Note: all you need are the val_atom_ho_exist_mismatch_ho*.h5 files
DATASET_PATH = "/media/administrator/extdrive/vqa-frame/hdf5_atom_ho_exist_CLEVR_val_v1.0/val_v1.0_ques_dir"
# path to atom_CLEVR_val_v1.0_scenes.json file
SCENES_PATH = "/media/administrator/extdrive/vqa-frame/hdf5_atom_CLEVR_val_v1.0/atom_CLEVR_val_v1.0_scenes.json"

HO_choices = [  # HO6-28
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

# NOTE: Reorder attributes to be SIZ_SHA_COL_MAT
HO_choices = [(x[0], x[2], x[1], x[3]) for x in HO_choices]

# Add HO0-5
HO_choices = [
    (
        None,
        "cylinder",
        None,
        "rubber",
    ),  # Note these are already reordered to be SIZ_SHA_COL_MAT
    (None, None, "cyan", "rubber"),
    ("large", None, None, "rubber"),
    (None, "cylinder", "cyan", None),
    ("large", "cylinder", None, None),
    ("large", None, "cyan", None),
] + HO_choices

HO_choices = tuple(HO_choices)

assert HO_choices[0] == (None, "cylinder", None, "rubber")
assert HO_choices[5] == ("large", None, "cyan", None)
assert HO_choices[6] == (None, "sphere", None, "rubber"), HO_choices[6]
assert HO_choices[28] == ("small", None, "purple", None), HO_choices[28]
assert len(HO_choices) == 29

with open(SCENES_PATH) as f:
    all_scenes = json.load(f)["scenes"]


def match_ho(scene, ho):
    assert len(scene["objects"]) == 1
    o = scene["objects"][0]

    if ho[0] is not None and ho[0] != o["size"]:
        return False
    if ho[1] is not None and ho[1] != o["shape"]:
        return False
    if ho[2] is not None and ho[2] != o["color"]:
        return False
    if ho[3] is not None and ho[3] != o["material"]:
        return False

    return True


def filter(ho_idx):
    ho_tuple = HO_choices[ho_idx]
    with h5py.File(
        os.path.join(DATASET_PATH, f"val_atom_ho_exist_mismatch_ho{ho_idx}.h5"), "r"
    ) as infile:
        # 1) Determine which indices to skip
        bad_indices = [
            infile_idx
            for infile_idx, img_idx in enumerate(infile["image_idxs"])  # type: ignore
            if match_ho(scene=all_scenes[img_idx], ho=ho_tuple)
        ]

        # double check out of paranoia
        for bad_idx in bad_indices:
            assert match_ho(
                scene=all_scenes[infile["image_idxs"][bad_idx]], ho=ho_tuple  # type: ignore
            )

        # 2) Copy the rest
        remaining = None
        with h5py.File(
            os.path.join(
                DATASET_PATH, f"val_atom_ho_exist_mismatch_ho{ho_idx}_filtered.h5"
            ),
            "w",
        ) as f:
            for key in [
                "questions",
                "image_idxs",
                "orig_idxs",
                "questions_len",
                "programs",
                "programs_len",
                "question_families",
                "answers",
                "types",
            ]:
                data = [
                    x
                    for ori_idx, x in enumerate(infile[key])  # type: ignore
                    if ori_idx not in bad_indices
                ]
                f.create_dataset(key, data=np.asarray(data, dtype=np.int32))
                if remaining is None:
                    remaining = len(data)
                else:
                    assert remaining == len(data)

    # 3) Report the percentage of the dataset removed, remaining questions, and resulting base rate.
    print(f"HO{ho_idx}: {ho_tuple}")
    print(f"Removed: {len(bad_indices)}")
    print(f"Remaining: {remaining}")

    # sleep(5)
    with h5py.File(
        os.path.join(
            DATASET_PATH, f"val_atom_ho_exist_mismatch_ho{ho_idx}_filtered.h5"
        ),
        "r",
    ) as infile2:
        answers = infile2["answers"][()]  # type: ignore
        assert np.logical_or(answers == 31, answers == 24).all()
        base_rate = sum([0 if x == 31 else 1 for x in answers]) / len(  # type: ignore
            answers  # type: ignore
        )  # "yes" is 31
    print(f"Base rate: {base_rate}")


def main():
    for ho_idx in range(29):
        filter(ho_idx)


if __name__ == "__main__":
    main()
