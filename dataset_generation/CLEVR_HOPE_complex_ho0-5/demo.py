"""
Sample script that loads a dataset and displays a few images.
"""
from typing import Tuple
import os
from vqa_framework.data_modules.generic_clevr_loader import (
    GenericCLEVRDataModule,
    ClevrFeature,
)


# NOTE: THESE ARE THE USER PROVIDED VALUES THAT VARY BY CLUSTER/SETUP
# Path to the downloaded location of 8GB directory held_out_CLEVR/hdf5_held_out_CLEVR/
hdf5_held_out_CLEVR = "outputs/hdf5_held_out_CLEVR"
# Choose the features you want out of "%s_ims.h5", "%s_features.h5", or "user_%s_vg_frcnn.h5"
# [signifying: raw images, resnet features, or faster-rcnn respectively]
# the %s will be replaced with the split name
img_features = "%s_ims.h5"
# Path to downloaded location of held_out_CLEVR/hdf5_held_out_CLEVR_ims/
# CAUTION! This is 440GB! In the likely case that you do not have this
# much space, then preserve the folder structure but only download the hdf5
# files for the img_features you've selected.
# e.g., If we are interested in just the raw images, then we'd only download:
#         - hdf5_held_out_CLEVR_ims/test_v1.0_img/test_ims.h5
#         - hdf5_held_out_CLEVR_ims/train_v1.0_img/train_ims.h5
#         - hdf5_held_out_CLEVR_ims/val_iid_v1.0_img/val_ims.h5
# For just the raw images, I think you only need ~125GB (ResNet features are the largest, at roughly half the total)
img_features_path = "outputs/hdf5_held_out_CLEVR_ims"
# Path to download location of <1GB directory, held_out_CLEVR/scenes/
# Note that you can skip downloading this if you don't need the scenegraphs;
# although you will need to adjust the DataLoader's arguments accordingly
# (just set all of the scenegraph paths to None)
scene_path = "outputs/scenes"
# Set to true to not load scenes
disable_scenes = True

# These are our possible held-out combinations.
# The order is (size, color, shape, material)
HO_choices = [
    (None, None, "cylinder", "rubber"),
    (None, "cyan", None, "rubber"),
    ("large", None, None, "rubber"),
    (None, "cyan", "cylinder", None),
    ("large", None, "cylinder", None),
    ("large", "cyan", None, None),
]

# For any given choice of held-out combination, ho, in HO_choices, we've got
# 5 possible tests (see names in list).
# The first 4 are straightforwards:
#     - `no_text` signifies no text-exposure to ho
#     - `exp_text` signifies exposure to text
#     - same respectively for `vis` signifying visual exposure
# The 5th is 'zero_shot_text'; this is a subset of 'no_text_no_vis', further
# filtered so that no text mention of either of the concepts in ho.
# e.g., if ho=(large, rubber) then 'zero_shot_text' has removed all questions
#       involving 'large' in the text, as well as all questions involving 'rubber'
#       in the text.
test_type = [
    "no_text_no_vis",
    "exp_text_no_vis",
    "no_text_exp_vis",
    "exp_text_exp_vis",
    "zero_shot_text",
]

DOWNLOAD_DIR = "outputs"


def demo_visualize(ho: Tuple, tt: str):
    print("Create Data Module (loading scenegraphs if provided)")
    dm = GenericCLEVRDataModule(
        # 'data_dir' is the root directory where you've downloaded the
        # hdf5 files for the question data.
        # Specifically, for each possible ho+test_type, there is a corresponding
        # folder in held_out_CLEVR/hdf5_held_out_CLEVR/
        # called f'{test_type}_{str(ho)}_v1.0_ques'
        # All of the question data is only 8GB, so it's simplest to download the
        # entire held_out_CLEVR/hdf5_held_out_CLEVR/
        # folder, and point 'data_dir' to it.
        data_dir=hdf5_held_out_CLEVR,
        # This is a hard-coded value corresponding to the question folder name.
        # please do not change it.
        dm_questions_name=f"{tt}_{str(ho)}_v1.0_ques",
        # Same as in pytorch. 0 for the main thread doing the dataloading.
        # This can & should be changed as needed.
        loader_num_workers=0,
        # Do not change. Puts the generic dataloader in USER_PROVIDED image feature
        # mode.
        image_features=ClevrFeature.USER_PROVIDED,
        # Paths to the hdf5 files you want to use
        # Note that this, strictly speaking, violates the typing on the
        # generic dataloader. The GenericCLEVRDataModule documentation needs to be
        # updated.
        user_image_features_train=os.path.join(
            img_features_path, "train_v1.0_img", img_features % "train"
        ),
        user_image_features_val=os.path.join(
            img_features_path, "val_iid_v1.0_img", img_features % "val"
        ),
        user_image_features_test=os.path.join(
            img_features_path, "test_v1.0_img", img_features % "test"
        ),
        # Paths to .json files containing scenegraphs corresponding to images
        # NOTE = You can make these None instead if you don't want/need the dataloader
        #       to provide scenegraphs
        dm_train_scenes=None
        if disable_scenes
        else os.path.join(scene_path, "CLEVR_held_out_train.json"),
        dm_val_scenes=None
        if disable_scenes
        else os.path.join(scene_path, "CLEVR_held_out_val_iid.json"),
        dm_test_scenes=None
        if disable_scenes
        else os.path.join(scene_path, "CLEVR_held_out_test.json"),
        # These would normally be the paths to question .json files.
        # Once the .json files are converted to .hdf5 files, the .json
        # files are no longer required. We still however need to pass in the
        # names of the .json files, as that determines the names of the .hdf5
        # files we should be reading in & using.
        # Do not change these values.
        dm_train_questions=f"CLEVR_ho_train_{str(ho)}_{tt}_questions.json",
        dm_val_questions=[f"CLEVR_ho_val_iid_{str(ho)}_{tt}_questions.json"],
        dm_test_questions=[f"CLEVR_held_out_questions_test_{str(ho)}.json"],
        # This value _should_ be None, but I need to update vqa-framework to allow
        # it. When *not* in USER_PROVIDED image feature mode, the dataloader normally
        # follows strict rules requiring all image features to be under this
        # directory. In this case, it's not required as we are in USER_PROVIDED feature mode
        dm_images_name="if_I_exist_something_is_very_wrong",
        # Normally these would be the paths to the directories of images. As
        # we are in USER_PROVIDED image feature mode, these inputs are not used.
        dm_train_images=None,
        dm_val_images=None,
        dm_test_images=None,
    )

    # You are now done! Congratulations. For testing purposes, we will now have
    # some simple code to visualize a few examples

    import matplotlib.pyplot as plt
    import numpy as np

    # print("Preparing")
    dm.prepare_data()
    # print("Setup")
    dm.setup()
    print("Start Demo")
    vocab = dm.get_vocab()

    """
  # Quick test to double check image with ho shows up sooner or later
  if 'exp_vis' in tt:
    print(f"val {tt} {ho}")
    for batch in dm.train_dataloader():
      features = batch[2]
      answers = batch[4]
      programs = batch[5]
      questions = batch[0]
      for i in range(features.size(0)):
        print("-------------------------------------------------")
        print("QUESTION:")
        print(
          " ".join(
            [
              vocab.question_idx_to_token(x)
              for x in questions[i].numpy().tolist()
            ]
          )
        )
        print("ANSWER:")
        print(vocab.answer_idx_to_token(answers[i].item()))
        print("PROGRAM:")
        print(
          " ".join(
            [
              vocab.program_idx_to_token(x)
              for x in programs[i].numpy().tolist()
            ]
          )
        )

        plt.imshow(
          features[i].cpu().numpy().transpose(1, 2, 0).astype(np.int32))
        plt.show()
  """

    TO_SEE = 1

    # Note: the test & val dataloaders are actually *lists* of dataloaders
    for l_name, loader in [
        ("test", dm.test_dataloader()[0]),
        ("val", dm.val_dataloader()[0]),
        ("train", dm.train_dataloader()),
    ]:
        print("============================================")
        print(f"{l_name} {tt} {ho}")
        total_seen = 0
        for batch in loader:
            features = batch[2]
            answers = batch[4]
            programs = batch[5]
            questions = batch[0]
            for i in range(features.size(0)):
                print("-------------------------------------------------")
                print("QUESTION:")
                print(
                    " ".join(
                        [
                            vocab.question_idx_to_token(x)
                            for x in questions[i].numpy().tolist()
                        ]
                    )
                )
                print("ANSWER:")
                print(vocab.answer_idx_to_token(answers[i].item()))
                print("PROGRAM:")
                print(
                    " ".join(
                        [
                            vocab.program_idx_to_token(x)
                            for x in programs[i].numpy().tolist()
                        ]
                    )
                )

                plt.imshow(
                    features[i].cpu().numpy().transpose(1, 2, 0).astype(np.int32)
                )
                plt.show()

                total_seen += 1
                if total_seen % TO_SEE == 0:
                    break
            if total_seen % TO_SEE == 0:
                break


if __name__ == "__main__":
    for ho in HO_choices:
        for tt in test_type:
            print(f"{ho}: {tt}")
            demo_visualize(ho, tt)
