"""
Create the question hdf5 files.
You should already have the hdf5 files for the raw images at the ready.
"""
from vqa_framework.data_modules.generic_clevr_loader import (
    GenericCLEVRDataModule,
    ClevrFeature,
)

HO_choices = [
    (None, None, "cylinder", "rubber"),
    (None, "cyan", None, "rubber"),
    ("large", None, None, "rubber"),
    (None, "cyan", "cylinder", None),
    ("large", None, "cylinder", None),
    ("large", "cyan", None, None),
]

OUT_DIR = "outputs"

if __name__ == "__main__":
    for test_name in [
        "no_text_no_vis",
        "zero_shot_text",
        "exp_text_no_vis",
        "no_text_exp_vis",
        "exp_text_exp_vis",
    ]:
        for ho in HO_choices:
            print(f"Processing {test_name} {ho}")
            dm_kwargs = {
                "data_dir": "outputs/hdf5_held_out_CLEVR",
                "dm_images_name": "if_I_exist_something_is_very_wrong",
                "dm_questions_name": f"{test_name}_{str(ho)}_v1.0_ques",
                "dm_train_images": None,
                "dm_val_images": None,
                "dm_test_images": None,
                "loader_num_workers": 0,
                "image_features": ClevrFeature.USER_PROVIDED,
                # Note: feature not currently documented in vqa-framework.
                "user_image_features_train": "outputs/hdf5_held_out_CLEVR/train_v1.0_img/train_ims.h5",
                "user_image_features_val": "outputs/hdf5_held_out_CLEVR/val_iid_v1.0_img/val_ims.h5",
                "user_image_features_test": "outputs/hdf5_held_out_CLEVR/test_v1.0_img/test_ims.h5",
                # paths to questions
                "dm_train_questions": f"{OUT_DIR}/questions/train/CLEVR_ho_train_{str(ho)}_{test_name}_questions.json",
                "dm_val_questions": [
                    f"{OUT_DIR}/questions/val_iid/CLEVR_ho_val_iid_{str(ho)}_{test_name}_questions.json"
                ],
                "dm_test_questions": [
                    f"{OUT_DIR}/questions/test/CLEVR_held_out_questions_test_{str(ho)}.json"
                ],  # This is inefficient and will be duplicated. Too bad!
            }

            dm = GenericCLEVRDataModule(**dm_kwargs)
            # Actually perform the question hdf5 creation. Note: the image hdf5's should already exist or else this will be a real short trip.
            dm.prepare_data()
