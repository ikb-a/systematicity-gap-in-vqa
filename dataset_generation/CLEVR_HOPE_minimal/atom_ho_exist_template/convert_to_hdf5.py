"""
Script for converting our generated questions into hdf5 files that can be loaded by the GenericCLEVRDataModule.
"""

from vqa_framework.data_modules.generic_clevr_loader import GenericCLEVRDataModule
import os

# This is the path to the directory of question .jsons that was generated
INPUT_QUESTIONS_DIR = "./output_atom_ho_exists"

# This is where the created hdf5 files will be saved
OUTPUT_DIR = "./output_atom_ho_exists/questions/minimal/"


def main():
    question_file_paths = []
    for filename in os.listdir(INPUT_QUESTIONS_DIR):
        if filename.endswith(".json") and filename.startswith("atom_ho_exist_"):
            question_file_paths.append(os.path.join(INPUT_QUESTIONS_DIR, filename))

    dm = GenericCLEVRDataModule(
        data_dir=OUTPUT_DIR,
        dm_images_name=None,
        dm_questions_name="val_v1.0_ques_dir",
        dm_train_images=None,
        dm_train_scenes=None,
        dm_train_questions=None,
        dm_val_questions=question_file_paths,
        image_features=None,
    )
    dm.prepare_data()
    dm.setup()


if __name__ == "__main__":
    main()
