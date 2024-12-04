import PIL.Image
import h5py
import os
from PIL import Image
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm

QUESTIONS_DIR = "./output_atom_ho_exists"
HDF5_IMS_FILE_PATH = "./output_atom_ho_exists/images/minimal/val_ims.h5"
OUTPUT_DIR = os.path.join(QUESTIONS_DIR, "visualizations")


def create_visualization(questions: dict, filename: str, feature_h5):
    row_questions = defaultdict(lambda: set())
    row_var_questions = defaultdict(lambda: defaultdict(lambda: set()))

    variants = set(q["variant_index"] for q in questions["questions"])
    assert variants == set(range(0, max(variants) + 1))

    os.makedirs(os.path.join(OUTPUT_DIR, filename))

    for var in tqdm(range(max(variants) + 1)):
        var_questions = [q for q in questions["questions"] if q["variant_index"] == var]
        max_row = max(q["row_index"] for q in var_questions) + 1
        max_col = max(q["col_index"] for q in var_questions) + 1
        composite = np.zeros((160 * max_row, 240 * max_col, 3), dtype="uint8")

        for q in var_questions:
            feats = feature_h5["features"][q["image_index"]].transpose(1, 2, 0)
            image = Image.fromarray(feats.astype("uint8"), "RGB")
            image = image.resize((240, 160))
            array_img = np.array(image, dtype="uint8")

            r = q["row_index"]
            c = q["col_index"]
            composite[160 * r : 160 * (r + 1), 240 * c : 240 * (c + 1), :] = array_img
            row_questions[r].add(q["question"])
            row_var_questions[r][var].add(q["question"])

        final_results = Image.fromarray(composite, "RGB")
        final_results.save(os.path.join(OUTPUT_DIR, filename, f"variant{var}.png"))

    with open(os.path.join(OUTPUT_DIR, filename, "questions.txt"), "w") as outfile:
        max_row = max(q["row_index"] for q in questions["questions"]) + 1
        for r in range(max_row):
            print(f"Row {r}: {sorted(list(row_questions[r]))}", file=outfile)

        print("\n\n========================================\n", file=outfile)

        for r in range(max_row):
            print(f"Row {r}", file=outfile)
            for var in range(max(variants)):
                print(
                    f"\t\tVar {var}: {sorted(list(row_var_questions[r][var]))}",
                    file=outfile,
                )


def main():
    feature_h5 = h5py.File(HDF5_IMS_FILE_PATH, "r")

    for file in os.listdir(QUESTIONS_DIR):
        if file.endswith(".json") and file.startswith("atom_ho_exist"):
            with open(os.path.join(QUESTIONS_DIR, file), "r") as infile:
                questions_data = json.load(infile)
            create_visualization(questions_data, file[:-5], feature_h5)

    image_idx = 10
    feats = feature_h5["features"][image_idx].transpose(1, 2, 0)  # PIL needs HxWx3
    print(feats.shape)


if __name__ == "__main__":
    main()
