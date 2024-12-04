"""
Quick script to check that the generated images on a small set make sense.
"""
import matplotlib.pyplot as plt
from vqa_framework.data_modules.clevr_scripts.deprecated_scipy import imread
import json
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("output_dir", default="output", help="Directory where files are")
args = parser.parse_args()
OUTPUT_DIR = args.output_dir


def sg_to_str(sg):
    summary = ""
    for obj in sorted(
        sg["objects"], key=lambda x: np.dot(x["3d_coords"], sg["directions"]["right"])
    ):
        summary += (
            "".join(obj[key][:2] for key in ["size", "color", "shape", "material"])
            + " "
        )
    return summary


def sg_diff(match_sg, pair_sg):
    keys = set()
    keys.update(match_sg.keys())
    keys.update(pair_sg.keys())

    for key in keys:
        if key not in pair_sg:
            print(f"Pair has no key: {key}")
            continue
        if key not in match_sg:
            print(f"Match has no key: {key}")
            continue
        if key in ["objects", "image_filename", "image_index"]:
            assert match_sg[key] != pair_sg[key]
        else:
            assert match_sg[key] == pair_sg[key]

    assert len(match_sg["objects"]) == len(pair_sg["objects"])
    for match_obj, pair_obj in zip(match_sg["objects"], pair_sg["objects"]):
        diff = False
        assert set(match_obj.keys()) == set(pair_obj.keys())
        for key in match_obj:
            if match_obj[key] != pair_obj[key]:
                diff = True
                print(f"Key: {key} Match: {match_obj[key]} Pair: {pair_obj[key]}")
        if diff:
            print("-")


if __name__ == "__main__":
    img_template = ""
    scene_template = ""

    for split in ["train", "val_iid", "test"]:
        num_digits = 6
        prefix = "%s_%s_" % ("CLEVR_held_out", split)
        img_template = "%s%%0%dd.png" % (prefix, num_digits)

        print("==============================================")
        print(split)

        # Get mapping from (global, local) indices to image index
        with open(
            f"{OUTPUT_DIR}/scenes_to_render/glob_loc_index_to_img_index_{split}.json",
            "r",
        ) as infile:
            glob_loc_to_img_idx = json.load(infile)
        glob_loc_to_img_idx_map = {}
        for img_idx, glob_loc in enumerate(glob_loc_to_img_idx):
            glob_loc_to_img_idx_map[tuple(glob_loc)] = img_idx

        # Get the categories of all images
        with open(f"{OUTPUT_DIR}/held_out_objects_{split}.json", "r") as infile:
            tmp = json.load(infile)
            match_ho_pair = tmp["match_ho_pair"]
            not_match_ho = tmp["not_match_ho"]
            match_ho = tmp["match_ho"]

        # Load the scenegraphs of all images
        with open(f"{OUTPUT_DIR}/scenes/CLEVR_held_out_{split}.json", "r") as infile:
            scenegraphs = json.load(infile)["scenes"]

        HO_choices = set(match_ho.keys())
        assert HO_choices == set(match_ho_pair.keys())
        assert HO_choices == set(not_match_ho.keys())

        for ho in sorted(HO_choices):
            print("------------------------------------------")
            print(ho)
            print("No match:")
            print(len(not_match_ho[ho]))
            print("Match:")
            print(len(match_ho[ho]))
            if split != "test":
                assert len(match_ho[ho]) == len(match_ho_pair[ho])

            print("Displaying matching:")
            # Check the matching objects
            if split != "test":
                for match_struct, paired_struct in zip(match_ho[ho], match_ho_pair[ho]):
                    assert match_struct["global_idx"] == paired_struct["global_idx"]
                    match_img_idx = glob_loc_to_img_idx_map[
                        (match_struct["global_idx"], match_struct["local_idx"])
                    ]
                    paired_img_idx = glob_loc_to_img_idx_map[
                        (paired_struct["global_idx"], paired_struct["local_idx"])
                    ]

                    match_img = imread(
                        os.path.join(
                            OUTPUT_DIR, "images", split, img_template % match_img_idx
                        )
                    )
                    paired_img = imread(
                        os.path.join(
                            OUTPUT_DIR, "images", split, img_template % paired_img_idx
                        )
                    )
                    side_by_side = np.concatenate((match_img, paired_img), axis=1)

                    match_sg = scenegraphs[match_img_idx]
                    assert match_sg["image_index"] == match_img_idx
                    assert match_sg["image_filename"] == img_template % match_img_idx
                    print("************************")
                    print(f"Match: {sg_to_str(match_sg)}")

                    paired_sg = scenegraphs[paired_img_idx]
                    assert paired_sg["image_index"] == paired_img_idx
                    assert paired_sg["image_filename"] == img_template % paired_img_idx
                    print(f"Pair:  {sg_to_str(paired_sg)}")

                    sg_diff(match_sg, paired_sg)
                    print()

                    plt.imshow(side_by_side)
                    plt.show(block=True)
            else:  # Test split
                assert len(match_ho_pair[ho]) == 0
                for match_struct in match_ho[ho]:
                    match_img_idx = glob_loc_to_img_idx_map[
                        (match_struct["global_idx"], match_struct["local_idx"])
                    ]
                    match_img = imread(
                        os.path.join(
                            OUTPUT_DIR, "images", split, img_template % match_img_idx
                        )
                    )

                    match_sg = scenegraphs[match_img_idx]
                    assert match_sg["image_index"] == match_img_idx
                    assert match_sg["image_filename"] == img_template % match_img_idx
                    print(f"Match: {sg_to_str(match_sg)}")

                    plt.imshow(match_img)
                    plt.show(block=True)

            print("Displaying non-matching")
            for not_match_struct in not_match_ho[ho]:
                not_match_img_idx = glob_loc_to_img_idx_map[
                    (not_match_struct["global_idx"], not_match_struct["local_idx"])
                ]
                match_img = imread(
                    os.path.join(
                        OUTPUT_DIR, "images", split, img_template % not_match_img_idx
                    )
                )

                not_match_sg = scenegraphs[not_match_img_idx]
                assert not_match_sg["image_index"] == not_match_img_idx
                assert (
                    not_match_sg["image_filename"] == img_template % not_match_img_idx
                )
                print(f"Not Match: {sg_to_str(not_match_sg)}")

                plt.imshow(match_img)
                plt.show(block=True)
