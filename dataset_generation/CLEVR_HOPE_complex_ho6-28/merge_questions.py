from test_generated_ques_auto import question_to_attrs, attr_match

import json
import os
import argparse
from utils import load_ho_config


parser = argparse.ArgumentParser()
parser.add_argument(
    "--ho_tuples_config",
    default="ho_tuples_config.json",
    type=str,
    help="Path to a .json config file specifying what held-out combinations to generate for",
)
parser.add_argument(
    "--dir", default="output", help="Directory where dataset is being created"
)
args = parser.parse_args()

HO_choices = load_ho_config(args.ho_tuples_config)
OUT_DIR = args.dir  #'outputs'


def merge_jsons(
    question_paths: list,
    split,
    outpath,
    version="1.0",
    date="2023-06-09",
    rejector=lambda x: False,
):
    if os.path.exists(outpath):
        print(f"WARNING: Skipping path because it exists; {outpath}")
        return None

    total_questions = 0

    result = {
        "info": {
            "date": date,
            "license": "Creative Commons Attribution (CC-BY 4.0)",
            "split": split,
            "version": version,
        },
        "questions": [],
    }

    new_idx = 0
    for q_path in question_paths:
        basename = os.path.basename(q_path)
        if not os.path.exists(q_path):
            print(f"WARNING: Skipping because we're missing the file {q_path}")
            return None

        with open(q_path, "r") as infile:
            sub_questions = json.load(infile)
            assert sub_questions["info"]["split"] == split
            sub_questions = sub_questions["questions"]

        for q in sub_questions:
            total_questions += 1
            if rejector(q):
                continue

            # For paranoia's sake, record where we got this question from
            q["q_stage"] = basename
            q["q_stage_idx"] = q["question_index"]

            assert q["split"] == split
            q["question_index"] = new_idx
            new_idx += 1

            result["questions"].append(q)

    with open(outpath, "w") as outfile:
        json.dump(result, outfile, sort_keys=True)
    return result, total_questions


if __name__ == "__main__":
    for ho in HO_choices:
        for split in ["val_iid", "train"]:
            # Stage 1: No presence of HO in vision or text
            questions_path_s1 = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S1_{split}_{str(ho)}.json"

            # Stage 2: Same image (not containing HO), different questions (one containing HO, one not containing)
            questions_s2_path_match = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S2_match_ho_{split}_{str(ho)}.json"
            questions_s2_path_pair = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S2_pair_ho_{split}_{str(ho)}.json"

            # Stage 3: Same question (not containing HO), different images (one containing HO, one not containing)
            questions_s3_path_match = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S3_match_ho_{split}_{str(ho)}.json"
            questions_s3_path_pair = f"{OUT_DIR}/questions/{split}/tmp/CLEVR_held_out_questions_S3_pair_ho_{split}_{str(ho)}.json"

            # Split ho into atomic filters.
            # e.g., (None, 'cyan', None, 'rubber') -> [(None, 'cyan', None, None), (None, None, None, 'rubber')]
            atomic_filters = [
                ((None,) * 4)[:i] + (x,) + ((None,) * 4)[i + 1 :]
                for i, x in enumerate(ho)
                if x is not None
            ]

            def zero_shot_filter_rejector(question: dict) -> bool:
                attrs = question_to_attrs(question["program"])
                # Reject if any combination of filter operations in the program matches
                # any atom in ho
                return any(
                    attr_match(x, template=t) for x in attrs for t in atomic_filters
                )

            # The new datasets we're creating are just S1 -- i.e., no paired datapoints (S2 & S3)
            merge_jsons(
                question_paths=[questions_path_s1],
                split=split,
                outpath=f"{OUT_DIR}/questions/{split}/CLEVR_ho_{split}_{str(ho)}_no_text_no_vis_questions.json",
            )
