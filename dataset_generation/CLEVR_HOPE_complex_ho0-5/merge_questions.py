"""
1) no vis, no text
2) no vis, exposed text
3) exposed vis, no text
4) exposed vis, exposed text [Note: not at once]
5) zero_shot_text
   Purge all mentions of any individual attribute in HO.
   i.e., if ho=(red, sphere); then purge all mentions of red, and purge all
   mentions of sphere.

Size of dataset 5 (zero shot text) is stored in a text file.
"""
from test_generated_ques_auto import question_to_attrs, attr_match

import json
import os

HO_choices = [
    (None, None, "cylinder", "rubber"),
    (None, "cyan", None, "rubber"),
    ("large", None, None, "rubber"),
    (None, "cyan", "cylinder", None),
    ("large", None, "cylinder", None),
    ("large", "cyan", None, None),
]

OUT_DIR = "outputs"


def merge_jsons(
    question_paths: list,
    split,
    outpath,
    version="1.0",
    date="2022-11-14",
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

            # No text, No vis
            merge_jsons(
                question_paths=[
                    questions_path_s1,
                    questions_s2_path_pair,
                    questions_s3_path_pair,
                ],
                split=split,
                outpath=f"{OUT_DIR}/questions/{split}/CLEVR_ho_{split}_{str(ho)}_no_text_no_vis_questions.json",
            )

            # Exp text, No vis
            merge_jsons(
                question_paths=[
                    questions_path_s1,
                    questions_s2_path_match,  # Exposed to text here
                    questions_s3_path_pair,
                ],
                split=split,
                outpath=f"{OUT_DIR}/questions/{split}/CLEVR_ho_{split}_{str(ho)}_exp_text_no_vis_questions.json",
            )

            # No text, Exp vis
            merge_jsons(
                question_paths=[
                    questions_path_s1,
                    questions_s2_path_pair,
                    questions_s3_path_match,
                ],  # Exposed to vision here
                split=split,
                outpath=f"{OUT_DIR}/questions/{split}/CLEVR_ho_{split}_{str(ho)}_no_text_exp_vis_questions.json",
            )

            # Exp text, Exp vis [Note: not exposed to both at once]
            merge_jsons(
                question_paths=[
                    questions_path_s1,
                    questions_s2_path_match,  # Exposed to text here
                    questions_s3_path_match,
                ],  # Exposed to vision here
                split=split,
                outpath=f"{OUT_DIR}/questions/{split}/CLEVR_ho_{split}_{str(ho)}_exp_text_exp_vis_questions.json",
            )

            # Final one uses a rejector & creates the zero-shot train set
            zero_shot_dset = merge_jsons(
                question_paths=[
                    questions_path_s1,
                    questions_s2_path_pair,  # We exclude S2 match as that's certain to contain ho in the text
                    questions_s3_path_pair,  # We exclude S3 match as it's the same text as S3 pairs
                ],
                split=split,
                outpath=f"{OUT_DIR}/questions/{split}/CLEVR_ho_{split}_{str(ho)}_zero_shot_text_questions.json",
                rejector=zero_shot_filter_rejector,
            )
            if zero_shot_dset is not None:
                zero_shot_dset, total_qs = zero_shot_dset
                print(
                    f"Zero-Shot-Text length: {len(zero_shot_dset['questions'])} out of {total_qs}"
                )
                with open(
                    f"{OUT_DIR}/questions/{split}/zero_shot_sizes.txt", "a"
                ) as outfile:
                    print(split, file=outfile)
                    print(ho, file=outfile)
                    print(
                        f"Zero-Shot-Text length: {len(zero_shot_dset['questions'])} out of {total_qs}\n",
                        file=outfile,
                    )
