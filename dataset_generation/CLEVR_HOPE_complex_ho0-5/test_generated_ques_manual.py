"""
Quick script to check some of the generated questions.
"""
import random

import matplotlib.pyplot as plt
from vqa_framework.data_modules.clevr_scripts.deprecated_scipy import imread
import json
import os
import numpy as np
from collections import Counter

TARGETS = ["S1", "S2", "S3", "test"]
IN_DEPTH_FAMS = False  # Did S1, S2, S3 val for the first ho. Note: over 89 templates. Couldn't bring self to do 'test' beyond a quick look.
DISP_NUM = 1
FIRST_ONLY = False

tuple_ho_choices = [
    (None, None, "cylinder", "rubber"),
    (None, "cyan", None, "rubber"),
    ("large", None, None, "rubber"),
    (None, "cyan", "cylinder", None),
    ("large", None, "cylinder", None),
    ("large", "cyan", None, None),
]
HO_choices = set(str(x) for x in tuple_ho_choices)

if __name__ == "__main__":
    for TARGET in TARGETS:
        if TARGET == "S2":
            for split in ["val_iid", "train"]:
                num_digits = 6
                prefix = "%s_%s_" % ("CLEVR_held_out", split)
                img_template = "%s%%0%dd.png" % (prefix, num_digits)

                print("==============================================")
                print("Stage 2: Yes Text, No Vis")
                print(split)

                # Get the categories of all images
                with open(f"outputs/held_out_objects_{split}.json", "r") as infile:
                    tmp = json.load(infile)
                    match_ho_pair = tmp["match_ho_pair"]
                    not_match_ho = tmp["not_match_ho"]
                    match_ho = tmp["match_ho"]

                # HO_choices = set(match_ho.keys())
                assert HO_choices == set(match_ho_pair.keys())
                assert HO_choices == set(not_match_ho.keys())

                first = True
                for ho in sorted(HO_choices):
                    if not first:
                        continue
                    first = not FIRST_ONLY
                    questions_path_match = f"outputs/questions/{split}/tmp/CLEVR_held_out_questions_S2_match_ho_{split}_{str(ho)}.json"
                    questions_path_pair = f"outputs/questions/{split}/tmp/CLEVR_held_out_questions_S2_pair_ho_{split}_{str(ho)}.json"

                    with open(questions_path_match, "r") as infile:
                        questions_match = json.load(infile)["questions"]
                    with open(questions_path_pair, "r") as infile:
                        questions_pair = json.load(infile)["questions"]

                    print("------------------------------------------")
                    print(ho)
                    print(f"No match ims: {len(not_match_ho[ho])}")
                    print(f"Match ims [NOT USED FOR STAGE 2]: {len(match_ho[ho])}")
                    print(f"Total questions: {len(questions_pair)}")
                    assert len(match_ho[ho]) == len(match_ho_pair[ho])
                    assert len(questions_match) == len(questions_pair)

                    print("Displaying matching + Pair:")
                    # NOTE: First DISP_NUM, last DISP_NUM, first 1 for each QFI, + DISP_NUM random
                    # NOTE: 1 per QFI is too many; disabled

                    question_indices = (
                        list(range(DISP_NUM))
                        + list(
                            range(len(questions_match) - DISP_NUM, len(questions_match))
                        )
                        + random.sample(
                            population=list(
                                range(DISP_NUM, len(questions_match) - DISP_NUM)
                            ),
                            k=DISP_NUM,
                        )
                    )

                    # Check the matching objects
                    def display_q_idx(q_idx):
                        print(f"Question Index: {q_idx}", flush=True)
                        q_m = questions_match[q_idx]
                        assert q_m["question_index"] == q_idx
                        q_p = questions_pair[q_idx]
                        assert q_p["question_index"] == q_idx
                        assert q_m["question"] != q_p["question"]
                        assert q_m["answer"] == q_p["answer"]

                        match_img_idx = q_m["image_index"]
                        match_img = imread(
                            os.path.join(
                                "outputs", "images", split, img_template % match_img_idx
                            )
                        )

                        print("************************")
                        print(f"Image index: {match_img_idx}")
                        print(f"Match Question: {q_m['question']}")
                        print(f"Pair Question: {q_p['question']}")
                        print(f"Answer: {q_m['answer']}")
                        print()

                        plt.imshow(match_img)
                        plt.show(block=True)

                    for q_idx in question_indices:
                        display_q_idx(q_idx)

                    qfi_counter = Counter()
                    indexes = list(range(DISP_NUM, len(questions_match) - DISP_NUM))
                    random.shuffle(indexes)
                    for q_idx in indexes:
                        if q_idx in question_indices:
                            continue

                        # print(f"Question Index: {q_idx}", flush=True)
                        q_m = questions_match[q_idx]
                        assert q_m["question_index"] == q_idx
                        q_p = questions_pair[q_idx]
                        assert q_p["question_index"] == q_idx
                        assert q_m["question"] != q_p["question"]
                        assert q_m["answer"] == q_p["answer"]
                        assert q_m["template_filename"] == q_p["template_filename"]
                        assert (
                            q_m["question_family_index"] == q_p["question_family_index"]
                        )

                        if (
                            IN_DEPTH_FAMS
                            and qfi_counter[
                                (q_m["template_filename"], q_m["question_family_index"])
                            ]
                            < 1
                        ):
                            print(
                                (q_m["template_filename"], q_m["question_family_index"])
                            )
                            qfi_counter.update(
                                [
                                    (
                                        q_m["template_filename"],
                                        q_m["question_family_index"],
                                    )
                                ]
                            )
                            assert (
                                qfi_counter[
                                    (
                                        q_m["template_filename"],
                                        q_m["question_family_index"],
                                    )
                                ]
                                > 0
                            )
                            display_q_idx(q_idx)

        elif TARGET == "S3":
            for split in ["val_iid", "train"]:
                num_digits = 6
                prefix = "%s_%s_" % ("CLEVR_held_out", split)
                img_template = "%s%%0%dd.png" % (prefix, num_digits)

                print("==============================================")
                print("Stage 3: No text, Yes vis.")
                print(split)

                # Get the categories of all images
                with open(f"outputs/held_out_objects_{split}.json", "r") as infile:
                    tmp = json.load(infile)
                    match_ho_pair = tmp["match_ho_pair"]
                    not_match_ho = tmp["not_match_ho"]
                    match_ho = tmp["match_ho"]

                # HO_choices = set(match_ho.keys())
                assert HO_choices == set(match_ho_pair.keys())
                assert HO_choices == set(not_match_ho.keys())

                first = True
                for ho in sorted(HO_choices):
                    if not first:
                        continue
                    first = not FIRST_ONLY

                    questions_path_match = f"outputs/questions/{split}/tmp/CLEVR_held_out_questions_S3_match_ho_{split}_{str(ho)}.json"
                    questions_path_pair = f"outputs/questions/{split}/tmp/CLEVR_held_out_questions_S3_pair_ho_{split}_{str(ho)}.json"

                    with open(questions_path_match, "r") as infile:
                        questions_match = json.load(infile)["questions"]
                    with open(questions_path_pair, "r") as infile:
                        questions_pair = json.load(infile)["questions"]

                    print("------------------------------------------")
                    print(ho)
                    print(
                        f"No match ims [NOT USED FOR STAGE 3]: {len(not_match_ho[ho])}"
                    )
                    print(f"Match ims: {len(match_ho[ho])}")
                    print(f"Match questions (Stage 3): {len(questions_pair)}")
                    assert len(match_ho[ho]) == len(match_ho_pair[ho])
                    assert len(questions_match) == len(questions_pair)

                    print("Displaying matching + Pair:")
                    # NOTE: First DISP_NUM, last DISP_NUM, first 1 for each QFI, + DISP_NUM random
                    # NOTE: 1 per QFI is too many; disabled

                    question_indices = (
                        list(range(DISP_NUM))
                        + list(
                            range(len(questions_match) - DISP_NUM, len(questions_match))
                        )
                        + random.sample(
                            population=list(
                                range(DISP_NUM, len(questions_match) - DISP_NUM)
                            ),
                            k=DISP_NUM,
                        )
                    )

                    # Check the matching objects
                    def display_q_idx(q_idx):
                        print(f"Question Index: {q_idx}", flush=True)
                        q_m = questions_match[q_idx]
                        assert q_m["question_index"] == q_idx
                        q_p = questions_pair[q_idx]
                        assert q_p["question_index"] == q_idx
                        assert q_m["question"] == q_p["question"]
                        assert q_m["answer"] == q_p["answer"]

                        match_img_idx = q_m["image_index"]
                        paired_img_idx = q_p["image_index"]
                        match_img = imread(
                            os.path.join(
                                "outputs", "images", split, img_template % match_img_idx
                            )
                        )
                        paired_img = imread(
                            os.path.join(
                                "outputs",
                                "images",
                                split,
                                img_template % paired_img_idx,
                            )
                        )
                        side_by_side = np.concatenate((match_img, paired_img), axis=1)

                        print("************************")
                        print(f"Match Image index: {match_img_idx}")
                        print(f"Pair Image index: {paired_img_idx}")
                        print(f"Question: {q_m['question']}")
                        print(f"Answer: {q_m['answer']}")
                        print()

                        plt.imshow(side_by_side)
                        plt.show(block=True)

                    for q_idx in question_indices:
                        display_q_idx(q_idx)

                    qfi_counter = Counter()
                    indexes = list(range(DISP_NUM, len(questions_match) - DISP_NUM))
                    random.shuffle(indexes)
                    for q_idx in indexes:
                        if q_idx in question_indices:
                            continue

                        print(f"Question Index: {q_idx}", flush=True)
                        q_m = questions_match[q_idx]
                        assert q_m["question_index"] == q_idx
                        q_p = questions_pair[q_idx]
                        assert q_p["question_index"] == q_idx
                        assert q_m["question"] == q_p["question"]
                        assert q_m["answer"] == q_p["answer"]
                        assert q_m["template_filename"] == q_p["template_filename"]
                        assert (
                            q_m["question_family_index"] == q_p["question_family_index"]
                        )

                        if (
                            IN_DEPTH_FAMS
                            and qfi_counter[
                                (q_m["template_filename"], q_m["question_family_index"])
                            ]
                            < 1
                        ):
                            print(
                                (q_m["template_filename"], q_m["question_family_index"])
                            )
                            qfi_counter.update(
                                [
                                    (
                                        q_m["template_filename"],
                                        q_m["question_family_index"],
                                    )
                                ]
                            )
                            assert (
                                qfi_counter[
                                    (
                                        q_m["template_filename"],
                                        q_m["question_family_index"],
                                    )
                                ]
                                > 0
                            )
                            display_q_idx(q_idx)

                        if (
                            False and qfi_counter[q_m["template_filename"]] < 1
                        ):  # NOTE: remove False for more testing (runs 1 per QFI)
                            print(q_m["template_filename"])
                            qfi_counter.update([q_m["template_filename"]])
                            display_q_idx(q_idx)

        elif TARGET == "S1":
            for split in ["val_iid", "train"]:
                num_digits = 6
                prefix = "%s_%s_" % ("CLEVR_held_out", split)
                img_template = "%s%%0%dd.png" % (prefix, num_digits)

                print("==============================================")
                print("Stage 1: No text, no vis")
                print(split)

                # Get the categories of all images
                with open(f"outputs/held_out_objects_{split}.json", "r") as infile:
                    tmp = json.load(infile)
                    match_ho_pair = tmp["match_ho_pair"]
                    not_match_ho = tmp["not_match_ho"]
                    match_ho = tmp["match_ho"]

                # HO_choices = set(match_ho.keys())
                assert HO_choices == set(match_ho_pair.keys())
                assert HO_choices == set(not_match_ho.keys())

                first = True
                for ho in sorted(HO_choices):
                    if not first:
                        continue
                    first = not FIRST_ONLY
                    questions_path = f"outputs/questions/{split}/tmp/CLEVR_held_out_questions_S1_{split}_{str(ho)}.json"

                    if not os.path.exists(questions_path):
                        print("WARNING! File no existy. Skipy skipy")
                        continue

                    with open(questions_path, "r") as infile:
                        questions = json.load(infile)["questions"]

                    print("------------------------------------------")
                    print(ho)
                    print(f"No match ims: {len(not_match_ho[ho])}")
                    print(f"Match ims [NOT USED FOR STAGE 1]: {len(match_ho[ho])}")
                    print(f"Total questions: {len(questions)}")

                    print("Displaying matching + Pair:")
                    # NOTE: First DISP_NUM, last DISP_NUM, first 1 for each QFI, + DISP_NUM random
                    # NOTE: 1 per QFI is too many; disabled

                    question_indices = (
                        list(range(DISP_NUM))
                        + list(range(len(questions) - DISP_NUM, len(questions)))
                        + random.sample(
                            population=list(range(DISP_NUM, len(questions) - DISP_NUM)),
                            k=DISP_NUM,
                        )
                    )

                    # Check the matching objects
                    def display_q_idx(q_idx):
                        print(f"Question Index: {q_idx}", flush=True)
                        q1 = questions[q_idx]
                        assert q1["question_index"] == q_idx
                        img_idx = q1["image_index"]
                        img = imread(
                            os.path.join(
                                "outputs", "images", split, img_template % img_idx
                            )
                        )

                        print("************************")
                        print(f"Image index: {img_idx}")
                        print(f"Question: {q1['question']}")
                        print(f"Answer: {q1['answer']}")
                        print()

                        plt.imshow(img)
                        plt.show(block=True)

                    for q_idx in question_indices:
                        display_q_idx(q_idx)

                    qfi_counter = Counter()
                    indexes = list(range(DISP_NUM, len(questions) - DISP_NUM))
                    random.shuffle(indexes)
                    for q_idx in indexes:
                        if q_idx in question_indices:
                            continue

                        # print(f"Question Index: {q_idx}", flush=True)
                        q_n = questions[q_idx]
                        assert q_n["question_index"] == q_idx

                        template_id = (
                            q_n["template_filename"],
                            q_n["question_family_index"],
                        )
                        if IN_DEPTH_FAMS and qfi_counter[template_id] < 1:
                            print(template_id)
                            qfi_counter.update([template_id])
                            assert qfi_counter[template_id] > 0
                            display_q_idx(q_idx)

        else:
            assert TARGET == "test"
            split = "test"
            num_digits = 6
            prefix = "%s_%s_" % ("CLEVR_held_out", split)
            img_template = "%s%%0%dd.png" % (prefix, num_digits)

            print("==============================================")
            print("Test: Yes text, yes vis")
            print(split)

            # Get the categories of all images
            with open(f"outputs/held_out_objects_{split}.json", "r") as infile:
                tmp = json.load(infile)
                match_ho_pair = tmp["match_ho_pair"]
                not_match_ho = tmp["not_match_ho"]
                match_ho = tmp["match_ho"]

            # HO_choices = set(match_ho.keys())
            assert HO_choices == set(match_ho_pair.keys())
            assert HO_choices == set(not_match_ho.keys())

            first = True
            for ho in sorted(HO_choices):
                assert len(match_ho_pair[ho]) == 0
                assert len(not_match_ho[ho]) == 0

                if not first:
                    continue
                first = not FIRST_ONLY
                questions_path = f"outputs/questions/{split}/CLEVR_held_out_questions_{split}_{str(ho)}.json"

                with open(questions_path, "r") as infile:
                    questions = json.load(infile)["questions"]

                print("------------------------------------------")
                print(ho)
                print(f"Match ims: {len(match_ho[ho])}")
                print(f"Total questions: {len(questions)}")

                print("Displaying matching + Pair:")
                # NOTE: First DISP_NUM, last DISP_NUM, first 1 for each QFI, + DISP_NUM random
                # NOTE: 1 per QFI is too many; disabled

                question_indices = (
                    list(range(DISP_NUM))
                    + list(range(len(questions) - DISP_NUM, len(questions)))
                    + random.sample(
                        population=list(range(DISP_NUM, len(questions) - DISP_NUM)),
                        k=DISP_NUM,
                    )
                )

                # Check the matching objects
                def display_q_idx(q_idx):
                    print(f"Question Index: {q_idx}", flush=True)
                    q1 = questions[q_idx]
                    assert q1["question_index"] == q_idx
                    img_idx = q1["image_index"]
                    img = imread(
                        os.path.join("outputs", "images", split, img_template % img_idx)
                    )

                    print("************************")
                    print(f"Image index: {img_idx}")
                    print(f"Question: {q1['question']}")
                    print(f"Answer: {q1['answer']}")
                    print()

                    plt.imshow(img)
                    plt.show(block=True)

                for q_idx in question_indices:
                    display_q_idx(q_idx)

                qfi_counter = Counter()
                indexes = list(range(DISP_NUM, len(questions) - DISP_NUM))
                random.shuffle(indexes)
                for q_idx in indexes:
                    if q_idx in question_indices:
                        continue

                    # print(f"Question Index: {q_idx}", flush=True)
                    q_n = questions[q_idx]
                    assert q_n["question_index"] == q_idx

                    template_id = (
                        q_n["template_filename"],
                        q_n["question_family_index"],
                    )
                    if IN_DEPTH_FAMS and qfi_counter[template_id] < 1:
                        print(template_id)
                        qfi_counter.update([template_id])
                        assert qfi_counter[template_id] > 0
                        display_q_idx(q_idx)
