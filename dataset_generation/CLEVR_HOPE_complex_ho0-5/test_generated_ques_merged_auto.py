"""
Script to check perform some automated tests on the merged
question .jsons; mostly sanity checks + symbolic execution on the scenegraph.
"""
import json
from collections import defaultdict
from tqdm import tqdm
import question_engine as qeng
from typing import Tuple
import os
import copy

HO_choices = [
    (None, None, "cylinder", "rubber"),
    (None, "cyan", None, "rubber"),
    ("large", None, None, "rubber"),
    (None, "cyan", "cylinder", None),
    ("large", None, "cylinder", None),
    ("large", "cyan", None, None),
]

str_HO_choices = set(str(x) for x in HO_choices)

A_choices = ("small", "gray", "cube", "metal")
OUT_DIR = "outputs"


def scene_to_objs(scene):
    return set(
        (x["size"], x["color"], x["shape"], x["material"]) for x in scene["objects"]
    )


def question_to_attrs(program):
    name_to_idx = {"size": 0, "color": 1, "shape": 2, "material": 3}

    stack = {}
    for i, node in enumerate(program):
        assert "type" in node or "function" in node
        assert "type" not in node or "function" not in node
        if ("type" in node and node["type"].startswith("filter_")) or (
            "function" in node and node["function"].startswith("filter_")
        ):
            attr_name = node["type"][7:] if "type" in node else node["function"][7:]
            input = node["inputs"]
            assert len(input) == 1
            input = input[0]

            value = node["value_inputs"]
            assert len(value) == 1
            value = value[0]

            if input in stack:
                stack[i] = stack.pop(input)
            else:
                stack[i] = [None] * 4

            stack[i][name_to_idx[attr_name]] = value

    results = stack.values()
    return [tuple(x) for x in results]


def attr_match(attributes: Tuple, template: Tuple) -> bool:
    """
    Return true iff template[i] is None or template[i] == attributes[i] for all i

    :param attributes: Tuple of strings
    :param template: Tuple of strings or None
    :return:
    """
    for temp_attr, attr in zip(template, attributes):
        # if both are None then we may yet still match
        # if temp_attr is None and attr is not None then we yet may still match
        #    e.g., template = (None, blue, None, None) and attr = (small, ?, ?, ?)
        #    then we still have the possibility of a match if attr = (small, blue, None, None)

        # if temp_attr is not None and attr *is* None, then we cannot match
        #    e.g., template = (None, blue, None, None); this cannot match (small, None, cube, rubber)
        if temp_attr is not None and attr is None:
            return False
        # if neither is not None, and they disagree, then we cannot match
        if temp_attr is not None and attr is not None and temp_attr != attr:
            return False
    return True


def standarize_ans(ans):
    if isinstance(ans, bool):
        return "yes" if ans else "no"
    elif isinstance(ans, int):
        assert 0 <= ans <= 10
        return str(ans)
    else:
        assert isinstance(ans, str)
        return ans


def is_sublist(l1, l2):
    """
    True iff l1 is sublist of l2
    """
    if l1 == []:
        return True

    l1 = copy.deepcopy(l1)
    for l2_idx in range(len(l2) - 1, -1, -1):
        if l2[l2_idx] == l1[-1]:
            l1.pop()

        if len(l1) == 0:
            return True
    return False


def main():
    for split in ["val_iid", "train"]:
        # Load the scenegraphs of all images
        with open(f"outputs/scenes/CLEVR_held_out_{split}.json", "r") as infile:
            scenegraphs = json.load(infile)["scenes"]

        # Get the categories of all images
        with open(f"outputs/held_out_objects_{split}.json", "r") as infile:
            tmp = json.load(infile)
            match_ho_pair = tmp["match_ho_pair"]
            not_match_ho = tmp["not_match_ho"]
            match_ho = tmp["match_ho"]

        assert str_HO_choices == set(match_ho_pair.keys())
        assert str_HO_choices == set(not_match_ho.keys())

        # Unpack silly dicts into tuples
        for var in [match_ho_pair, not_match_ho, match_ho]:
            for ho in str_HO_choices:
                var[ho] = [(x["global_idx"], x["local_idx"]) for x in var[ho]]

        # Get mapping from (global, local) indices to image index
        with open(
            f"outputs/scenes_to_render/glob_loc_index_to_img_index_{split}.json", "r"
        ) as infile:
            glob_loc_to_img_idx = json.load(infile)
        glob_loc_to_img_idx_map = {}
        img_idx_to_glob_loc_map = {}
        for img_idx, glob_loc in enumerate(glob_loc_to_img_idx):
            glob_loc_to_img_idx_map[tuple(glob_loc)] = img_idx
            img_idx_to_glob_loc_map[img_idx] = tuple(glob_loc)

        def check_question_valid(
            q_in: dict, scenes: list, ho, test_name, atomic_filters
        ):
            stage = q_in["q_stage"]
            n_objects = scene_to_objs(scenegraphs[q_in["image_index"]])
            n_descr = set(question_to_attrs(q_in["program"]))

            ans = standarize_ans(q_in["answer"])
            assert ans == standarize_ans(
                qeng.answer_final_question(q_in, scenes[q_in["image_index"]])
            )

            if "_S1_" in stage:
                # No vis or text match
                assert (
                    img_idx_to_glob_loc_map[q_in["image_index"]]
                    in not_match_ho[str(ho)]
                )
                assert all(not attr_match(x, template=ho) for x in n_objects)
                assert all(not attr_match(x, template=ho) for x in n_descr)
            elif "_S2_" in stage:
                if test_name in ["no_text_no_vis", "no_text_exp_vis", "zero_shot_text"]:
                    # No vis or text match
                    assert (
                        img_idx_to_glob_loc_map[q_in["image_index"]]
                        in not_match_ho[str(ho)]
                    )
                    assert all(not attr_match(x, template=ho) for x in n_objects)
                    assert all(not attr_match(x, template=ho) for x in n_descr)
                else:
                    assert test_name in ["exp_text_exp_vis", "exp_text_no_vis"]
                    assert (
                        img_idx_to_glob_loc_map[q_in["image_index"]]
                        in not_match_ho[str(ho)]
                    )
                    assert all(not attr_match(x, template=ho) for x in n_objects)
                    assert any(attr_match(x, template=ho) for x in n_descr)  # text

            else:
                assert "_S3" in stage
                if test_name in ["no_text_no_vis", "exp_text_no_vis", "zero_shot_text"]:
                    # No vis or text match
                    assert (
                        img_idx_to_glob_loc_map[q_in["image_index"]]
                        in match_ho_pair[str(ho)]
                    )
                    assert all(not attr_match(x, template=ho) for x in n_objects)
                    assert all(not attr_match(x, template=ho) for x in n_descr)
                else:
                    assert test_name in ["no_text_exp_vis", "exp_text_exp_vis"]
                    assert (
                        img_idx_to_glob_loc_map[q_in["image_index"]]
                        in match_ho[str(ho)]
                    )
                    assert any(attr_match(x, template=ho) for x in n_objects)  # vision
                    assert all(not attr_match(x, template=ho) for x in n_descr)

            # Zero shot text should also have no text mention of atomic filters
            if test_name == "zero_shot_text":
                assert all(
                    not attr_match(x, template=t)
                    for x in n_descr
                    for t in atomic_filters
                )

        # Tuples
        for ho in HO_choices:
            atomic_filters = [
                ((None,) * 4)[:i] + (x,) + ((None,) * 4)[i + 1 :]
                for i, x in enumerate(ho)
                if x is not None
            ]

            answers = []  # should match (except zero-shot)
            ori_q_idxs = []  # should match (except zero-shot)

            for test_name in [
                "no_text_no_vis",
                "exp_text_no_vis",
                "no_text_exp_vis",
                "exp_text_exp_vis",
                "zero_shot_text",
            ]:
                test_name_answers = []
                test_name_ori_q_idxs = []

                print(f"{split} {test_name}: {ho}")

                json_path = f"{OUT_DIR}/questions/{split}/CLEVR_ho_{split}_{str(ho)}_{test_name}_questions.json"

                with open(json_path, "r") as infile:
                    json_file = json.load(infile)

                for q in tqdm(json_file["questions"]):
                    ans = q["answer"]
                    ans = standarize_ans(ans)
                    if test_name == "no_text_no_vis":  # Initial propagation
                        answers.append(ans)
                        ori_q_idxs.append(q["q_stage_idx"])
                    else:
                        test_name_answers.append(ans)
                        test_name_ori_q_idxs.append(q["q_stage_idx"])

                    check_question_valid(q, scenegraphs, ho, test_name, atomic_filters)

                if test_name != "no_text_no_vis" and test_name != "zero_shot_text":
                    assert answers == test_name_answers
                    assert ori_q_idxs == test_name_ori_q_idxs
                elif test_name == "zero_shot_text":
                    assert is_sublist(test_name_answers, answers)
                    assert is_sublist(test_name_ori_q_idxs, ori_q_idxs)


if __name__ == "__main__":
    main()
