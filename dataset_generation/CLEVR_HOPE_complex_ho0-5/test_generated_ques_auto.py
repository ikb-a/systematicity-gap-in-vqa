"""
Script to check perform some automated tests on the generated
questions; mostly sanity checks + symbolic execution on the scenegraph.
"""
import json
from collections import defaultdict
from tqdm import tqdm
import question_engine as qeng
from typing import Tuple
import os

A_choices = ("small", "gray", "cube", "metal")


def scene_to_objs(scene):
    return set(
        (x["size"], x["color"], x["shape"], x["material"]) for x in scene["objects"]
    )


def question_to_attrs(program):
    name_to_idx = {"size": 0, "color": 1, "shape": 2, "material": 3}

    stack = {}
    for i, node in enumerate(program):
        if node["type"].startswith("filter_"):
            attr_name = node["type"][7:]
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


if __name__ == "__main__":
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

        HO_choices = set(match_ho.keys())
        assert HO_choices == set(match_ho_pair.keys())
        assert HO_choices == set(not_match_ho.keys())

        # Unpack silly dicts into tuples
        for var in [match_ho_pair, not_match_ho, match_ho]:
            for ho in HO_choices:
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

        s1s2_ho_img_to_templates = defaultdict(lambda: list())

        print("==============================================")
        print("STAGE 1 Type-Question Testing")
        print(split)

        for ho in sorted(HO_choices):
            ho_tuple = eval(ho)
            questions_path = f"outputs/questions/{split}/tmp/CLEVR_held_out_questions_S1_{split}_{str(ho)}.json"

            if not os.path.exists(questions_path):
                print(f"WARNING WARNING WARNING MISSING FILE: {questions_path}")
                continue

            img_to_templates = defaultdict(lambda: list())

            with open(questions_path, "r") as infile:
                questions = json.load(infile)["questions"]

            print("------------------------------------------")
            print(ho)

            for q_idx in tqdm(range(len(questions))):
                q_n = questions[q_idx]
                assert q_n["question_index"] == q_idx
                assert split in q_n["image_filename"]

                # Check it's the right category of image
                assert img_idx_to_glob_loc_map[q_n["image_index"]] in not_match_ho[ho]

                # Make sure the image actually obeys the constraint required
                # i.e. for Stage 2 does not contain HO
                n_objects = scene_to_objs(scenegraphs[q_n["image_index"]])
                assert all(not attr_match(x, template=ho_tuple) for x in n_objects)

                # Make sure the question obeys the constraint required for Stage 1
                # are met i.e., do not contain HO
                n_descr = set(question_to_attrs(q_n["program"]))
                assert all(not attr_match(x, template=ho_tuple) for x in n_descr)

                template = (q_n["template_filename"], q_n["question_family_index"])
                assert template not in img_to_templates[q_n["image_index"]]
                img_to_templates[q_n["image_index"]].append(template)
                assert (
                    template not in s1s2_ho_img_to_templates[(ho, q_n["image_index"])]
                )
                s1s2_ho_img_to_templates[(ho, q_n["image_index"])].append(template)

                # Execute and make sure it works!
                n_answer = qeng.answer_final_question(
                    q_n, scenegraphs[q_n["image_index"]]
                )
                assert n_answer == q_n["answer"]
                assert n_answer != "__INVALID__"

            for img_idx in img_to_templates:
                assert (
                    len(img_to_templates[img_idx]) <= 9
                )  # At most 9 Stage 1 question per image
            # Make sure we actually do have 9 templates for at least one image
            assert max([len(x) for x in img_to_templates.values()]) == 9

        print("==============================================")
        print("STAGE 2 Type-Question Testing")
        print(split)

        for ho in sorted(HO_choices):
            ho_tuple = eval(ho)
            alt_ho_tuple = tuple(
                A_choices[i] if x is not None else x for i, x in enumerate(ho_tuple)
            )

            questions_path_match = f"outputs/questions/{split}/tmp/CLEVR_held_out_questions_S2_match_ho_{split}_{str(ho)}.json"
            questions_path_pair = f"outputs/questions/{split}/tmp/CLEVR_held_out_questions_S2_pair_ho_{split}_{str(ho)}.json"

            img_to_templates = defaultdict(lambda: list())

            with open(questions_path_match, "r") as infile:
                questions_match = json.load(infile)["questions"]
            with open(questions_path_pair, "r") as infile:
                questions_pair = json.load(infile)["questions"]

            print("------------------------------------------")
            print(ho)
            assert len(match_ho[ho]) == len(match_ho_pair[ho])
            assert len(questions_match) == len(questions_pair)

            for q_idx in tqdm(range(len(questions_pair))):
                q_m = questions_match[q_idx]
                assert q_m["question_index"] == q_idx
                assert split in q_m["image_filename"]
                q_p = questions_pair[q_idx]
                assert q_p["question_index"] == q_idx
                assert split in q_p["image_filename"]
                assert q_m["question"] != q_p["question"]
                assert q_m["program"] != q_p["program"]
                assert q_m["answer"] == q_p["answer"]
                assert q_m["template_filename"] == q_p["template_filename"]
                assert q_m["question_family_index"] == q_p["question_family_index"]
                assert q_m["image_index"] == q_p["image_index"]

                # Check it's the right category of image
                assert img_idx_to_glob_loc_map[q_m["image_index"]] in not_match_ho[ho]

                # Make sure the image actually obeys the constraint required
                # i.e. for Stage 2 does not contain HO
                n_objects = scene_to_objs(scenegraphs[q_p["image_index"]])
                assert all(not attr_match(x, template=ho_tuple) for x in n_objects)

                # Make sure the question obeys the constraint required for Stage 2
                # are met i.e., do or do not contain HO
                m_descr = set(question_to_attrs(q_m["program"]))
                p_descr = set(question_to_attrs(q_p["program"]))
                assert any(attr_match(x, template=ho_tuple) for x in m_descr)
                assert all(not attr_match(x, template=ho_tuple) for x in p_descr)
                p_descr -= m_descr
                assert all(attr_match(x, template=alt_ho_tuple) for x in p_descr)

                template = (q_m["template_filename"], q_m["question_family_index"])
                assert template not in img_to_templates[q_m["image_index"]]
                img_to_templates[q_m["image_index"]].append(template)
                assert (
                    template not in s1s2_ho_img_to_templates[(ho, q_m["image_index"])]
                )
                s1s2_ho_img_to_templates[(ho, q_m["image_index"])].append(template)

                # Execute and make sure it works!
                p_answer = qeng.answer_final_question(
                    q_p, scenegraphs[q_p["image_index"]]
                )
                assert p_answer == q_p["answer"]
                m_answer = qeng.answer_final_question(
                    q_m, scenegraphs[q_m["image_index"]]
                )
                assert m_answer == q_p["answer"]
                assert m_answer != "__INVALID__"

            for img_idx in img_to_templates:
                assert (
                    len(img_to_templates[img_idx]) <= 1
                )  # At most 1 Stage 2 question per image
            # Make sure we actually do have 1 template for at least one image
            assert max([len(x) for x in img_to_templates.values()]) == 1

        for key in s1s2_ho_img_to_templates:
            assert (
                len(s1s2_ho_img_to_templates[key]) <= 10
            )  # At most 10 templates total per image
        # Make sure we actually do have 10 templates for at least one image
        assert max([len(x) for x in s1s2_ho_img_to_templates.values()]) == 10

        print("==============================================")
        print("STAGE 3 Type-Question Testing")
        print(split)

        for ho in sorted(HO_choices):
            ho_tuple = eval(ho)
            alt_ho_tuple = tuple(
                A_choices[i] if x is not None else x for i, x in enumerate(ho_tuple)
            )

            questions_path_match = f"outputs/questions/{split}/tmp/CLEVR_held_out_questions_S3_match_ho_{split}_{str(ho)}.json"
            questions_path_pair = f"outputs/questions/{split}/tmp/CLEVR_held_out_questions_S3_pair_ho_{split}_{str(ho)}.json"

            img_to_templates = defaultdict(lambda: list())

            with open(questions_path_match, "r") as infile:
                questions_match = json.load(infile)["questions"]
            with open(questions_path_pair, "r") as infile:
                questions_pair = json.load(infile)["questions"]

            print("------------------------------------------")
            print(ho)
            assert len(match_ho[ho]) == len(match_ho_pair[ho])
            assert len(questions_match) == len(questions_pair)

            for q_idx in tqdm(range(len(questions_pair))):
                q_m = questions_match[q_idx]
                assert q_m["question_index"] == q_idx
                assert split in q_m["image_filename"]
                q_p = questions_pair[q_idx]
                assert q_p["question_index"] == q_idx
                assert split in q_p["image_filename"]
                assert q_m["question"] == q_p["question"]
                assert q_m["answer"] == q_p["answer"]
                assert q_m["template_filename"] == q_p["template_filename"]
                assert q_m["question_family_index"] == q_p["question_family_index"]

                # Check it's the right category of image
                assert img_idx_to_glob_loc_map[q_m["image_index"]] in match_ho[ho]
                assert img_idx_to_glob_loc_map[q_p["image_index"]] in match_ho_pair[ho]

                # Make sure the image actually obeys the constraint required
                # i.e. for Stage 3 either does or does not contain HO
                # depending on which of the images in the pair we're looking at.
                m_objects = scene_to_objs(scenegraphs[q_m["image_index"]])
                assert any(attr_match(x, template=ho_tuple) for x in m_objects)
                p_objects = scene_to_objs(scenegraphs[q_p["image_index"]])
                assert all(not attr_match(x, template=ho_tuple) for x in p_objects)
                p_objects -= m_objects  # The changed objects should match alt_ho
                assert all(attr_match(x, template=alt_ho_tuple) for x in p_objects)

                # Make sure the question obeys the constraint required for Stage 3
                # are met i.e., do no contain HO
                assert all(
                    not attr_match(x, template=ho_tuple)
                    for x in question_to_attrs(q_m["program"])
                )
                assert all(
                    not attr_match(x, template=ho_tuple)
                    for x in question_to_attrs(q_p["program"])
                )

                template = (q_m["template_filename"], q_m["question_family_index"])
                img_to_templates[q_m["image_index"]].append(template)
                img_to_templates[q_p["image_index"]].append(template)

                # Execute and make sure it works!
                p_answer = qeng.answer_final_question(
                    q_p, scenegraphs[q_p["image_index"]]
                )
                assert p_answer == q_p["answer"]
                m_answer = qeng.answer_final_question(
                    q_m, scenegraphs[q_m["image_index"]]
                )
                assert m_answer == q_p["answer"]
                assert m_answer != "__INVALID__"

            for img_idx in img_to_templates:
                assert (
                    len(img_to_templates[img_idx]) <= 10
                )  # At most 10 questions per image
                assert len(img_to_templates[img_idx]) == len(
                    set(img_to_templates[img_idx])
                )  # No duplicate templates

    print("==============================================")
    print("TEST SET TESTING")
    for split in ["test"]:
        # Load the scenegraphs of all images
        with open(f"outputs/scenes/CLEVR_held_out_{split}.json", "r") as infile:
            scenegraphs = json.load(infile)["scenes"]

        # Get the categories of all images
        with open(f"outputs/held_out_objects_{split}.json", "r") as infile:
            tmp = json.load(infile)
            match_ho_pair = tmp["match_ho_pair"]
            not_match_ho = tmp["not_match_ho"]
            match_ho = tmp["match_ho"]

        HO_choices = set(match_ho.keys())
        assert HO_choices == set(match_ho_pair.keys())
        assert HO_choices == set(not_match_ho.keys())

        # Unpack silly dicts into tuples
        for var in [match_ho_pair, not_match_ho, match_ho]:
            for ho in HO_choices:
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

        for ho in sorted(HO_choices):
            ho_tuple = eval(ho)

            print(ho)
            assert len(match_ho_pair[ho]) == 0
            assert len(not_match_ho[ho]) == 0

            questions_path_test = f"outputs/questions/{split}/CLEVR_held_out_questions_{split}_{str(ho)}.json"
            img_to_templates = defaultdict(lambda: list())

            with open(questions_path_test, "r") as infile:
                questions_test = json.load(infile)["questions"]

            for q_idx in tqdm(range(len(questions_test))):
                q_t = questions_test[q_idx]
                assert q_t["question_index"] == q_idx
                assert split in q_t["image_filename"]
                # Check it's the right category of image
                assert img_idx_to_glob_loc_map[q_t["image_index"]] in match_ho[ho]

                # Make sure the image actually obeys the constraint required
                # i.e., for test must contain HO
                t_objects = scene_to_objs(scenegraphs[q_t["image_index"]])
                assert any(attr_match(x, template=ho_tuple) for x in t_objects)

                # Make sure the question obeys the constraint required for Test
                # are met i.e., contains HO
                assert any(
                    attr_match(x, template=ho_tuple)
                    for x in question_to_attrs(q_t["program"])
                )

                template = (q_t["template_filename"], q_t["question_family_index"])
                img_to_templates[q_t["image_index"]].append(template)

                t_answer = qeng.answer_final_question(
                    q_t, scenegraphs[q_t["image_index"]]
                )
                assert t_answer == q_t["answer"]
                assert t_answer != "__INVALID__"

            for img_idx in img_to_templates:
                assert (
                    len(img_to_templates[img_idx]) <= 1
                )  # At most 1 question per image
