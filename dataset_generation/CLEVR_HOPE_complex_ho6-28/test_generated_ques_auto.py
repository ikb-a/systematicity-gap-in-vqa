"""
Script to check perform some automated tests on the generated
questions; mostly sanity checks + symbolic execution on the scenegraph.
"""
import json
from collections import defaultdict
from tqdm import tqdm
from typing import Tuple
import os
from utils import (
    load_ho_config,
    CLEVR_COLORS,
    CLEVR_MATERIALS,
    CLEVR_SHAPES,
    CLEVR_SIZES,
)
from vqa_framework.models.symbolic_exec import SymbolicClevrExecutor
from vqa_framework.data_modules.generic_clevr_loader import (
    GenericCLEVRDataModule,
    ClevrFeature,
)
import torch

# NOTE: This'll use the default ho's, you'll need to change this otherwise.
# HO_choices = load_ho_config("ho_tuples_config.json")
HO_choices = [
    (None, None, "sphere", "rubber"),
    (None, "brown", None, "rubber"),
    ("small", None, None, "rubber"),
    (None, "brown", "sphere", None),
    ("small", None, "sphere", None),
    ("small", "brown", None, None),
    (None, None, "cylinder", "metal"),
    (None, "red", None, "metal"),
    ("small", None, None, "metal"),
    (None, "red", "cylinder", None),
    ("small", None, "cylinder", None),
    ("small", "red", None, None),
    (None, None, "cube", "metal"),
    (None, "gray", None, "metal"),
    ("large", None, None, "metal"),
    (None, "gray", "cube", None),
    ("large", None, "cube", None),
    ("large", "gray", None, None),
    (None, None, "cube", "rubber"),
    (None, "purple", None, "rubber"),
    (None, "purple", "sphere", None),
    ("small", None, "cube", None),
    ("small", "purple", None, None),
]


# Path to held_out_CLEVR_add/
# (need at least `hdf5_held_out_CLEVR` and the 3 topmost .json files in `scenes`)
DATASET_PATH = "/media/administrator/extdrive/vqa-frame/held_out_CLEVR_add/"


def create_datamodule(ho: tuple):
    """
    create data module for ho, with no images
    :param ho:
    :return:
    """
    e_type = "no_text_no_vis"

    return GenericCLEVRDataModule(
        dm_train_scenes=os.path.join(
            DATASET_PATH, "scenes", "CLEVR_held_out_train.json"
        ),
        dm_val_scenes=os.path.join(
            DATASET_PATH, "scenes", "CLEVR_held_out_val_iid.json"
        ),
        dm_test_scenes=os.path.join(DATASET_PATH, "scenes", "CLEVR_held_out_test.json"),
        # These would normally be the paths to question .json files.
        # Once the .json files are converted to .hdf5 files, the .json
        # files are no longer required. We still however need to pass in the
        # names of the .json files, as that determines the names of the .hdf5
        # files we should be reading in & using.
        # Do not change these values.
        dm_train_questions=f"CLEVR_ho_train_{str(ho)}_{e_type}_questions.json",
        dm_val_questions=[f"CLEVR_ho_val_iid_{str(ho)}_{e_type}_questions.json"],
        dm_test_questions=[f"CLEVR_held_out_questions_test_{str(ho)}.json"],
        data_dir=os.path.join(DATASET_PATH, "hdf5_held_out_CLEVR"),
        image_features=None,
        dm_shuffle_train_data=False,
        dm_batch_size=1,
        val_batch_size=1,
        dm_questions_name=f"{e_type}_{str(ho)}_v1.0_ques",
        as_tokens=False,
        # Not needed b/c we used user-provided features
        dm_images_name=None,
    )


def scene_to_objs(scene):
    return set((x["size"], x["color"], x["shape"], x["material"]) for x in scene)


def question_to_attrs(program):
    name_to_idx = {"size": 0, "color": 1, "shape": 2, "material": 3}
    results = []
    in_prog = False
    for node in program[::-1]:
        if node.startswith("filter_"):
            attr_name = node[7 : node.index("[")]
            value = node[node.index("[") + 1 : -1]
            assert value in CLEVR_COLORS + CLEVR_MATERIALS + CLEVR_SHAPES + CLEVR_SIZES

            if not in_prog:
                results.append([None] * 4)

            assert results[-1][name_to_idx[attr_name]] is None
            results[-1][name_to_idx[attr_name]] = value
            in_prog = True
        else:
            in_prog = False

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
    for ho in HO_choices:
        print(ho)
        dm = create_datamodule(ho)
        dm.prepare_data()
        dm.setup()
        vocab = dm.get_vocab()
        symbolic_executor = SymbolicClevrExecutor(vocab)

        print("==============================================")
        for split, dl in [("test", dm.test_dataloader()[0])]:
            print(split)
            print(len(dl))

            for batch in tqdm(dl):
                scene_graphs = batch[3]
                assert len(scene_graphs) == 1
                programs = batch[5]
                assert len(programs) == 1
                answers = batch[4]

                # Execute and make sure it works
                assert torch.all(symbolic_executor(scene_graphs, programs) == answers)

                # Make sure the image actually obeys the constraint required
                # i.e., for test must contain HO
                t_objects = scene_to_objs(scene_graphs[0])
                assert any(attr_match(x, template=ho) for x in t_objects)

                # Make sure the question obeys the constraint required for Test
                # are met i.e., contains HO
                assert any(
                    attr_match(x, template=ho)
                    for x in question_to_attrs(
                        [vocab.program_idx_to_token(x.item()) for x in programs[0]]
                    )
                )

        for split, dl in [
            ("val", dm.val_dataloader()[0]),
            ("train", dm.train_dataloader()),
        ]:
            print(split)
            print(len(dl))

            for batch in tqdm(dl):
                scene_graphs = batch[3]
                assert len(scene_graphs) == 1
                programs = batch[5]
                assert len(programs) == 1
                answers = batch[4]

                # Execute and make sure it works
                assert torch.all(symbolic_executor(scene_graphs, programs) == answers)

                # Make sure the image actually obeys the constraint required
                # i.e. for Stage 2 does not contain HO
                n_objects = scene_to_objs(scene_graphs[0])
                assert all(not attr_match(x, template=ho) for x in n_objects)

                # Make sure the question obeys the constraint required for Stage 1
                # are met i.e., do not contain HO
                n_descr = set(
                    question_to_attrs(
                        [vocab.program_idx_to_token(x.item()) for x in programs[0]]
                    )
                )
                assert all(not attr_match(x, template=ho) for x in n_descr)
