import sys
import copy
import json
import os
from sg_utils import (
    generate_sg_and_questions,
    AbstractCLEVRTestCase,
    Variant,
    Directions,
    SceneGraph,
)
from sg_question_utils import generate_questions
from typing import List, Optional, Tuple, Dict
from generate_atom_sg_v1_0 import (
    NamedList,
    CLEVR_SIZES,
    CLEVR_SHAPES,
    CLEVR_MATERIALS,
    CLEVR_COLORS,
    TEMPLATE_SIDE_INPUT_INDEX,
    generate_all_identical_render_sg,
)

# Note: may need to edit some files in the blender install directory to allow it to see the needed code.
try:
    import bpy
except ImportError as e:
    INSIDE_BLENDER = False
    print("This script is intended to be called from blender like this:")
    print()
    print("blender --background --python generate_atom_ho_exists_v1_0.py")
    sys.exit(1)

# export PYTHONPATH='.'
# blender --background --python generate_atom_ho_exists_v1_0.py
# blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --width 320 --height 240 --render_num_samples 2 --scene_graphs_path output_atom_ho_exists/dev_tmp_sg.json --start_idx 0 --num_images 500

# EVEN TINIER PICS (for testing)
# blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --width 160 --height 120 --min_pixels_per_object 25 --render_num_samples 2 --scene_graphs_path output_atom_ho_exists/dev_tmp_sg.json --start_idx 0 --num_images 500

# Note: Defaults are 320x480 image with 200 pixels visible.
# Therefore, scaling down to 120*160, estimate [(120*160)/(320*480)] * 200 = 25 pixels visible.

# =========================================================================================================================


# NOTE: SIZ_SHA_COL_MAT should be the order for rows when possible.
HO_choices = [  # HOP6-28
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

# NOTE: Reorder attributes to be SIZ_SHA_COL_MAT
HO_choices = tuple((x[0], x[2], x[1], x[3]) for x in HO_choices)
assert HO_choices[0] == (None, "sphere", None, "rubber"), HO_choices[0]
assert HO_choices[22] == ("small", None, "purple", None), HO_choices[22]
assert len(HO_choices) == 23

# Add the first 6 HOPs.
HO_choices = (
    (None, "cylinder", None, "rubber"),
    (None, None, "cyan", "rubber"),
    ("large", None, None, "rubber"),
    (None, "cylinder", "cyan", None),
    ("large", "cylinder", None, None),
    ("large", None, "cyan", None),
) + HO_choices

# Note: We restrict colours for colour pairs to reduce the number of options; too many otherwise.
COL_PAIRS = NamedList(
    [(x, y) for (idx, x) in enumerate(CLEVR_COLORS) for y in CLEVR_COLORS[idx + 1 :]],
    "color",
)
# print(COL_PAIRS.list)
SIZ_PAIRS = NamedList([("small", "large")], "size")
MAT_PAIRS = NamedList([("metal", "rubber")], "material")
SHA_PAIRS = NamedList(
    [("cube", "sphere"), ("cube", "cylinder"), ("sphere", "cylinder")], "shape"
)

# Mapping from attribute name, to the corresponding side input index
# in the CLEVR question templates. The CLEVR question templates for
# the atom eval dataset can be found in ./CLEVR_1.0_atom_templates
# i.e., If we want to set one of these attributes in the question template
#       then we must adjust the corresponding index.
# TEMPLATE_SIDE_INPUT_INDEX = {'size': 0, 'color':1, 'material': 2, 'shape': 3}


# =========================================================================================================================

# AND test; currently AND of attributes, note CLEVR treats that as 'red cube' rather than AND.


class HoAndExistTestCase(AbstractCLEVRTestCase):
    """
    Number of variants varies by attribute.
    """

    def __init__(
        self,
        ho: Tuple,
        filename_template: str,  # = "CLEVR_MinimalAndExist_r%03d_v%02d_n%03d",
        match_ho: bool = True,
        question_template: str = "atom_ho_exist_template/and_ho_exist.json",
        answer_fun=lambda x, x_t, y, y_t: x == x_t and y == y_t,
    ):
        """

        :param ho: a tuple of (size, shape, colour, material)
        :param filename_template:
        :param question_template:
        :param answer_fun: function returning a boolean value based on 4 inputs:
                            x (actual value for attribute X)
                            x_t (target value for attribute X)
                            y (actual value for attribute Y)
                            y_t (target value for attribute Y)
        """
        super().__init__()

        self.answer_fun = answer_fun
        self.ho = ho
        self.match_ho = match_ho

        self.filename_template = filename_template
        self.question_template_path = question_template
        with open(question_template, "r") as infile:
            self.question_template = json.load(infile)

    def must_reject_variant(self, variant: Variant, directions: Directions) -> bool:
        return False

    def generate_render_sgs(
        self,
        variant: Variant,
        variant_id: int,
        num_objs: int,
        directions: Directions = None,
    ) -> List[SceneGraph]:
        if num_objs != 1:
            return []
        # possibly *slight* overkill, but good enough
        else:
            return generate_all_identical_render_sg(variant=variant, num_objs=num_objs)

    def generate_full_sgs(
        self,
        variant: Variant,
        variant_id: int,
        directions: Directions = None,
        num_objs: Optional[int] = None,
    ) -> Tuple[
        List[str], List[str], Dict[str, List[dict]]
    ]:  # -> Tuple[List[SceneGraph], List[str], Dict[SceneGraph, List[dict]]]:
        results = []
        filenames = []
        questions = {}

        if num_objs is not None and num_objs != 1:
            return results, filenames, questions

        num_obj = 1
        if len(variant.positions) < num_obj:
            raise ValueError(
                "Variant needs at least %i objects to create this test case" % num_obj
            )

        CLEVR_ATTRIBUTES = [CLEVR_SIZES, CLEVR_SHAPES, CLEVR_COLORS, CLEVR_MATERIALS]

        # "exists(filter_X[{target_x_val}](filter_Y[{target_y_val}](scene)))"
        attrs_of_interest = [i for i in range(4) if self.ho[i] is not None]
        assert len(attrs_of_interest) == 2

        CLEVR_ATTR_NAMES = [CLEVR_SIZES, CLEVR_SHAPES, CLEVR_COLORS, CLEVR_MATERIALS]
        X_attr = CLEVR_ATTR_NAMES[attrs_of_interest[0]]
        assert self.ho[attrs_of_interest[0]] in X_attr
        Y_attr = CLEVR_ATTR_NAMES[attrs_of_interest[1]]
        assert self.ho[attrs_of_interest[1]] in Y_attr

        row_num = 0

        def matches_ho(ho, target_x, target_y):
            return (
                target_x == ho[attrs_of_interest[0]]
                and target_y == ho[attrs_of_interest[1]]
            )

        for target_x_val in X_attr:
            for target_y_val in Y_attr:
                # Skip row if we're trying to match ho and we don't,
                # or if we're not trying to match ho and we do
                if self.match_ho:
                    if not matches_ho(self.ho, target_x_val, target_y_val):
                        continue
                elif matches_ho(self.ho, target_x_val, target_y_val):
                    continue
                # Start iterating over images in this column
                imgs_in_row = 0

                distractor_vals = [
                    (dis_x, dis_y)
                    for dis_x in X_attr
                    for dis_y in Y_attr
                    if dis_x != target_x_val and dis_y != target_y_val
                ]
                # Chose the distractor values based on variant_id;
                # reduces total images, but more importantly it
                # prevents repeated True images, so the test is less imbalanced.
                distractor_vals = distractor_vals[variant_id % len(distractor_vals)]

                irrelevant_attr = [
                    x
                    for x in CLEVR_ATTRIBUTES
                    if x.name not in [X_attr.name, Y_attr.name]
                ]
                for irrev0 in irrelevant_attr[0]:
                    for irrev1 in irrelevant_attr[1]:
                        for x_val in [target_x_val, distractor_vals[0]]:
                            for y_val in [target_y_val, distractor_vals[1]]:
                                positions = variant.positions[:num_obj]
                                thetas = variant.thetas[:num_obj]

                                final_sg = SceneGraph(
                                    camera_jitter=variant.camera_jitter,
                                    lamp_back=variant.lamp_back,
                                    lamp_fill=variant.lamp_fill,
                                    lamp_key=variant.lamp_key,
                                )
                                final_sg.add_objects(
                                    positions=positions,
                                    thetas=thetas,
                                    sizes=["X"] * num_obj,
                                    materials=None,
                                    shapes=["X"] * num_obj,
                                    colors=None,
                                )

                                assert (len(final_sg)) == 1
                                final_sg[0][irrelevant_attr[0].name] = irrev0
                                final_sg[0][irrelevant_attr[1].name] = irrev1
                                final_sg[0][X_attr.name] = x_val
                                final_sg[0][Y_attr.name] = y_val

                                # Prevent mutation by now switching to immutable string
                                final_sg_string = final_sg._json()

                                # Note this causes duplicates.
                                results.append(final_sg_string)
                                filenames.append(
                                    self.filename_template
                                    % (row_num, variant_id, imgs_in_row)
                                )

                                # Make sure not to overwrite any questions we've already created for this SG if it's come up before
                                if final_sg_string not in questions:
                                    questions[final_sg_string] = []

                                # We try the question
                                for i, (q_x, q_x_name, q_y, q_y_name) in enumerate(
                                    [
                                        (
                                            target_x_val,
                                            X_attr.name,
                                            target_y_val,
                                            Y_attr.name,
                                        )
                                    ]
                                ):
                                    remap_dict = None

                                    filled_question_template = copy.deepcopy(
                                        self.question_template
                                    )
                                    filled_question_template["nodes"][1][
                                        "side_input_vals"
                                    ][TEMPLATE_SIDE_INPUT_INDEX[q_x_name]] = q_x
                                    filled_question_template["nodes"][1][
                                        "side_input_vals"
                                    ][TEMPLATE_SIDE_INPUT_INDEX[q_y_name]] = q_y

                                    # NOTE: Could just do this once for exist as question does not change with image here.
                                    # NOTE: Fixed seed to the variant idx for reproducibility
                                    tmp_questions = generate_questions(
                                        scene_graphs=[final_sg.sg],
                                        # NOTE: This *may cause mutation*!
                                        template=filled_question_template,
                                        template_name=os.path.basename(
                                            self.question_template_path
                                        ),
                                        seed=variant_id,
                                        value_remap=remap_dict,
                                    )
                                    # We mark question with row & column so we can have duplicates (... need to think through; avg vs. best case)
                                    assert len(tmp_questions) == 1
                                    tmp_questions[0]["variant_index"] = variant_id
                                    tmp_questions[0]["row_index"] = row_num
                                    tmp_questions[0]["col_index"] = imgs_in_row

                                    # Check we have the right answer
                                    assert (
                                        tmp_questions[0]["answer"] == "yes"
                                        if self.answer_fun(
                                            x_val, target_x_val, y_val, target_y_val
                                        )
                                        else "no"
                                    )

                                    # print(questions)
                                    questions[final_sg_string] += tmp_questions

                                # we've added another img to the row
                                imgs_in_row += 1
                                # we've added another row (i.e., combination of non-queried attributes)
                row_num += 1

        assert len(results) == len(filenames)
        return results, filenames, questions


def main():
    # A "variant" corresponds to fixed settings of light & camera positions.
    # We generate several images that correspond to a variant -- this allows for
    # more controlled comparisons between similar images.
    # Number of variants must be a multiple of the number of distractor pair options.
    def number_of_variants(ho):
        if ho[0] is None:
            if ho[1] is None:
                return 35  # (None, None, 'cyan', 'rubber'):35, # (8-1) * (2-1) = 7 choices of distractor;
            elif ho[2] is None:
                return 40  # (None, 'cylinder', None, 'rubber'): 40, # (3-1) * (2-1) = 2 choices of distractor;
            else:
                assert ho[3] is None
                return 28  # (None, 'cylinder', 'cyan', None):28,  # (3-1) * (8-1) = 14 choices
        elif ho[1] is None:
            if ho[2] is None:
                return 40  # ('large', None, None, 'rubber'):40,  # (2-1) * (2-1) = 1 choice of distractor;
            else:
                assert ho[3] is None
                return (
                    35  # ('large', None, 'cyan', None):35}  # (2-1) * (8-1) = 7 choices
                )
        else:
            assert ho[2] is None
            assert ho[3] is None
            return 40  # ('large', 'cylinder', None,  None):40,  # (2-1) * (3-1) = 2 choices

    # For testing, set number of variants to 1.
    # number_of_variants = lambda x: 1

    test_cases = [
        (
            HoAndExistTestCase(
                ho=ho,
                filename_template="CLEVR_Atom_matchHo" + str(i) + "_r%03d_v%02d_n%03d",
                match_ho=True,
            ),
            number_of_variants(ho),
            "atom_ho_exist_match_ho%i.json" % (i),
        )
        for i, ho in enumerate(HO_choices)
    ]
    test_cases += [
        (
            HoAndExistTestCase(
                ho=ho,
                filename_template="CLEVR_Atom_mismatchHo"
                + str(i)
                + "_r%03d_v%02d_n%03d",
                match_ho=False,
            ),
            number_of_variants(ho),
            "atom_ho_exist_mismatch_ho%i.json" % (i),
        )
        for i, ho in enumerate(HO_choices)
    ]

    outdir = "output_atom_ho_exists"
    os.makedirs(outdir)

    # Generative code instead of reusing images:
    tmpa_sg, tmpb_q, variants = generate_sg_and_questions(
        [(x[0], x[1]) for x in test_cases],
        split="val",
        desired_num_objects=1,
        min_pixels_per_object=200,
        img_width=480,
        img_height=320,
    )

    with open(os.path.join(outdir, "tmp_sg_for_generation.json"), "w") as outfile:
        json.dump(tmpa_sg, indent=2, fp=outfile)

    for i, (_, _, filename) in enumerate(test_cases):
        with open(os.path.join(outdir, filename), "w") as outfile:
            json.dump(tmpb_q[i], indent=2, fp=outfile)

    # Save all the generated variants. This file *shouldn't* be needed, but
    # it's a nice bit of insurance in case we need to go back in the future
    # and generate new things, or debug something.
    with open(os.path.join(outdir, "variants.json"), "w") as outfile:
        json.dump([json.loads(str(v)) for v in variants], indent=2, fp=outfile)


if __name__ == "__main__":
    main()
