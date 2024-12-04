import json
from typing import List, Tuple

CLEVR_COLORS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
CLEVR_MATERIALS = ["rubber", "metal"]
CLEVR_SHAPES = ["cube", "cylinder", "sphere"]
CLEVR_SIZES = ["large", "small"]


def load_ho_config(path: str = "ho_tuples_config.json") -> List[Tuple[str]]:
    """
    Given a config file, return the corresponding list of HO choices as a list of tuples of attributes.

    Attribute order is (size, color, shape, material)

    The config .json file is expected to have 2 keys:
        1) "choice_tuples" to a list of strings. Each string should be the __repr__ of a full tuple of values.
           e.g., "('small', 'brown', 'sphere', 'rubber')"
           The ordering should be (size, color, shape, material)
        2) "ho_choices", a list of the corresponding held-out combinations.
            There should be 6 for each tuple in choice_tuples, and they should
            be in the same order.
            e.g.,   "(None, None, 'sphere', 'rubber')",
                    "(None, 'brown', None, 'rubber')",
                    "('small', None, None, 'rubber')",
                    "(None, 'brown', 'sphere', None)",
                    "('small', None, 'sphere', None)",
                    "('small', 'brown', None, None)",

    NOTE: You can delete/substitute some of the ho_choices -- e.g., change one of them
    if the pair already exists earlier on
    """

    with open(path, "r") as infile:
        data = json.load(infile)

    full_tuples = [eval(x) for x in data["choice_tuples"]]
    data = data["ho_choices"]
    data = [eval(x) for x in data]

    for i, tuple in enumerate(data):
        for val_i, (val, superset) in enumerate(
            zip(tuple, [CLEVR_SIZES, CLEVR_COLORS, CLEVR_SHAPES, CLEVR_MATERIALS])
        ):
            # Check the type matches the ordering we expect
            assert val in superset + [None], val
            # Check the type matches the full tuple this choice allegedly came from
            # assert val is None or val == full_tuples[i//6][val_i], (i, tuple, val)

    assert len(data) == len(set(data)), "Each choice in HO_choices should be unique"

    return data
