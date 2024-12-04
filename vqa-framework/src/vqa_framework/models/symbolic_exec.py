"""
Symbolic Executors for existing datasets. Given scenegraph + program,
will perform symbolic execution.

Modified from N2NMN code for CLEVR.
"""
from torch import nn
import torch
import random
from collections import deque
import math
from typing import Dict
import logging
import numpy as np
from vqa_framework.utils.vocab import ClosureVocab
import enum
from argparse import ArgumentParser

SHAPES_COLORS = ["blue", "green", "red"]
SHAPES_SHAPES = [
    "cube",
    "cylinder",
    "sphere",
]  # NOTE cube -> Square; sphere -> circle; cylinder -> Triangle


SHAPES_ANSWER_CANDIDATES = {"_Answer": ["yes", "no"]}


class SymbExec(enum.Enum):
    shapes = 1
    clevr = 2


def symbolic_exec_factory(symb_exec: SymbExec, vocab: ClosureVocab) -> nn.Module:
    if symb_exec == SymbExec.shapes:
        return SymbolicShapesExecutor(vocab)
    elif symb_exec == SymbExec.clevr:
        return SymbolicClevrExecutor(vocab)
    else:
        raise NotImplementedError


def add_ee_args(
    parent_parser: ArgumentParser,
    postfix: str = "",
    prefix: str = "",
    defaults: Dict[str, any] = {},
):
    """
    Update <parent_parser> with the arguments required to create a symbolic executor.

    :param parent_parser: An ArgumentParser to update with arguments required by symbolic EE
    :param postfix: Any postfix that should be added to all the arg names
    :param prefix: Any prefix that should be added to all the arg names
    :param defaults: Any default values that should be overridden (map from arg name to new default value)
    :return:
    """
    parser = parent_parser.add_argument_group(f"{prefix}EE{postfix}")
    parser.add_argument(
        f"--{prefix}ee{postfix}",
        type=str,
        default=defaults.get("ee", "shapes"),
        choices=[e.name for e in SymbExec],
    )
    return parent_parser


class SymbolicShapesExecutor(nn.Module):
    """
    Symbolic program executor for SHAPES

    Note: no trainable parameters.
    """

    def __init__(self, vocab: ClosureVocab):
        """

        <vocab> has the keys
        'question_token_to_idx',
        'program_token_to_idx',
        'answer_token_to_idx',
        'program_token_arity',
        'question_idx_to_token',
        'program_idx_to_token',
        'answer_idx_to_token'

        """
        super().__init__()
        self.vocab = vocab
        self.colors = SHAPES_COLORS
        self.shapes = SHAPES_SHAPES
        self.answer_candidates = SHAPES_ANSWER_CANDIDATES

        self.modules = {}
        self._register_modules()

    def forward(self, scenes, programs: torch.Tensor) -> torch.LongTensor:
        """

        :param scenes:
        :param programs: Nxp Tensor for batchsize N, and max program len p.
                         should be encoded using <self.vocab>['program_token_to_idx']
        :return:  Tensor of longs containing answer index for each program,
                  encoded using <self.vocab>['answer_token_to_idx']
        """
        preds = []
        for i in range(programs.shape[0]):
            try:
                pred = self.run(programs[i].cpu().numpy(), scenes[i])
                preds.append(self.vocab["answer_token_to_idx"].get(pred, -1))
            except (IndexError, NotImplementedError):
                preds.append(
                    -2
                )  # symbolic executor crashed (popped from empy deque, tried to execute <NULL>)
        return torch.LongTensor(preds).to(programs.device)
        # NOTE: type_as is recommended by Lightning docs, but I'm using to
        #       so that only the device is altered, and not the type
        #       https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu.html?highlight=create%20tensor

    def run(self, x: np.array, scene, guess=False):
        logging.debug(f"SCENE: {scene}")

        assert self.modules, "Must have scene annotations and define modules first"

        ans = None

        # Find the length of the program sequence before the '<END>' token
        length = 0
        for k in range(len(x)):
            l = len(x) - k
            if self.vocab["program_idx_to_token"][x[l - 1]] == "<END>":
                length = l
        if length == 0:
            return "error"

        self.exe_trace = []
        stack = deque()

        IGNORE_TOKENS = ["<START>", "<END>"]

        for j in range(length):
            i = length - 1 - j
            token = self.vocab["program_idx_to_token"][x[i]]

            # note: I think code is using temp as a stack. Maybe. Eliminated
            # and made proper stack (well, less improper stack anyways)

            if token not in self.modules and token not in IGNORE_TOKENS:
                raise NotImplementedError(f"Unk token {token}")
            elif token not in IGNORE_TOKENS:
                module = self.modules[token]
                if token.startswith("_Find"):
                    ans = module(list(scene), None)
                    stack.append(ans)
                elif token.startswith("_And"):
                    ans = module(stack.pop(), stack.pop())
                    stack.append(ans)
                elif token.startswith("_Transform"):
                    # Note: I believe SHAPES does not have uniqueness requirement
                    #       when performing spatial comparisons
                    tmp_input_ans = stack.pop()
                    ans = []
                    for obj in tmp_input_ans:
                        ans += module(obj, list(scene))
                    stack.append(ans)
                elif token.startswith("_Answer"):
                    ans = module(stack.pop(), None)

            logging.debug(token)
            logging.debug(f"stack: {stack}")
            # print(f"ans: {ans}")

            if ans == "error":
                break

            self.exe_trace.append(ans)

        ans = str(ans)

        if ans == "error" and guess:
            final_module = self.vocab["program_idx_to_token"][x[0]]
            if final_module in self.answer_candidates:
                ans = random.choice(self.answer_candidates[final_module])
        return ans

    def _object_info(self, obj):
        return "%s %s %s %s at %s" % (
            obj["size"],
            obj["color"],
            obj["material"],
            obj["shape"],
            str(obj["position"]),
        )

    def _register_modules(self):
        self.modules["_Answer"] = self.exist
        self.modules["_Find[blue]"] = self.filter_blue
        self.modules["_Find[green]"] = self.filter_green
        self.modules["_Find[red]"] = self.filter_red
        self.modules["_Find[square]"] = self.filter_cube
        self.modules["_Find[triangle]"] = self.filter_cylinder
        self.modules["_Find[circle]"] = self.filter_sphere
        self.modules["_And"] = self.intersect
        self.modules["_Transform[above]"] = self.relate_behind
        self.modules["_Transform[below]"] = self.relate_front
        self.modules["_Transform[left_of]"] = self.relate_left
        self.modules["_Transform[right_of]"] = self.relate_right
        # self.modules['unique'] = self.unique

    def exist(self, scene, _):
        if type(scene) == list:
            scene = [s for s in scene if s["shape"] != "EMPTY"]
            if len(scene) != 0:
                return "True"
            else:
                return "False"
        return "error"

    def filter_blue(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "blue":
                    output.append(o)
            return output
        return "error"

    def filter_green(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "green":
                    output.append(o)
            return output
        return "error"

    def filter_red(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "red":
                    output.append(o)
            return output
        return "error"

    def filter_cube(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["shape"] == "cube":
                    output.append(o)
            return output
        return "error"

    def filter_cylinder(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["shape"] == "cylinder":
                    output.append(o)
            return output
        return "error"

    def filter_sphere(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["shape"] == "sphere":
                    output.append(o)
            return output
        return "error"

    def intersect(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            scene1 = [s for s in scene1 if s["shape"] != "EMPTY"]
            scene2 = [s for s in scene2 if s["shape"] != "EMPTY"]
            output = []
            for o in scene1:
                if o in scene2:
                    output.append(o)
            return output
        return "error"

    # NOTE: original relate_left/relate_right, etc... from CLEVR is generally
    # left rather than DIRECT left; this causes problems for SHAPES

    # NOTE: relations are _directly_ above/below left/right, i.e.,
    # at most one row/col off, with the other col/row fixed.

    # NOTE: Data loader reverses position axis, back to [x,y,z] rather than
    #       the [y,x,z] used in scenes

    def relate_left(self, obj, scene):
        if type(obj) == dict and "position" in obj and type(scene) == list:
            output = []
            for o in scene:
                if (
                    math.isclose(obj["position"][0] - 1, o["position"][0])
                    and obj["position"][1] == o["position"][1]
                ):
                    output.append(o)
            if output == [] and obj["position"][0] > 0:
                x, y, z = obj["position"]
                output = [{"position": [x - 1, y, z], "shape": "EMPTY"}]
            return output
        return "error"

    def relate_right(self, obj, scene):
        if type(obj) == dict and "position" in obj and type(scene) == list:
            output = []
            for o in scene:
                if (
                    math.isclose(obj["position"][0] + 1, o["position"][0])
                    and obj["position"][1] == o["position"][1]
                ):
                    output.append(o)
            if output == [] and obj["position"][0] < 2:
                x, y, z = obj["position"]
                output = [{"position": [x + 1, y, z], "shape": "EMPTY"}]
            return output
        return "error"

    def relate_behind(self, obj, scene):
        if type(obj) == dict and "position" in obj and type(scene) == list:
            output = []
            for o in scene:
                if (
                    math.isclose(obj["position"][1] - 1, o["position"][1])
                    and obj["position"][0] == o["position"][0]
                ):
                    output.append(o)
            if output == [] and obj["position"][1] > 0:
                x, y, z = obj["position"]
                output = [{"position": [x, y - 1, z], "shape": "EMPTY"}]
            return output
        return "error"

    def relate_front(self, obj, scene):
        if type(obj) == dict and "position" in obj and type(scene) == list:
            output = []
            for o in scene:
                if (
                    math.isclose(obj["position"][1] + 1, o["position"][1])
                    and obj["position"][0] == o["position"][0]
                ):
                    output.append(o)
            if output == [] and obj["position"][1] < 2:
                x, y, z = obj["position"]
                output = [{"position": [x, y + 1, z], "shape": "EMPTY"}]
            return output
        return "error"


"""
CLEVR Executor modified from CLOSURE repo
vr/ns_vqa/clevr_executor.py
"""

CLEVR_COLORS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
CLEVR_MATERIALS = ["rubber", "metal"]
CLEVR_SHAPES = ["cube", "cylinder", "sphere"]
CLEVR_SIZES = ["large", "small"]


CLEVR_ANSWER_CANDIDATES = {
    "count": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
    "equal_color": ["yes", "no"],
    "equal_integer": ["yes", "no"],
    "equal_material": ["yes", "no"],
    "equal_shape": ["yes", "no"],
    "equal_size": ["yes", "no"],
    "exist": ["yes", "no"],
    "greater_than": ["yes", "no"],
    "less_than": ["yes", "no"],
    "query_color": [
        "blue",
        "brown",
        "cyan",
        "gray",
        "green",
        "purple",
        "red",
        "yellow",
    ],
    "query_material": ["metal", "rubber"],
    "query_size": ["small", "large"],
    "query_shape": ["cube", "cylinder", "sphere"],
    "same_color": ["yes", "no"],
    "same_material": ["yes", "no"],
    "same_size": ["yes", "no"],
    "same_shape": ["yes", "no"],
}


class SymbolicClevrExecutor(nn.Module):
    """Symbolic program executor for CLEVR"""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.colors = CLEVR_COLORS
        self.materials = CLEVR_MATERIALS
        self.shapes = CLEVR_SHAPES
        self.sizes = CLEVR_SIZES
        self.answer_candidates = CLEVR_ANSWER_CANDIDATES

        self.modules = {}
        self._register_modules()

    def forward(self, scenes, programs: torch.LongTensor) -> torch.LongTensor:
        preds = []
        for i in range(programs.shape[0]):
            try:
                pred = self.run(programs[i].cpu().numpy(), scenes[i])
                preds.append(self.vocab["answer_token_to_idx"].get(pred, -1))
            except NotImplementedError:
                preds.append(-2)  # symbolic executor crashed (tried to execute <NULL>)
        return torch.LongTensor(preds).to(programs.device)

    def run(self, x, scene, guess=False, debug=False):
        assert self.modules, "Must have scene annotations and define modules first"

        ans, temp = None, None

        # Find the length of the program sequence before the '<END>' token
        length = 0
        for k in range(len(x)):
            l = len(x) - k
            if self.vocab["program_idx_to_token"][x[l - 1]] == "<END>":
                length = l
        if length == 0:
            return "error"

        self.exe_trace = []
        for j in range(length):
            i = length - 1 - j
            token = self.vocab["program_idx_to_token"][x[i]]
            if token == "scene":
                if temp is not None:
                    ans = "error"
                    break
                temp = ans
                ans = list(scene)
            elif token in self.modules:
                module = self.modules[token]
                if token.startswith("same") or token.startswith("relate"):
                    ans = module(ans, scene)
                else:
                    ans = module(ans, temp)
                if ans == "error":
                    break
            self.exe_trace.append(ans)
            if debug:
                print(token)
                print("ans:")
                self._print_debug_message(ans)
                print("temp: ")
                self._print_debug_message(temp)
                print()
        ans = str(ans)

        if ans == "error" and guess:
            final_module = self.vocab["program_idx_to_token"][x[0]]
            if final_module in self.answer_candidates:
                ans = random.choice(self.answer_candidates[final_module])
        return ans

    def _print_debug_message(self, x):
        if type(x) == list:
            for o in x:
                print(self._object_info(o))
        elif type(x) == dict:
            print(self._object_info(x))
        else:
            print(x)

    def _object_info(self, obj):
        return "%s %s %s %s at %s" % (
            obj["size"],
            obj["color"],
            obj["material"],
            obj["shape"],
            str(obj["position"]),
        )

    def _register_modules(self):
        self.modules["count"] = self.count
        self.modules["equal_color"] = self.equal_color
        self.modules["equal_integer"] = self.equal_integer
        self.modules["equal_material"] = self.equal_material
        self.modules["equal_shape"] = self.equal_shape
        self.modules["equal_size"] = self.equal_size
        self.modules["exist"] = self.exist
        self.modules["filter_color[blue]"] = self.filter_blue
        self.modules["filter_color[brown]"] = self.filter_brown
        self.modules["filter_color[cyan]"] = self.filter_cyan
        self.modules["filter_color[gray]"] = self.filter_gray
        self.modules["filter_color[green]"] = self.filter_green
        self.modules["filter_color[purple]"] = self.filter_purple
        self.modules["filter_color[red]"] = self.filter_red
        self.modules["filter_color[yellow]"] = self.filter_yellow
        self.modules["filter_material[rubber]"] = self.filter_rubber
        self.modules["filter_material[metal]"] = self.filter_metal
        self.modules["filter_shape[cube]"] = self.filter_cube
        self.modules["filter_shape[cylinder]"] = self.filter_cylinder
        self.modules["filter_shape[sphere]"] = self.filter_sphere
        self.modules["filter_size[large]"] = self.filter_large
        self.modules["filter_size[small]"] = self.filter_small
        self.modules["greater_than"] = self.greater_than
        self.modules["less_than"] = self.less_than
        self.modules["intersect"] = self.intersect
        self.modules["query_color"] = self.query_color
        self.modules["query_material"] = self.query_material
        self.modules["query_shape"] = self.query_shape
        self.modules["query_size"] = self.query_size
        self.modules["relate[behind]"] = self.relate_behind
        self.modules["relate[front]"] = self.relate_front
        self.modules["relate[left]"] = self.relate_left
        self.modules["relate[right]"] = self.relate_right
        self.modules["same_color"] = self.same_color
        self.modules["same_material"] = self.same_material
        self.modules["same_shape"] = self.same_shape
        self.modules["same_size"] = self.same_size
        self.modules["union"] = self.union
        self.modules["unique"] = self.unique

    def count(self, scene, _):
        if type(scene) == list:
            return len(scene)
        return "error"

    def equal_color(self, color1, color2):
        if (
            type(color1) == str
            and color1 in self.colors
            and type(color2) == str
            and color2 in self.colors
        ):
            if color1 == color2:
                return "yes"
            else:
                return "no"
        return "error"

    def equal_integer(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 == integer2:
                return "yes"
            else:
                return "no"
        return "error"

    def equal_material(self, material1, material2):
        if (
            type(material1) == str
            and material1 in self.materials
            and type(material2) == str
            and material2 in self.materials
        ):
            if material1 == material2:
                return "yes"
            else:
                return "no"
        return "error"

    def equal_shape(self, shape1, shape2):
        if (
            type(shape1) == str
            and shape1 in self.shapes
            and type(shape2) == str
            and shape2 in self.shapes
        ):
            if shape1 == shape2:
                return "yes"
            else:
                return "no"
        return "error"

    def equal_size(self, size1, size2):
        if (
            type(size1) == str
            and size1 in self.sizes
            and type(size2) == str
            and size2 in self.sizes
        ):
            if size1 == size2:
                return "yes"
            else:
                return "no"
        return "error"

    def exist(self, scene, _):
        if type(scene) == list:
            if len(scene) != 0:
                return "yes"
            else:
                return "no"
        return "error"

    def filter_blue(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "blue":
                    output.append(o)
            return output
        return "error"

    def filter_brown(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "brown":
                    output.append(o)
            return output
        return "error"

    def filter_cyan(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "cyan":
                    output.append(o)
            return output
        return "error"

    def filter_gray(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "gray":
                    output.append(o)
            return output
        return "error"

    def filter_green(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "green":
                    output.append(o)
            return output
        return "error"

    def filter_purple(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "purple":
                    output.append(o)
            return output
        return "error"

    def filter_red(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "red":
                    output.append(o)
            return output
        return "error"

    def filter_yellow(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == "yellow":
                    output.append(o)
            return output
        return "error"

    def filter_rubber(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["material"] == "rubber":
                    output.append(o)
            return output
        return "error"

    def filter_metal(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["material"] == "metal":
                    output.append(o)
            return output
        return "error"

    def filter_cube(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["shape"] == "cube":
                    output.append(o)
            return output
        return "error"

    def filter_cylinder(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["shape"] == "cylinder":
                    output.append(o)
            return output
        return "error"

    def filter_sphere(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["shape"] == "sphere":
                    output.append(o)
            return output
        return "error"

    def filter_large(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["size"] == "large":
                    output.append(o)
            return output
        return "error"

    def filter_small(self, scene, _):
        if type(scene) == list:
            output = []
            for o in scene:
                if o["size"] == "small":
                    output.append(o)
            return output
        return "error"

    def greater_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 > integer2:
                return "yes"
            else:
                return "no"
        return "error"

    def less_than(self, integer1, integer2):
        if type(integer1) == int and type(integer2) == int:
            if integer1 < integer2:
                return "yes"
            else:
                return "no"
        return "error"

    def intersect(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = []
            for o in scene1:
                if o in scene2:
                    output.append(o)
            return output
        return "error"

    def query_color(self, obj, _):
        if type(obj) == dict and "color" in obj:
            return obj["color"]
        return "error"

    def query_material(self, obj, _):
        if type(obj) == dict and "material" in obj:
            return obj["material"]
        return "error"

    def query_shape(self, obj, _):
        if type(obj) == dict and "shape" in obj:
            return obj["shape"]
        return "error"

    def query_size(self, obj, _):
        if type(obj) == dict and "size" in obj:
            return obj["size"]
        return "error"

    def relate_behind(self, obj, scene):
        if type(obj) == dict and "position" in obj and type(scene) == list:
            output = []
            for o in scene:
                if o["position"][1] < obj["position"][1]:
                    output.append(o)
            return output
        return "error"

    def relate_front(self, obj, scene):
        if type(obj) == dict and "position" in obj and type(scene) == list:
            output = []
            for o in scene:
                if o["position"][1] > obj["position"][1]:
                    output.append(o)
            return output
        return "error"

    def relate_left(self, obj, scene):
        if type(obj) == dict and "position" in obj and type(scene) == list:
            output = []
            for o in scene:
                if o["position"][0] < obj["position"][0]:
                    output.append(o)
            return output
        return "error"

    def relate_right(self, obj, scene):
        if type(obj) == dict and "position" in obj and type(scene) == list:
            output = []
            for o in scene:
                if o["position"][0] > obj["position"][0]:
                    output.append(o)
            return output
        return "error"

    def same_color(self, obj, scene):
        if type(obj) == dict and "color" in obj and type(scene) == list:
            output = []
            for o in scene:
                if o["color"] == obj["color"] and o["id"] != obj["id"]:
                    output.append(o)
            return output
        return "error"

    def same_material(self, obj, scene):
        if type(obj) == dict and "material" in obj and type(scene) == list:
            output = []
            for o in scene:
                if o["material"] == obj["material"] and o["id"] != obj["id"]:
                    output.append(o)
            return output
        return "error"

    def same_shape(self, obj, scene):
        if type(obj) == dict and "shape" in obj and type(scene) == list:
            output = []
            for o in scene:
                if o["shape"] == obj["shape"] and o["id"] != obj["id"]:
                    output.append(o)
            return output
        return "error"

    def same_size(self, obj, scene):
        if type(obj) == dict and "size" in obj and type(scene) == list:
            output = []
            for o in scene:
                if o["size"] == obj["size"] and o["id"] != obj["id"]:
                    output.append(o)
            return output
        return "error"

    def union(self, scene1, scene2):
        if type(scene1) == list and type(scene2) == list:
            output = list(scene2)
            for o in scene1:
                if o not in scene2:
                    output.append(o)
            return output
        return "error"

    def unique(self, scene, _):
        if type(scene) == list and len(scene) > 0:
            return scene[0]
        return "error"
