from typing import Tuple, List, Optional, Iterable, Dict
import random
import os
import tempfile
from abc import ABC, abstractmethod
from copy import deepcopy
import math
from collections import Counter
from collections import namedtuple
import json
import sys
from datetime import datetime as dt
import copy

# Note: import bpy & mathutils is only going to work when run by Blender python
INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
    if __name__ == "__main__":
        print("This script is intended to be called from blender like this:")
        print()
        print("blender --background --python sg_utils.py")
    else:
        print(
            "sg_utils must be imported by blender python; blender --background --python "
        )
    sys.exit(1)
if INSIDE_BLENDER:
    try:
        import atom_utils
        from sg_question_utils import generate_questions
    except ImportError as e:
        print("\nERROR")
        print(
            "Running render_images.py from Blender and cannot import atom_utils.py or sg_questions_utils."
        )
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print(
            "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/atom_clevr.pth"
        )
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

CLEVR_COLORS = ["blue", "brown", "cyan", "gray", "green", "purple", "red", "yellow"]
CLEVR_MATERIALS = ["rubber", "metal"]
CLEVR_SHAPES = ["cube", "cylinder", "sphere"]
CLEVR_SIZES = ["large", "small"]

properties = {
    "shapes": {
        "cube": "SmoothCube_v2",
        "sphere": "Sphere",
        "cylinder": "SmoothCylinder",
    },
    "sizes": {"large": 0.7, "small": 0.35},
}

# If true, include more information in the scene graph (directions & camera position)
DEBUG_VERBOSE_SG = False


# Function taken from original CLEVR generation code.
def rand(L):
    return 2.0 * L * (random.random() - 0.5)


class Variant:
    """
    A class storing the position of 10 objects, and the scene's camera position
    & lighting. The intent of this class is that this information be used
    to create a complete scene graph (i.e., by specifying each object's
    attributes; namely color, shape, size & material).

    # 3D offsets for the camera, and each of the light sources.
    camera_jitter: Tuple[float, float, float]
    lamp_key: Tuple[float, float, float]
    lamp_back: Tuple[float, float, float]
    lamp_fill: Tuple[float, float, float]

    # 3D x & y coordinates of each object. Note that the z-coordinate is
    # implicit (i.e., whatever rests the object on top of the plane).
    # Note that these are the 3D coordinates used in blender,
    # They are NOT the pixel positions.
    # Note also that the z-coordinate cannot be specified here, as it varies
    # based on the object's shape & size.
    # I *believe* the z-position only depends on the object's size,
    # 0.3499999940395355 = {'small'}
    # 0.699999988079071 = {'large'}


    positions: List[Tuple[float, float]]

    # Rotation for each object
    thetas: List[float]
    """

    def __init__(
        self,
        camera_jitter: Optional[Tuple[float, float, float]] = None,
        lamp_key: Optional[Tuple[float, float, float]] = None,
        lamp_back: Optional[Tuple[float, float, float]] = None,
        lamp_fill: Optional[Tuple[float, float, float]] = None,
    ):
        super().__init__()

        # If the user-provided value is not None then return that, otherwise
        # generate a random 3-tuple of jitter.
        def get_jitter(input_val, L):
            if input_val is not None:
                return input_val
            else:
                return tuple((rand(L) for i in range(3)))

        # Same default random jitter amounts as original CLEVR generation script
        self.camera_jitter = get_jitter(camera_jitter, 0.5)
        self.lamp_key = get_jitter(lamp_key, 1.0)
        self.lamp_back = get_jitter(lamp_back, 1.0)
        self.lamp_fill = get_jitter(lamp_fill, 1.0)

        self.positions = []
        self.thetas = []

    def __str__(self):
        return json.dumps(
            {
                "camera_jitter": self.camera_jitter,
                "lamp_key": self.lamp_key,
                "lamp_back": self.lamp_back,
                "lamp_fill": self.lamp_fill,
                "positions": self.positions,
                "thetas": self.thetas,
            },
            indent=2,
        )

    def add_object(
        self, pos: Optional[Tuple[float, float]] = None, theta: Optional[float] = None
    ):
        if theta is None:
            theta = 360.0 * random.random()

        if pos is None:
            pos = (random.uniform(-3, 3), random.uniform(-3, 3))

        self.thetas.append(theta)
        self.positions.append(pos)
        assert len(self.thetas) == len(self.positions)

    def pop_object(self):
        self.thetas.pop()
        self.positions.pop()
        assert len(self.thetas) == len(self.positions)

    #       If this is too slow, then re-design so that resetting blender isn't req'd.
    #       Or have the variant alter Blender, though that's dangerous.
    #       Probably best for this to just not be a method.
    def simple_check(self):
        """
        Check we meet CLEVR restrictions on object closeness.

        Check that, if all shapes are the same size & shape, then we pass
        CLEVR occlusion restrictions.
        :return:
        """
        pass


class SceneGraph:
    def __init__(
        self,
        camera_jitter: List[float] = None,
        lamp_key: List[float] = None,
        lamp_back: List[float] = None,
        lamp_fill: List[float] = None,
        json_str: str = None,
    ):
        super().__init__()

        self.sg = {
            "objects": [],
            "camera_jitter": camera_jitter if camera_jitter is not None else [0, 0, 0],
            "Lamp_Key": lamp_key if lamp_key is not None else [0, 0, 0],
            "Lamp_Back": lamp_back if lamp_back is not None else [0, 0, 0],
            "Lamp_Fill": lamp_fill if lamp_fill is not None else [0, 0, 0],
        }

        if json_str is not None:
            self.sg = json.loads(json_str)
            assert isinstance(self.sg, list)

    def __iter__(self):
        return self.sg["objects"].__iter__()

    def __getitem__(self, item):
        return self.sg["objects"].__getitem__(item)

    def __len__(self):
        return len(self.sg["objects"])

    def __repr__(self):
        return "SceneGraph('%s')" % self._json()

    def _json(self):
        # Sorted by key for ease of deduplication
        return json.dumps(self.sg, sort_keys=True)

    def add_objects(
        self,
        positions: List[Tuple[float, float]],
        thetas: List[float],
        sizes: List[str],
        materials: Optional[List[str]],
        shapes: List[str],
        colors: Optional[List[str]],
    ):
        """
        Add the provided objects to the scene graph.

        NOTE: When creating from a variant, ideally keep object order as in the
              variant. Thank you.

        :param positions: (x,y) positions of the objects. z is ommited as it
                          depends only on object size.
        :param thetas: Rotations of each object
        :param sizes: 'large' or 'small' for each object
        :param materials: 'rubber' or 'metal' for each object. Can be None, in
                          which case the value is stored as None. This is used
                          for render scenegraphs that are used to check object
                          occlusion; here material does not matter.
        :param shapes: 'cube', 'sphere', or 'cylinder' for each object
        :param colors: 8 possible colours, one per object. Can be None, in which
                       case the value is stored as None. This is used
                          for render scenegraphs that are used to check object
                          occlusion; here color does not matter.
        :return:
        """
        if materials is None:
            materials = [None] * len(thetas)
        if colors is None:
            colors = [None] * len(thetas)

        for (x, y), t, siz, mat, sha, col in zip(
            positions, thetas, sizes, materials, shapes, colors
        ):
            self.sg["objects"].append(
                {
                    "color": col,
                    "size": siz,
                    "rotation": t,
                    "shape": sha,
                    "3d_coords": [
                        x,
                        y,
                        -1,  # NOTE: This is not the correct z-coordinate; that must be computed with blender
                    ],
                    "material": mat,
                }
            )


Directions = namedtuple("Directions", ["behind", "front", "left", "right"])


class AbstractCLEVRTestCase(ABC):
    @abstractmethod
    def must_reject_variant(self, variant: Variant, directions: Directions) -> bool:
        """
        Returns true if it is impossible to add objects to this variant
        such that it works for this test case.

        e.g., if the test case requires very specific balancing of spatial
        properties, then this test case could reject a variant.

        PRECONDITION: This method will be called every time an object is added.
        (i.e., if variant contains 5 objects, then this method has already been
        called when it had 0, 1, 2, 3, and 4 objects).

        NOTE: This method does not check the CLEVR restrictions on
        occlusion/spacing between objects, etc...
        This method should not use Blender.
        NOTE: actually, it might be able to if setup properly. But for now,
        please don't.

        :param variant:
        :param directions:
        :return:
        """
        return False

    @abstractmethod
    def generate_full_sgs(
        self,
        variant: Variant,
        variant_id: int,
        directions: Directions,
        num_objs: Optional[int] = None,
    ) -> Tuple[
        List[str], List[str], Dict[str, List[dict]]
    ]:  # -> Tuple[List[SceneGraph], List[str], Dict[SceneGraph, List[dict]]]:
        """
        Generate all the images for this variant (if num_objs isn't none, then just
        that many). Also return name for each scenegraph.

        Recommended filename structure is "CLEVR_add_id_r%04d_n%04d_v%03d"
        i.e. CLEVR_{description of test case}_r{row #}_n{column $}_v{variand_id}

        NOTE: probably want to add index? Needs to be done outside of here though
              due to de-duplication.
              MUST GO AT THE END, i.e. _{idx}.png else CLEVR scripts will need modifying.

        NOTE: z-coordinates of objects in the SceneGraphs should be fixed to -1.

        Where rows are over the irrelevant factors, and columns are over
        the important changes. e.g., for counting, rows are over the
        shape/color/size/material, and columns are over the number of objects
        in the scene.

        The description of the test case is probably best editable via
        __init__ (e.g., if you want to add a split after the description,
        e.g. {_train_add_id}

        WARNING: Might contain duplicates.

        Note: may be easier to implement by calling generate_render_sgs, and
              then add in remaining attributes.

        :param variant:
        :return: A list of scene graphs that need to be generated, and a
        list of filenames for the images that the scenegraphs will create.
        The strings are just the .json representations of the SceneGraphs.
        Because using sets on mutable objects is a dumpster fire next to a
        fireworks factory under unfavourable wind conditions.
        Yes, I know using strings as datastructure is unholy. May the gods
        have mercy on my soul.
        We also return a dictionary from the string .json representations
        of the SceneGraphs to a list of corresponding questions.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_render_sgs(
        self, variant: Variant, variant_id: int, num_objs: int, directions: Directions
    ) -> List[SceneGraph]:
        """
        Generate the *test* images for this variant, which use only the first
        num_objs objects. Specifically, color & material are kept as None as
        they do not impact occlusion/spacing tests.

        The generated scenegraphs only contain shape, size, x/y coordinates,
        and rotation.

        NOTE: z-coordinates of objects in the SceneGraphs should be fixed to -1.

        :param variant:
        :return:
        """
        raise NotImplementedError


class AllIdenticalObjs(AbstractCLEVRTestCase):
    """
    Example test case, which is just of all objects being identical, over
    all possible shape/size/material/color combinations.
    """

    def __init__(
        self,
        filename_template: str = "CLEVR_AllIdent_r%03d_v%02d_n%03d",
        question_template: str = "CLEVR_1.0_atom_templates/count.json",
    ):
        super().__init__()
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
        positions = variant.positions[:num_objs]
        thetas = variant.thetas[:num_objs]
        assert len(positions) == num_objs

        results = []
        for size in CLEVR_SIZES:
            for shape in CLEVR_SHAPES:
                sg = SceneGraph(
                    camera_jitter=variant.camera_jitter,
                    lamp_back=variant.lamp_back,
                    lamp_fill=variant.lamp_fill,
                    lamp_key=variant.lamp_key,
                )
                sg.add_objects(
                    positions=positions,
                    thetas=thetas,
                    sizes=[size] * num_objs,
                    materials=None,
                    shapes=[shape] * num_objs,
                    colors=None,
                )
                results.append(sg)
        return results

    def generate_full_sgs(
        self,
        variant: Variant,
        variant_id: int,
        num_objs: Optional[int] = None,
        directions: Directions = None,
    ) -> Tuple[
        List[str], List[str], Dict[str, List[dict]]
    ]:  # -> Tuple[List[SceneGraph], List[str], Dict[SceneGraph, List[dict]]]:
        results = []
        filenames = []
        questions = {}

        if num_objs is not None:
            num_objs = [num_objs]
        else:
            num_objs = [i for i in range(len(variant.positions) + 1)]

        for num_obj in num_objs:
            render_sgs = self.generate_render_sgs(variant, variant_id, num_obj)

            # For each render scenegraph, expand it so that every object has
            # the same colour & material.
            row_num = 0
            for (
                render_sg
            ) in (
                render_sgs
            ):  # Each is a different row here; assumes generate_render_sgs is consistent with row order
                for color in CLEVR_COLORS:
                    for mat in CLEVR_MATERIALS:
                        final_sg = deepcopy(render_sg)
                        for obj in final_sg:
                            assert obj["color"] is None
                            assert obj["material"] is None
                            obj["color"] = color
                            obj["material"] = mat

                        # Prevent mutation by now switching to immutable string
                        final_sg_string = final_sg._json()

                        # Note this causes duplicates.
                        results.append(final_sg_string)
                        imgs_in_row = num_obj
                        filenames.append(
                            self.filename_template % (row_num, variant_id, imgs_in_row)
                        )

                        # Make sure not to overwrite any questions we've already created for this SG if it's come up before
                        if final_sg_string not in questions:
                            questions[final_sg_string] = []

                        # NOTE: Could just do this once for count as question does not change with image here.
                        # NOTE: Fixed seed to the variant idx for reproducibility
                        tmp_questions = generate_questions(
                            scene_graphs=[
                                final_sg.sg
                            ],  # NOTE: This *may cause mutation*!
                            template=self.question_template,
                            template_name=os.path.basename(self.question_template_path),
                            seed=variant_id,
                        )
                        assert len(tmp_questions) == 1
                        tmp_questions[0]["variant_index"] = variant_id
                        tmp_questions[0]["row_index"] = row_num
                        tmp_questions[0]["col_index"] = imgs_in_row

                        questions[final_sg_string] += tmp_questions

                        row_num += 1

        assert len(results) == len(filenames)
        return results, filenames, questions


class MinimalExistsTestCase(AbstractCLEVRTestCase):
    """
    Example test case, which is just of all objects being identical, over
    all possible shape/size/material/color combinations.
    """

    def __init__(
        self,
        filename_template: str = "CLEVR_MinimalExist_r%03d_v%02d_n%03d",
        question_template: str = "CLEVR_1.0_atom_templates/exist.json",
    ):
        super().__init__()
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
        if num_objs not in [0, 1]:
            return []
        else:
            positions = variant.positions[:num_objs]
            thetas = variant.thetas[:num_objs]
            assert len(positions) == num_objs

            results = []
            for size in CLEVR_SIZES:
                for shape in CLEVR_SHAPES:
                    sg = SceneGraph(
                        camera_jitter=variant.camera_jitter,
                        lamp_back=variant.lamp_back,
                        lamp_fill=variant.lamp_fill,
                        lamp_key=variant.lamp_key,
                    )
                    sg.add_objects(
                        positions=positions,
                        thetas=thetas,
                        sizes=[size] * num_objs,
                        materials=None,
                        shapes=[shape] * num_objs,
                        colors=None,
                    )
                    results.append(sg)
            return results

    def generate_full_sgs(
        self,
        variant: Variant,
        variant_id: int,
        num_objs: Optional[int] = None,
        directions: Directions = None,
    ) -> Tuple[
        List[str], List[str], Dict[str, List[dict]]
    ]:  # -> Tuple[List[SceneGraph], List[str], Dict[SceneGraph, List[dict]]]:
        results = []
        filenames = []
        questions = {}

        if num_objs is not None:
            num_objs = [num_objs]
        else:
            num_objs = [0, 1]

        for num_obj in num_objs:
            render_sgs = self.generate_render_sgs(variant, variant_id, num_obj)

            # For each render scenegraph, expand it so that every object has
            # the same colour & material.
            row_num = 0
            for (
                render_sg
            ) in (
                render_sgs
            ):  # Each is a different row here; assumes generate_render_sgs is consistent with row order
                for color in CLEVR_COLORS:
                    for mat in CLEVR_MATERIALS:
                        final_sg = deepcopy(render_sg)
                        for obj in final_sg:
                            assert obj["color"] is None
                            assert obj["material"] is None
                            obj["color"] = color
                            obj["material"] = mat

                        # Prevent mutation by now switching to immutable string
                        final_sg_string = final_sg._json()

                        # Note this causes duplicates.
                        results.append(final_sg_string)
                        imgs_in_row = num_obj
                        filenames.append(
                            self.filename_template % (row_num, variant_id, imgs_in_row)
                        )

                        # Make sure not to overwrite any questions we've already created for this SG if it's come up before
                        if final_sg_string not in questions:
                            questions[final_sg_string] = []

                        # NOTE: Could just do this once for exist as question does not change with image here.
                        # NOTE: Fixed seed to the variant idx for reproducibility
                        tmp_questions = generate_questions(
                            scene_graphs=[
                                final_sg.sg
                            ],  # NOTE: This *may cause mutation*!
                            template=self.question_template,
                            template_name=os.path.basename(self.question_template_path),
                            seed=variant_id,
                        )
                        assert len(tmp_questions) == 1
                        tmp_questions[0]["variant_index"] = variant_id
                        tmp_questions[0]["row_index"] = row_num
                        tmp_questions[0]["col_index"] = imgs_in_row

                        questions[final_sg_string] += tmp_questions

                        row_num += 1

        assert len(results) == len(filenames)
        return results, filenames, questions


def setup_blender(
    variant: Variant, use_gpu: bool = True, img_width: int = 480, img_height: int = 320
) -> Directions:
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath="data/base_scene.blend")
    # print(list(bpy.data.objects['Camera'].location))
    # if random.random() > 0.99:
    #    raise ZeroDivisionError

    # Load materials
    atom_utils.load_materials("data/materials")

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = "dummy_string.png"
    render_args.resolution_x = img_width
    render_args.resolution_y = img_height
    render_args.resolution_percentage = 100
    render_args.tile_x = 256
    render_args.tile_y = 256
    if use_gpu:
        # Blender changed the API for enabling CUDA at some point
        if bpy.app.version < (2, 78, 0):
            bpy.context.user_preferences.system.compute_device_type = "CUDA"
            bpy.context.user_preferences.system.compute_device = "CUDA_0"
        else:
            cycles_prefs = bpy.context.user_preferences.addons["cycles"].preferences
            cycles_prefs.compute_device_type = "CUDA"

    # Some CYCLES-specific stuff
    bpy.data.worlds["World"].cycles.sample_as_light = True
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = 512
    bpy.context.scene.cycles.transparent_min_bounces = 8
    bpy.context.scene.cycles.transparent_max_bounces = 8
    if use_gpu:
        bpy.context.scene.cycles.device = "GPU"

    # Maintain same order: Place plane, jitter camera, get directions:

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    # Apply the camera jitter (lighting jitter isn't needed for the render)
    for i in range(3):
        bpy.data.objects["Camera"].location[i] += variant.camera_jitter[i]

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects["Camera"]
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    atom_utils.delete_object(plane)

    # Save all four axis-aligned directions (ignoring up/down)
    return Directions(
        behind=tuple(plane_behind),
        front=tuple(-plane_behind),
        left=tuple(plane_left),
        right=tuple(-plane_left),
    )


# Modified from CLEVR code
def check_visibility(blender_objects, min_pixels_per_object: int = 200):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Precondition: blender has already been set to the correct image dimensions

    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix=".png")
    render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter(
        (p[i], p[i + 1], p[i + 2], p[i + 3]) for i in range(0, len(p), 4)
    )
    os.close(f)
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            return False
    return True


# Modified from CLEVR code
def render_shadeless(blender_objects, path="flat.png"):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render
    render_args.filepath = path

    # Add random shadeless materials to all objects
    object_colors = set()
    for i, obj in enumerate(blender_objects):
        bpy.ops.material.new()
        mat = bpy.data.materials["Material"]
        mat.name = "Material_%d" % i
        while True:
            r, g, b = [random.random() for _ in range(3)]
            if (r, g, b) not in object_colors:
                break
        object_colors.add((r, g, b))
        mat.diffuse_color = [r, g, b]
        mat.use_shadeless = True
        obj.data.materials[0] = mat

    # Render the scene
    bpy.ops.render.render(write_still=True)
    return object_colors


def pass_occlusion_test(
    variant: Variant,
    to_render: Iterable[SceneGraph],
    min_pixels_per_object: int = 200,
    img_width: int = 480,
    img_height: int = 320,
):
    """
    Precondition: variant & all scenegraphs in to_render have the exact same
    camera & lighting jitters.

    :param variant:
    :param to_render:
    :return:
    """
    for render_sg in to_render:
        # print("next render")

        assert variant.camera_jitter == render_sg.sg["camera_jitter"]
        assert variant.lamp_key == render_sg.sg["Lamp_Key"]
        assert variant.lamp_fill == render_sg.sg["Lamp_Fill"]
        assert variant.lamp_back == render_sg.sg["Lamp_Back"]

        # Note: might be able to speed up by editing objects instead of deleting & re-creating
        setup_blender(
            variant, img_width=img_width, img_height=img_height
        )  # Get blender ready to render test image, and apply camera jitter

        blender_objs = []
        for obj in render_sg:
            # Get size & shape, in the form that Blender understands
            r = properties["sizes"][obj["size"]]
            shape = properties["shapes"][obj["shape"]]
            x, y, _ = obj["3d_coords"]
            theta = obj["rotation"]

            # NOTE: Clevr uses 'SmoothCube_v2', therefore the adjustment
            # below should never happen.
            # Also, 'Cube' was the name of the blender file, not the human
            # readable equivalent, so this if was buggy anyways.
            # Adjust size for cubes (as per CLEVR code)
            # if obj['shape'] == 'Cube':
            #    r /= math.sqrt(2)

            atom_utils.add_object(
                "data/shapes", shape, r, (x, y), theta=theta, original_translate=False
            )
            blender_objs.append(bpy.context.object)
            # Add dummy material; doesn't matter
            atom_utils.add_material(
                "Rubber", Color=[87.0 / 255.0, 87.0 / 255.0, 87.0 / 255.0, 1.0]
            )

        # Switch Blender to rendering mode & hide the lights/floor
        # Override some render settings to have flat shading
        render_args = bpy.context.scene.render
        render_args.engine = "BLENDER_RENDER"
        render_args.use_antialiasing = False

        # Move the lights and ground to layer 2 so they don't render
        atom_utils.set_layer(bpy.data.objects["Lamp_Key"], 2)
        atom_utils.set_layer(bpy.data.objects["Lamp_Fill"], 2)
        atom_utils.set_layer(bpy.data.objects["Lamp_Back"], 2)
        atom_utils.set_layer(bpy.data.objects["Ground"], 2)

        if not check_visibility(
            blender_objs, min_pixels_per_object=min_pixels_per_object
        ):
            return False
    return True


def generate_variant(
    test_cases: List[AbstractCLEVRTestCase],
    variant_id: int,
    desired_num_objects: int = 10,
    min_pixels_per_object: int = 200,
    img_width: int = 480,
    img_height: int = 320,
):
    """
    NOTE: Kinda ugly/hacky since will reject until a variant passing all test
       cases in test_cases is created.
       This can result in an infinite loop if you're not careful (e.g., if
       the test_cases are looking for mutually exclusive properties.)
       Could fix/expand by later on going through accepted variants & re-using
       as many as possible.

    :param test_cases:
    :return:
    """
    camera_debug = []

    def pass_test_cases(variant: Variant, directions: Directions):
        for tc in test_cases:
            if tc.must_reject_variant(variant=variant, directions=directions):
                return False
        return True

    # Add an object, check if it passes the CLEVR object spacing & occlusion
    # requriements, as well as any other checks the test cases must perform.
    # If the new object fails, remove it.
    # If we fail to add a new object 50 times, clear the scene and start
    # from scratch

    def reset_variant():
        # Create initial empty scene
        wip_var = Variant()
        directions = setup_blender(wip_var, img_width=img_width, img_height=img_height)
        passing_objects = 0
        # Check the empty variant (i.e., random jitters) are acceptable
        # for the tests.
        while not pass_test_cases(wip_var, directions):
            wip_var = Variant()
            directions = setup_blender(
                wip_var, img_width=img_width, img_height=img_height
            )
        object_add_tries = 0
        return wip_var, passing_objects, directions, object_add_tries

    while True:
        # Create initial empty scene
        wip_var, passing_objects, directions, object_add_tries = reset_variant()

        while (
            passing_objects < desired_num_objects
        ):  # Must add desired_num_objects objects
            # print("Objects: %i  Tries: %i"%(passing_objects, object_add_tries))

            # If we've failed to add a new object 50 times, then wipe all & restart.
            if object_add_tries >= 50:
                wip_var, passing_objects, directions, object_add_tries = reset_variant()

            # Add the new object
            wip_var.add_object()
            object_add_tries += 1

            # Check distance between the objects is acceptable:
            (x, y) = wip_var.positions[-1]
            reject = False
            for xx, yy in wip_var.positions[:-1]:
                # Note: we assume here that the objects are at maximum width
                #       this will cause a slight change in distribution compared
                #       to clevr.
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                # Note 0.7 here is a magic constant for shape size; 0.7 is
                # the largest possible value.
                # 0.25 is a magic constant for min-distance; again from the
                # CLEVR code.
                if dist - 0.7 - 0.7 < 0.25:
                    reject = True
                    break  # Stop checking distances, reject this object placement & retry

            if reject:
                wip_var.pop_object()
                continue

            # Check margins along cardinal directions between objects
            for xx, yy in wip_var.positions[:-1]:
                dx, dy = x - xx, y - yy
                for direction_name in ["left", "right", "front", "behind"]:
                    camera_debug = list(bpy.data.objects["Camera"].location)
                    direction_vec = getattr(directions, direction_name)
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < 0.4:  # args.margin from original CLEVR script
                        reject = True
                        break
                if reject:
                    break

            if reject:
                wip_var.pop_object()
                continue

            # We've passed the spacing check. Next, if our test cases don't
            # approve, then remove the object & retry.
            if not pass_test_cases(wip_var, directions):
                wip_var.pop_object()
                continue

            # Passed all tests except occlusion (this test is slow, so it's left
            # to the end after all other tests pass)
            passing_objects += 1
            object_add_tries = 0

        # Finally, check occlusion by rendering the images
        reject = False
        for num_obj in [x for x in range(1, desired_num_objects + 1)][::-1]:
            # Sets don't work properly with SceneGraph objects, so instead
            # convert them into strings & deduplicate it that way.
            # _json() sorts by key, and objects remain in the same order as
            # the variant, so this should actually work. Maybe. Hopefully.
            to_render = set()
            to_render_dedup = set()
            for tc in test_cases:
                for r_sg in tc.generate_render_sgs(
                    wip_var, variant_id, num_obj, directions=directions
                ):
                    if r_sg._json() not in to_render_dedup:
                        to_render_dedup.add(r_sg._json())
                        to_render.add(r_sg)
            # print("NUMBER OF SCENES BEING RENDERED")
            # print(len(to_render))
            # print(num_obj)
            if not pass_occlusion_test(
                variant=wip_var,
                to_render=to_render,
                min_pixels_per_object=min_pixels_per_object,
                img_width=img_width,
                img_height=img_height,
            ):
                reject = True
                break

        if not reject:  # Finally done.
            break

    return wip_var, directions, camera_debug


def generate_sg_and_questions(
    test_cases: List[Tuple[AbstractCLEVRTestCase, int]],
    split: str = "val",
    desired_num_objects: int = 10,
    min_pixels_per_object: int = 200,
    img_width: int = 480,
    img_height: int = 320,
    version: float = 1.0,
) -> Tuple[dict, List[dict], List[Variant]]:
    """
    Generate a separate questions .json for each test case in test_cases.
    Generate global scene graphs specification (NOTE: this is the input for
    the render_images_from_sg.py script; *NOT* the final scene graphs).

    :param split: Name of the split
    :param test_cases: List of (TestCase, requested # of variants) tuples.
    :return:
    """
    scene_graphs = set()
    final_variants = []
    scene_graph_names = {}
    if DEBUG_VERBOSE_SG:
        scene_graph_dirs = {}
        scene_graph_cam = {}

    header = {
        "info": {
            "date": dt.today().strftime("%m/%d/%Y"),
            "version": version,
            "split": split,
            "license": "Creative Commons Attribution (CC-BY 4.0)",
        },
        "questions": [],
    }

    final_questions = [
        copy.deepcopy(header) for i in range(len(test_cases))
    ]  # Each test case gets its own dict of questions.

    total_variants = max(x[1] for x in test_cases)
    curr_img_index = 0
    for var_idx in range(total_variants + 1):
        # Get index of test case so we put it in the right place in final_questions (since the test cases in use may vary)
        curr_test_cases = [
            (i, x[0]) for (i, x) in enumerate(test_cases) if var_idx < x[1]
        ]
        variant, directions_vec, cam_vec_tmp_debug = generate_variant(
            test_cases=[x[1] for x in curr_test_cases],
            variant_id=var_idx,
            desired_num_objects=desired_num_objects,
            min_pixels_per_object=min_pixels_per_object,
            img_width=img_width,
            img_height=img_height,
        )
        final_variants.append(variant)

        for i, tc in curr_test_cases:
            tc_scenes, tc_filenames, tc_questions = tc.generate_full_sgs(
                variant, variant_id=var_idx, directions=directions_vec
            )
            for scene, filename in zip(tc_scenes, tc_filenames):
                if scene not in scene_graphs:
                    # If this is a new scene, add it to the set & record its name
                    scene_graphs.add(scene)
                    scene_graph_names[scene] = (
                        filename + "_%06d.png" % curr_img_index,
                        curr_img_index,
                    )
                    if DEBUG_VERBOSE_SG:
                        scene_graph_dirs[scene] = directions_vec
                        scene_graph_cam[scene] = cam_vec_tmp_debug
                    curr_img_index += 1
            for scene in tc_questions:
                for question in tc_questions[scene]:
                    # Set the image filename & index in the question
                    # dictionary, now that we know these values.
                    img_filename, img_idx = scene_graph_names[scene]
                    question["image_filename"] = img_filename
                    question["image_index"] = img_idx
                    question["image"] = os.path.splitext(img_filename)[0]
                    question["split"] = split
                    final_questions[i]["questions"].append(question)

    # merge, format & return questions & scene graphs.
    final_scene_graphs = copy.deepcopy(header)
    final_scene_graphs.pop("questions", None)
    final_scene_graphs["scenes"] = []
    for sg in scene_graphs:
        img_filename, img_idx = scene_graph_names[sg]
        # Don't need to include directions & relationships as those will
        # be re-computed when the images are rendered, and the final
        # scene_graph .json file is created.
        sg_json = json.loads(sg)
        sg_json["image_index"] = img_idx
        sg_json["image_filename"] = img_filename
        sg_json["split"] = split
        if DEBUG_VERBOSE_SG:
            tmp_dir_vec = scene_graph_dirs[sg]
            sg_json["debug_int_dir"] = {
                "right": tmp_dir_vec.right,
                "left": tmp_dir_vec.left,
                "behind": tmp_dir_vec.behind,
                "front": tmp_dir_vec.front,
            }
            sg_json["debug_cam_loc"] = scene_graph_cam[sg]
        final_scene_graphs["scenes"].append(sg_json)
    # Image index is a terrible way to sort scene graphs (filename would be better)
    # but we have no choice due to CLEVR loader requiring this order
    final_scene_graphs["scenes"].sort(key=lambda x: x["image_index"])
    for tc_dict in final_questions:
        tc_dict["questions"].sort(
            key=lambda x: (x["row_index"], x["variant_index"], x["col_index"])
        )
        question_idx = 0
        for question in tc_dict["questions"]:
            question["question_index"] = question_idx
            question_idx += 1

    return final_scene_graphs, final_questions, final_variants


def add_sg_and_questions(
    test_cases: List[Tuple[AbstractCLEVRTestCase, int]],
    variants_path: str,
    generation_sg_path: str,
    desired_num_objects: int = 9,
    min_pixels_per_object: int = 200,
    img_width: int = 480,
    img_height: int = 320,
    split: str = "val",
    version: float = 1.1,
) -> Tuple[dict, List[dict]]:
    """
    A modification of generate_sg_and_questions which, given a set of test cases,
    creates the new images & scenegraphs required.

    Assumption: None of the test_cases require the directions for must_reject_variant
                or generate_full_sgs or generate_render_sgs
    Precondition: The dataset being added to has as many if not more objects
                  than the number of objects required by the new test case(s).
    Precondition: The dataset being added to has as many if not more variants
                  than the number of variants required by the new test case(s).
    Precondition: All the variants in the dataset being added to meet the
                  acceptability conditions of all the test cases (i.e., none
                  of the test cases will reject a variant in the original dataset)

    Generate a separate questions .json for each test case in test_cases.
    Generate global scene graphs specification (NOTE: this is the input for
    the render_images_from_sg.py script; *NOT* the final scene graphs).

    :param split: Name of the split
    :param variants_path: Path to a .json file containing the list of variants
                          produced by an earlier call to generate_sg_and_questions
                          that created the original dataset
    :param generation_sg_path: Path to a .json file containing the scene graphs
                               produced by an earlier call to generate_sg_and_questions
                               that created the original dataset
    :param test_cases: List of (TestCase, requested # of variants) tuples.
    :return:
    """
    with open(variants_path, "r") as infile:
        dataset_variants = json.load(infile)

    # Names of all existing & created scenegraphs
    all_scene_graph_names = {}
    max_ori_idx = -1

    with open(generation_sg_path, "r") as infile:
        sg_json = json.load(infile)
    sg_json = sg_json["scenes"]

    for sg in sg_json:  # cleanup by removing added cruft
        image_idx = sg.pop("image_index")
        max_ori_idx = max(max_ori_idx, image_idx)
        image_filename = sg.pop("image_filename")
        sg.pop("split")
        sg.pop("debug_int_dir", None)
        sg.pop("debug_cam_loc", None)

        sg_str = json.dumps(
            sg, sort_keys=True
        )  # Note: same as SceneGraph._json(); needs to match so that deduplication works later on
        all_scene_graph_names[sg_str] = (image_filename, image_idx)

    # Here's where we store new scene graphs & their names
    new_scene_graphs = set()

    header = {
        "info": {
            "date": dt.today().strftime("%m/%d/%Y"),
            "version": version,
            "split": split,
            "license": "Creative Commons Attribution (CC-BY 4.0)",
        },
        "questions": [],
    }

    final_questions = [
        copy.deepcopy(header) for i in range(len(test_cases))
    ]  # Each test case gets its own dict of questions.

    total_variants = max(x[1] for x in test_cases)
    assert total_variants <= len(dataset_variants)

    curr_img_index = max_ori_idx + 1
    for var_idx in range(total_variants):
        # Get index of test case so we put it in the right place in final_questions (since the test cases in use may vary)
        curr_test_cases = [
            (i, x[0]) for (i, x) in enumerate(test_cases) if var_idx < x[1]
        ]
        curr_variant_json = dataset_variants[var_idx]

        # Manually check the variants from the old dataset work with the new
        # test case(s).

        print("=" * 80)
        print("CHECKING IF TC MUST REJECT VARIANT %i of %i" % (var_idx, total_variants))
        print("=" * 80)
        # ==================================================================
        # 1) check if tc.must_reject_variant for any of the variants
        tmp_var = Variant(
            camera_jitter=tuple(curr_variant_json["camera_jitter"]),
            lamp_fill=tuple(curr_variant_json["lamp_fill"]),
            lamp_back=tuple(curr_variant_json["lamp_back"]),
            lamp_key=tuple(curr_variant_json["lamp_key"]),
        )

        for i, tc in curr_test_cases:
            if tc.must_reject_variant(variant=tmp_var, directions=None):
                raise ValueError(
                    "The test case %s is incompatible with one of the variants in the dataset, %s"
                    % (str(tc), str(curr_variant_json))
                )

        for pos, theta in zip(
            curr_variant_json["positions"], curr_variant_json["thetas"]
        ):
            tmp_var.add_object(pos=tuple(pos), theta=theta)
            for i, tc in curr_test_cases:
                if tc.must_reject_variant(variant=tmp_var, directions=None):
                    raise ValueError(
                        "The test case %s is incompatible with one of the variants in the dataset, %s"
                        % (str(tc), str(curr_variant_json))
                    )

        print("=" * 80)
        print("CHECKING OCCLUSION ON VARIANT %i of %i" % (var_idx, total_variants))
        print("=" * 80)
        # ===================================================================
        # 2) check if occlusion is a problem for any of the variants
        tmp_var = Variant(
            camera_jitter=tuple(curr_variant_json["camera_jitter"]),
            lamp_fill=tuple(curr_variant_json["lamp_fill"]),
            lamp_back=tuple(curr_variant_json["lamp_back"]),
            lamp_key=tuple(curr_variant_json["lamp_key"]),
        )
        for pos, theta in zip(
            curr_variant_json["positions"], curr_variant_json["thetas"]
        ):
            tmp_var.add_object(pos=tuple(pos), theta=theta)

        to_render = set()
        to_render_dedup = set()
        for i, tc in curr_test_cases:
            for num_obj in [x for x in range(1, desired_num_objects + 1)][::-1]:
                # Sets don't work properly with SceneGraph objects, so instead
                # convert them into strings & deduplicate it that way.
                # _json() sorts by key, and objects remain in the same order as
                # the variant, so this should actually work. Maybe. Hopefully.
                for r_sg in tc.generate_render_sgs(
                    tmp_var, var_idx, num_obj, directions=None
                ):
                    if r_sg._json() not in to_render_dedup:
                        to_render_dedup.add(r_sg._json())
                        to_render.add(r_sg)
        if not pass_occlusion_test(
            variant=tmp_var,
            to_render=to_render,
            min_pixels_per_object=min_pixels_per_object,
            img_width=img_width,
            img_height=img_height,
        ):
            raise ValueError(
                "A test case failed occlusion check with one of the variants in the dataset, %s"
                % str(curr_variant_json)
            )

        print("=" * 80)
        print(
            "CREATE FINAL SCENE GRAPHS FOR VARIANT %i of %i" % (var_idx, total_variants)
        )
        print("=" * 80)
        # Create the scene graphs to be generated, reusing those which already
        # exist.
        tmp_var = Variant(
            camera_jitter=tuple(curr_variant_json["camera_jitter"]),
            lamp_fill=tuple(curr_variant_json["lamp_fill"]),
            lamp_back=tuple(curr_variant_json["lamp_back"]),
            lamp_key=tuple(curr_variant_json["lamp_key"]),
        )
        for pos, theta in zip(
            curr_variant_json["positions"], curr_variant_json["thetas"]
        ):
            tmp_var.add_object(pos=tuple(pos), theta=theta)

        for i, tc in curr_test_cases:
            tc_scenes, tc_filenames, tc_questions = tc.generate_full_sgs(
                tmp_var, variant_id=var_idx, directions=None
            )
            for scene, filename in zip(tc_scenes, tc_filenames):
                if scene not in all_scene_graph_names:
                    # If this is a new scene, add it to the set & record its name
                    new_scene_graphs.add(scene)
                    all_scene_graph_names[scene] = (
                        filename + "_%06d.png" % curr_img_index,
                        curr_img_index,
                    )
                    curr_img_index += 1
            for scene in tc_questions:
                for question in tc_questions[scene]:
                    # Set the image filename & index in the question
                    # dictionary, now that we know these values.
                    img_filename, img_idx = all_scene_graph_names[
                        scene
                    ]  # reuse old scenes if possible
                    question["image_filename"] = img_filename
                    question["image_index"] = img_idx
                    question["image"] = os.path.splitext(img_filename)[0]
                    question["split"] = split
                    final_questions[i]["questions"].append(question)

    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    # merge, format & return questions & scene graphs.
    final_scene_graphs = copy.deepcopy(header)
    final_scene_graphs.pop("questions", None)
    final_scene_graphs["scenes"] = []
    for sg in new_scene_graphs:  # only save generation data for the new scenes
        img_filename, img_idx = all_scene_graph_names[sg]
        sg_json = json.loads(sg)
        sg_json["image_index"] = img_idx
        sg_json["image_filename"] = img_filename
        sg_json["split"] = split
        final_scene_graphs["scenes"].append(sg_json)
    # Image index is a terrible way to sort scene graphs (filename would be better)
    # but we have no choice due to CLEVR loader requiring this order
    final_scene_graphs["scenes"].sort(key=lambda x: x["image_index"])
    for tc_dict in final_questions:
        tc_dict["questions"].sort(
            key=lambda x: (x["row_index"], x["variant_index"], x["col_index"])
        )
        question_idx = 0
        for question in tc_dict["questions"]:
            question["question_index"] = question_idx
            question_idx += 1

    return final_scene_graphs, final_questions
