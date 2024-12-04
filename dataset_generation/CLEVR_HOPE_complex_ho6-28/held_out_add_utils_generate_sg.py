# Modified from original FB code; original FB header below.

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from __future__ import print_function
import math, sys, random, argparse, json, os, tempfile
from datetime import datetime as dt
from collections import Counter
from typing import List, Tuple, Callable, Dict

"""
Modification of original CLEVR code. This script now creates CLEVR-style 
scenegraphs (with camera & lighting jitters also recorded) which can, later,
be used for image generation.

This file expects to be run from Blender like this:

blender --background --python held_out_utils_generate_sg.py -- [arguments to this script]
"""

# This script is mostly for testing.
# Note: 200 pixels/object for 480x320,
#       therefore (320*240)/(480*320) * 200 = 100 pixels/obj after scale down
# blender --background --python render_images_from_sg.py  -- --full-spec --num_images 100 --use_gpu 1 --width 320 --height 240 --render_num_samples 4 --min_pixels_per_object 100 --scene_graphs_path tmp_CLEVR_scenes.json

INSIDE_BLENDER = True
try:
    import bpy, bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
    print("Please run from blender")
    exit(-1)
if INSIDE_BLENDER:
    try:
        import atom_utils as utils
    except ImportError as e:
        print("\nERROR")
        print(
            "Running held_out_utils_generate_sg.py from Blender and cannot import atom_utils.py."
        )
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print(
            "echo {full path to dataset_generation/atom_clevr/} >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/atom_clevr.pth"
        )
        print(
            "easier to change directory to dataset_generation/atom_clevr/ and then run:"
        )
        print(
            "echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/atom_clevr.pth"
        )
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)

parser = argparse.ArgumentParser()

# Input options
parser.add_argument(
    "--base_scene_blendfile",
    default="data/base_scene.blend",
    help="Base blender file on which all scenes are based; includes "
    + "ground plane, lights, and camera.",
)
parser.add_argument(
    "--properties_json",
    default="data/properties.json",
    help="JSON file defining objects, materials, sizes, and colors. "
    + 'The "colors" field maps from CLEVR color names to RGB values; '
    + 'The "sizes" field maps from CLEVR size names to scalars used to '
    + 'rescale object models; the "materials" and "shapes" fields map '
    + "from CLEVR material and shape names to .blend files in the "
    + "--object_material_dir and --shape_dir directories respectively.",
)
parser.add_argument(
    "--shape_dir",
    default="data/shapes",
    help="Directory where .blend files for object models are stored",
)
parser.add_argument(
    "--material_dir",
    default="data/materials",
    help="Directory where .blend files for materials are stored",
)

# Settings for objects
parser.add_argument(
    "--min_objects",
    default=3,
    type=int,
    help="The minimum number of objects to place in each scene",
)
parser.add_argument(
    "--max_objects",
    default=10,
    type=int,
    help="The maximum number of objects to place in each scene",
)
parser.add_argument(
    "--min_dist",
    default=0.25,
    type=float,
    help="The minimum allowed distance between object centers",
)
parser.add_argument(
    "--margin",
    default=0.4,
    type=float,
    help="Along all cardinal directions (left, right, front, back), all "
    + "objects will be at least this distance apart. This makes resolving "
    + "spatial relationships slightly less ambiguous.",
)
parser.add_argument(
    "--min_pixels_per_object",
    default=200,
    type=int,
    help="All objects will have at least this many visible pixels in the "
    + "final rendered images; this ensures that no objects are fully "
    + "occluded by other objects.",
)
parser.add_argument(
    "--max_retries",
    default=50,
    type=int,
    help="The number of times to try placing an object before giving up and "
    + "re-placing all objects in the scene.",
)

# Output settings
parser.add_argument(
    "--start_idx",
    default=0,
    type=int,
    help="The index at which to start for numbering rendered images. Setting "
    + "this to non-zero values allows you to distribute rendering across "
    + "multiple machines and recombine the results later.",
)
parser.add_argument(
    "--num_images", default=5, type=int, help="The number of images to render"
)
parser.add_argument(
    "--filename_prefix",
    default="CLEVR_HeldOut",
    help="This prefix will be prepended to the rendered images and JSON scenes",
)
parser.add_argument(
    "--split",
    default="new",
    help="Name of the split for which we are rendering. This will be added to "
    + "the names of rendered images, and will also be stored in the JSON "
    + "scene structure for each image.",
)
parser.add_argument(
    "--output_scene_file",
    default="tmp_CLEVR_scenes.json",
    help="Path to write a single JSON file containing all scene information",
)
parser.add_argument(
    "--version",
    default="1.0",
    help='String to store in the "version" field of the generated JSON file',
)
parser.add_argument(
    "--license",
    default="Creative Commons Attribution (CC-BY 4.0)",
    help='String to store in the "license" field of the generated JSON file',
)
parser.add_argument(
    "--date",
    default=dt.today().strftime("%m/%d/%Y"),
    help='String to store in the "date" field of the generated JSON file; '
    + "defaults to today's date",
)

# Rendering options
parser.add_argument(
    "--use_gpu",
    default=0,
    type=int,
    help="Setting --use_gpu 1 enables GPU-accelerated rendering using CUDA. "
    + "You must have an NVIDIA GPU with the CUDA toolkit installed for "
    + "to work.",
)
parser.add_argument(
    "--width",
    default=480,
    type=int,
    help="The width (in pixels) for the rendered images",
)
parser.add_argument(
    "--height",
    default=320,
    type=int,
    help="The height (in pixels) for the rendered images",
)
parser.add_argument(
    "--key_light_jitter",
    default=1.0,
    type=float,
    help="The magnitude of random jitter to add to the key light position.",
)
parser.add_argument(
    "--fill_light_jitter",
    default=1.0,
    type=float,
    help="The magnitude of random jitter to add to the fill light position.",
)
parser.add_argument(
    "--back_light_jitter",
    default=1.0,
    type=float,
    help="The magnitude of random jitter to add to the back light position.",
)
parser.add_argument(
    "--camera_jitter",
    default=0.5,
    type=float,
    help="The magnitude of random jitter to add to the camera position",
)
parser.add_argument(
    "--render_num_samples",
    default=512,
    type=int,
    help="The number of samples to use when rendering. Larger values will "
    + "result in nicer images but will cause rendering to take longer.",
)
parser.add_argument(
    "--render_min_bounces",
    default=8,
    type=int,
    help="The minimum number of bounces to use for rendering.",
)
parser.add_argument(
    "--render_max_bounces",
    default=8,
    type=int,
    help="The maximum number of bounces to use for rendering.",
)
parser.add_argument(
    "--render_tile_size",
    default=256,
    type=int,
    help="The tile size to use for rendering. This should not affect the "
    + "quality of the rendered image but may affect the speed; CPU-based "
    + "rendering may achieve better performance using smaller tile sizes "
    + "while larger tile sizes may be optimal for GPU-based rendering.",
)


def main(args):
    all_scenes = []
    num_digits = 6
    for i in range(args.num_images):
        num_objects = random.randint(args.min_objects, args.max_objects)
        scene_struct = create_scenegraph(
            output_image_path=(
                "%s_%s_%%0%dd.png" % (args.filename_prefix, args.split, num_digits)
            )
            % (i + args.start_idx),
            num_objects=num_objects,
            output_index=(i + args.start_idx),
            output_split=args.split,
            base_scene_blendfile=args.base_scene_blendfile,
            properties_json=args.properties_json,
            material_dir=args.material_dir,
            width=args.width,
            height=args.height,
            render_tile_size=args.render_tile_size,
            use_gpu=args.use_gpu,
            render_num_samples=args.render_num_samples,
            render_min_bounces=args.render_min_bounces,
            render_max_bounces=args.render_max_bounces,
            camera_jitter_param=args.camera_jitter,
            key_light_jitter_param=args.key_light_jitter,
            back_light_jitter_param=args.back_light_jitter,
            fill_light_jitter_param=args.fill_light_jitter,
            max_retries=args.max_retries,
            min_dist=args.min_dist,
            margin_param=args.margin,
            shape_dir=args.shape_dir,
            min_pixels_per_object=args.min_pixels_per_object,
        )

        # Quick check (mostly for testing purposes); this should do nothing
        # Just double checks that our scene struct meets all of the CLEVR
        # requirements (this should have already been ensured at generation time)
        assert validate_scenegraph(
            scene_struct,
            base_scene_blendfile=args.base_scene_blendfile,
            properties_json=args.properties_json,
            material_dir=args.material_dir,
            width=args.width,
            height=args.height,
            render_tile_size=args.render_tile_size,
            use_gpu=args.use_gpu,
            render_num_samples=args.render_num_samples,
            render_min_bounces=args.render_min_bounces,
            render_max_bounces=args.render_max_bounces,
            min_dist=args.min_dist,
            margin_param=args.margin,
            shape_dir=args.shape_dir,
            min_pixels_per_object=args.min_pixels_per_object,
            verbose=True,
        )

        all_scenes.append(scene_struct)

    # After creating all scenegraphs, combine the JSON files for each scene into a
    # single JSON file.
    output = {
        "info": {
            "date": args.date,
            "version": args.version,
            "split": args.split,
            "license": args.license,
        },
        "scenes": all_scenes,
    }
    with open(args.output_scene_file, "w") as f:
        json.dump(output, f, sort_keys=True, indent=2)


def select_attrs_unif_rand(
    obj_id: int,
    size_mapping: List[Tuple[str, float]],
    material_mapping: List[Tuple[str, str]],
    object_mapping: List[Tuple[str, str]],
    color_name_to_rgba: List[Tuple[str, List[float]]],
) -> Tuple[
    Tuple[str, float], Tuple[str, str], Tuple[str, str], Tuple[str, List[float]]
]:
    """
    Given the size, material, shape & colour mappings, uniformly randomly
    chose one each, and return them (in the same order)

    :param size_mapping:
    :param material_mapping:
    :param object_mapping:
    :param color_name_to_rgba:
    :return:
    """
    sz = random.choice(size_mapping)
    mt = random.choice(material_mapping)
    sh = random.choice(object_mapping)
    co = random.choice(color_name_to_rgba)
    return sz, mt, sh, co


def create_scenegraph(
    output_image_path: str,
    num_objects=5,
    output_index=0,
    output_split="none",
    base_scene_blendfile="data/base_scene.blend",
    properties_json="data/properties.json",
    material_dir="data/materials",
    width=480,
    height=320,
    render_tile_size=256,
    use_gpu=0,
    render_num_samples=512,
    render_min_bounces=8,
    render_max_bounces=8,
    camera_jitter_param=0.5,
    key_light_jitter_param=1.0,
    back_light_jitter_param=1.0,
    fill_light_jitter_param=1.0,
    max_retries=50,
    min_dist=0.25,
    margin_param=0.4,
    shape_dir="data/shapes",
    min_pixels_per_object=200,
    attr_selection_fun: Callable[
        [
            int,
            List[Tuple[str, float]],
            List[Tuple[str, str]],
            List[Tuple[str, str]],
            List[Tuple[str, List[float]]],
        ],
        Tuple[
            Tuple[str, float], Tuple[str, str], Tuple[str, str], Tuple[str, List[float]]
        ],
    ] = select_attrs_unif_rand,
):
    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=base_scene_blendfile)

    # Load materials
    utils.load_materials(material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = "dummy_string.png"
    render_args.resolution_x = width
    render_args.resolution_y = height
    render_args.resolution_percentage = 100
    render_args.tile_x = render_tile_size
    render_args.tile_y = render_tile_size
    if use_gpu == 1:
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
    bpy.context.scene.cycles.samples = render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = render_max_bounces
    if use_gpu == 1:
        bpy.context.scene.cycles.device = "GPU"

    # This will give ground-truth information about the scene and its objects
    scene_struct = {
        "split": output_split,
        "image_index": output_index,
        "image_filename": os.path.basename(output_image_path),
        "objects": [],
        "directions": {},
    }

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    def rand(L):
        if L > 0:
            return 2.0 * L * (random.random() - 0.5)
        else:
            return 0

    # Add random jitter to camera position
    camera_jitter = tuple(rand(camera_jitter_param) for i in range(3))
    if camera_jitter_param > 0:
        for i in range(3):
            bpy.data.objects["Camera"].location[i] += camera_jitter[i]

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects["Camera"]
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Save all six axis-aligned directions in the scene struct
    scene_struct["directions"]["behind"] = tuple(plane_behind)
    scene_struct["directions"]["front"] = tuple(-plane_behind)
    scene_struct["directions"]["left"] = tuple(plane_left)
    scene_struct["directions"]["right"] = tuple(-plane_left)
    scene_struct["directions"]["above"] = tuple(plane_up)
    scene_struct["directions"]["below"] = tuple(-plane_up)

    # Add random jitter to lamp positions
    key_light_jitter = tuple(rand(key_light_jitter_param) for i in range(3))
    back_light_jitter = tuple(rand(back_light_jitter_param) for i in range(3))
    fill_light_jitter = tuple(rand(fill_light_jitter_param) for i in range(3))

    if key_light_jitter_param > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Key"].location[i] += key_light_jitter[i]
    if back_light_jitter_param > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Back"].location[i] += back_light_jitter[i]
    if fill_light_jitter_param > 0:
        for i in range(3):
            bpy.data.objects["Lamp_Fill"].location[i] += fill_light_jitter[i]

    # record all random jitters in a variant object
    scene_struct["camera_jitter"] = camera_jitter
    scene_struct["Lamp_Key"] = key_light_jitter
    scene_struct["Lamp_Back"] = back_light_jitter
    scene_struct["Lamp_Fill"] = fill_light_jitter

    # Now make some random objects
    objects, blender_objects = add_random_objects(
        scene_struct,
        num_objects,
        camera,
        properties_json=properties_json,
        max_retries=max_retries,
        min_dist=min_dist,
        margin_param=margin_param,
        shape_dir=shape_dir,
        min_pixels_per_object=min_pixels_per_object,
        attr_selection_fun=attr_selection_fun,
    )

    # Complete & return the scene data structure
    scene_struct["objects"] = objects
    scene_struct["relationships"] = compute_all_relationships(scene_struct)

    return scene_struct


def add_random_objects(
    scene_struct,
    num_objects,
    camera,
    properties_json="data/properties.json",
    max_retries=50,
    min_dist=0.25,
    margin_param=0.4,
    shape_dir="data/shapes",
    min_pixels_per_object=200,
    attr_selection_fun: Callable[
        [
            int,
            List[Tuple[str, float]],
            List[Tuple[str, str]],
            List[Tuple[str, str]],
            List[Tuple[str, List[float]]],
        ],
        Tuple[
            Tuple[str, float], Tuple[str, str], Tuple[str, str], Tuple[str, List[float]]
        ],
    ] = select_attrs_unif_rand,
):
    """
    Add random objects to the current blender scene
    """

    # Load the property file
    with open(properties_json, "r") as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties["colors"].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        material_mapping = [(v, k) for k, v in properties["materials"].items()]
        object_mapping = [(v, k) for k, v in properties["shapes"].items()]
        size_mapping = list(properties["sizes"].items())

    color_name_to_rgba = list(color_name_to_rgba.items())

    positions = []
    objects = []
    blender_objects = []
    for i in range(num_objects):
        # Choose object attributes based on the provided method
        # size_name, r = random.choice(size_mapping)
        # mat_name, mat_name_out = random.choice(material_mapping)
        # obj_name, obj_name_out = random.choice(object_mapping)
        # color_name, rgba = random.choice(color_name_to_rgba)
        (
            (size_name, r),
            (mat_name, mat_name_out),
            (obj_name, obj_name_out),
            (color_name, rgba),
        ) = attr_selection_fun(
            i, size_mapping, material_mapping, object_mapping, color_name_to_rgba
        )

        # Try to place the object, ensuring that we don't intersect any existing
        # objects and that we are more than the desired margin away from all existing
        # objects along all cardinal directions.
        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            if num_tries > max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                return add_random_objects(
                    scene_struct,
                    num_objects,
                    camera,
                    properties_json=properties_json,
                    max_retries=max_retries,
                    min_dist=min_dist,
                    margin_param=margin_param,
                    shape_dir=shape_dir,
                    min_pixels_per_object=min_pixels_per_object,
                    attr_selection_fun=attr_selection_fun,
                )
            x = random.uniform(-3, 3)
            y = random.uniform(-3, 3)
            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            dists_good = True
            margins_good = True
            for xx, yy, rr in positions:
                dx, dy = x - xx, y - yy
                dist = math.sqrt(dx * dx + dy * dy)
                if dist - r - rr < min_dist:
                    dists_good = False
                    break
                for direction_name in ["left", "right", "front", "behind"]:
                    direction_vec = scene_struct["directions"][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < margin_param:
                        # print(margin, args.margin, direction_name)
                        # print('BROKEN MARGIN!')
                        margins_good = False
                        break
                if not margins_good:
                    break

            if dists_good and margins_good:
                break

        # NOTE: Clevr uses 'SmoothCube_v2', therefore this adjustment should never
        # happen.
        # if obj_name == 'Cube':
        #  r /= math.sqrt(2)

        # Choose random orientation for the object.
        theta = 360.0 * random.random()

        # Actually add the object to the scene
        utils.add_object(
            shape_dir, obj_name, r, (x, y), theta=theta, original_translate=False
        )
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Attach the randomly chosen material
        utils.add_material(mat_name, Color=rgba)

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)
        objects.append(
            {
                "shape": obj_name_out,
                "size": size_name,
                "material": mat_name_out,
                "3d_coords": tuple(obj.location),
                "rotation": theta,
                "pixel_coords": pixel_coords,
                "color": color_name,
            }
        )

    # Check that all objects are at least partially visible in the rendered image
    all_visible = check_visibility(blender_objects, min_pixels_per_object)
    if not all_visible:
        # If any of the objects are fully occluded then start over; delete all
        # objects from the scene and place them all again.
        print("Some objects are occluded; replacing objects")
        for obj in blender_objects:
            utils.delete_object(obj)
        return add_random_objects(
            scene_struct,
            num_objects,
            camera,
            properties_json=properties_json,
            max_retries=max_retries,
            min_dist=min_dist,
            margin_param=margin_param,
            shape_dir=shape_dir,
            min_pixels_per_object=min_pixels_per_object,
            attr_selection_fun=attr_selection_fun,
        )

    return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.2):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct["directions"].items():
        if name == "above" or name == "below":
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct["objects"]):
            coords1 = obj1["3d_coords"]
            related = set()
            for j, obj2 in enumerate(scene_struct["objects"]):
                if obj1 == obj2:
                    continue
                coords2 = obj2["3d_coords"]
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object, verbose=False):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    """
    f, path = tempfile.mkstemp(suffix=".png")
    object_colors = render_shadeless(blender_objects, path=path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    color_count = Counter(
        (p[i], p[i + 1], p[i + 2], p[i + 3]) for i in range(0, len(p), 4)
    )
    os.close(f)
    os.remove(path)
    if len(color_count) != len(blender_objects) + 1:
        if verbose:
            print("check_visibility failed: Mismatch between # of colours & objects")
            print(len(color_count))
            print(color_count)
            print(len(blender_objects))
            print(blender_objects)
            print(color_count.most_common())
        return False
    for _, count in color_count.most_common():
        if count < min_pixels_per_object:
            if verbose:
                print("check_visibility failed: Pixel count is not met")
                print(count)
                print(min_pixels_per_object)
                print(color_count.most_common())
            return False
    return True


def render_shadeless(blender_objects, path="flat.png"):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_engine = render_args.engine
    old_use_antialiasing = render_args.use_antialiasing

    # Override some render settings to have flat shading
    render_args.filepath = path
    render_args.engine = "BLENDER_RENDER"
    render_args.use_antialiasing = False

    # Move the lights and ground to layer 2 so they don't render
    utils.set_layer(bpy.data.objects["Lamp_Key"], 2)
    utils.set_layer(bpy.data.objects["Lamp_Fill"], 2)
    utils.set_layer(bpy.data.objects["Lamp_Back"], 2)
    utils.set_layer(bpy.data.objects["Ground"], 2)

    # Add random shadeless materials to all objects
    object_colors = set()
    old_materials = []
    for i, obj in enumerate(blender_objects):
        old_materials.append(obj.data.materials[0])
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

    # Undo the above; first restore the materials to objects
    for mat, obj in zip(old_materials, blender_objects):
        obj.data.materials[0] = mat

    # Move the lights and ground back to layer 0
    utils.set_layer(bpy.data.objects["Lamp_Key"], 0)
    utils.set_layer(bpy.data.objects["Lamp_Fill"], 0)
    utils.set_layer(bpy.data.objects["Lamp_Back"], 0)
    utils.set_layer(bpy.data.objects["Ground"], 0)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.engine = old_engine
    render_args.use_antialiasing = old_use_antialiasing

    return object_colors


def validate_scenegraph(
    scene_struct: Dict,
    base_scene_blendfile="data/base_scene.blend",
    properties_json="data/properties.json",
    material_dir="data/materials",
    width=480,
    height=320,
    render_tile_size=256,
    use_gpu=0,
    render_num_samples=512,
    render_min_bounces=8,
    render_max_bounces=8,
    min_dist=0.25,
    margin_param=0.4,
    shape_dir="data/shapes",
    min_pixels_per_object=200,
    verbose=False,
    check_pixel_coordinates: bool = True,
    overwrite_pixel_coordinates: bool = False,
) -> bool:
    """
    Given a scene struct for a CLEVR image, return whether the scene struct
    meets all CLEVR requirements (i.e., min distance between objects, occlusion,
    etc...), and whether the relationships in the scene_struct match the actual
    object relationships.

    This function will check if the pixel coordinates are correct iff check_pixel_coordinates=True

    If overwrite_pixel_coordinates=True, then the pixel scene_struct will be
    mutated so that the pixel coordinates match whatever blender says they are.

    NOTE: oddly, directions don't seem to change with camera position. Go figure.
    """

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath=base_scene_blendfile)

    # Load materials
    utils.load_materials(material_dir)

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    render_args.filepath = "dummy_string.png"
    render_args.resolution_x = width
    render_args.resolution_y = height
    render_args.resolution_percentage = 100
    render_args.tile_x = render_tile_size
    render_args.tile_y = render_tile_size
    if use_gpu == 1:
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
    bpy.context.scene.cycles.samples = render_num_samples
    bpy.context.scene.cycles.transparent_min_bounces = render_min_bounces
    bpy.context.scene.cycles.transparent_max_bounces = render_max_bounces
    if use_gpu == 1:
        bpy.context.scene.cycles.device = "GPU"

    # Put a plane on the ground so we can compute cardinal directions
    bpy.ops.mesh.primitive_plane_add(radius=5)
    plane = bpy.context.object

    # Add jitter to camera position
    for i in range(3):
        bpy.data.objects["Camera"].location[i] += scene_struct["camera_jitter"][i]

    # Figure out the left, up, and behind directions along the plane and record
    # them in the scene structure
    camera = bpy.data.objects["Camera"]
    plane_normal = plane.data.vertices[0].normal
    cam_behind = camera.matrix_world.to_quaternion() * Vector((0, 0, -1))
    cam_left = camera.matrix_world.to_quaternion() * Vector((-1, 0, 0))
    cam_up = camera.matrix_world.to_quaternion() * Vector((0, 1, 0))
    plane_behind = (cam_behind - cam_behind.project(plane_normal)).normalized()
    plane_left = (cam_left - cam_left.project(plane_normal)).normalized()
    plane_up = cam_up.project(plane_normal).normalized()

    # Delete the plane; we only used it for normals anyway. The base scene file
    # contains the actual ground plane.
    utils.delete_object(plane)

    # Check all six axis-aligned directions in the scene struct match blender
    for dir_name, dir_val in [
        ("behind", tuple(plane_behind)),
        ("front", tuple(-plane_behind)),
        ("left", tuple(plane_left)),
        ("right", tuple(-plane_left)),
        ("above", tuple(plane_up)),
        ("below", tuple(-plane_up)),
    ]:
        assert tuple(scene_struct["directions"][dir_name]) == dir_val

    for i in range(3):
        bpy.data.objects["Lamp_Key"].location[i] += scene_struct["Lamp_Key"][i]
    for i in range(3):
        bpy.data.objects["Lamp_Back"].location[i] += scene_struct["Lamp_Back"][i]
    for i in range(3):
        bpy.data.objects["Lamp_Fill"].location[i] += scene_struct["Lamp_Fill"][i]

    # Now introduce the objects:
    # Load the property file
    with open(properties_json, "r") as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties["colors"].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        # material_mapping = list(properties['materials'].items())
        # object_mapping = list(properties['shapes'].items())
        # size_mapping = list(properties['sizes'].items())
        # color_name_to_rgba = list(color_name_to_rgba.items())

    positions = []
    blender_objects = []
    for obj_struct in scene_struct["objects"]:
        # Get human-readable names of all this object's attributes from the struct
        size_name = obj_struct["size"]
        color_name = obj_struct["color"]
        obj_name_out = obj_struct["shape"]
        mat_name_out = obj_struct["material"]

        # Get object radius
        r = properties["sizes"][size_name]

        # Get object position
        x = obj_struct["3d_coords"][0]
        y = obj_struct["3d_coords"][1]

        # Check to make sure the new object is further than min_dist from all
        # other objects, and further than margin along the four cardinal directions
        for xx, yy, rr in positions:
            dx, dy = x - xx, y - yy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist - r - rr < min_dist:
                # Distances aren't meeting the requirement
                if verbose:
                    print("Failed distance check")
                return False
            for direction_name in ["left", "right", "front", "behind"]:
                direction_vec = scene_struct["directions"][direction_name]
                assert direction_vec[2] == 0
                margin = dx * direction_vec[0] + dy * direction_vec[1]
                if 0 < margin < margin_param:
                    # margins don't meet the requirements
                    if verbose:
                        print("Failed margin check")
                    return False

        # Note: original code adjusts size for 'Cube' blender object, but that no
        # longer exists (we now have the 'SmoothCube_v2' blender object which does
        # *not* have it's radius adjusted).

        theta = obj_struct["rotation"]

        # Actually add the object to the scene
        utils.add_object(
            shape_dir,
            properties["shapes"][obj_name_out],
            r,
            (x, y),
            theta=theta,
            original_translate=False,
        )
        obj = bpy.context.object
        blender_objects.append(obj)
        positions.append((x, y, r))

        # Attach the material
        utils.add_material(
            properties["materials"][mat_name_out], Color=color_name_to_rgba[color_name]
        )

        if overwrite_pixel_coordinates:
            obj_struct["pixel_coords"] = utils.get_camera_coords(camera, obj.location)
        elif (
            check_pixel_coordinates
            and utils.get_camera_coords(camera, obj.location)
            != obj_struct["pixel_coords"]
        ):
            if verbose:
                print("Pixel coordinates do not match")
            return False

        # Check z-position matches struct
        assert tuple(obj.location)[2] == obj_struct["3d_coords"][2]

    if not check_visibility(blender_objects, min_pixels_per_object, verbose=verbose):
        # Occluded objects, this scene graph is no good.
        for obj in blender_objects:
            utils.delete_object(obj)
        if verbose:
            print("Failed occlusion check")
        return False

    # Check the relationships between objects match
    matching_relationships = scene_struct["relationships"] == compute_all_relationships(
        scene_struct
    )

    if verbose and not matching_relationships:
        print("Failed to preserve relationships")
        print(scene_struct["relationships"])
        print("*" * 80)
        print(compute_all_relationships(scene_struct))
        exit(-1)

    return matching_relationships


if __name__ == "__main__":
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args1 = parser.parse_args(argv)
        main(args1)
    elif "--help" in sys.argv or "-h" in sys.argv:
        parser.print_help()
    else:
        print("This script is intended to be called from blender like this:")
        print()
        print("blender --background --python held_out_utils_generate_sg.py -- [args]")
        print()
        print("You can also run as a standalone python script to view all")
        print("arguments like this:")
        print()
        print("python held_out_utils_generate_sg.py --help")
