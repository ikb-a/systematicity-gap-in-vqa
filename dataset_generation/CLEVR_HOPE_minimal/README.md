# CLEVR-HOPE Minimal Test Splits Generation

[![Python Version](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This directory contains the code for creating all the minimal-IID and minimal-OOD splits of CLEVR-HOPE (i.e., the splits for all of HOP0-28).

This code has only been tested with Python 3.7 and [Blender 2.78c](https://www.blender.org/download/releases/2-78/) is required to generate images. 

Minimal requirements are found in [min_requirements.txt](../../min_requirements.txt)

If you also want to generate matplotlib visualizations, generate hdf5 files, and extract faster RCNN features then please install the packages in [full_requirements.txt](../../full_requirements.txt) (note that you may need to adjust torch CUDA requirements)and then install the vqa-framework package (in the root directory) to proceed:

```
pip install -e ./vqa-framework
```

NOTE: If blender is having difficulties importing files, then it may be necesary to update it's PATH variable via
```
echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr_hope.pth
```

## Steps

1. Run generate_atom_ho_exists_v1_0.py to generate questions and scenegraphs.

```
# from this directory
export PYTHONPATH="."
blender --background --python generate_atom_ho_exists_add_v1_0.py
```

This will create the subfolder `output_atom_ho_exists` containing:
- raw minimal-OOD question files (e.g., `atom_ho_exist_match_ho0.json`) 
- minimal-IID question files (e.g., `atom_ho_exist_mismatch_ho0.json`)
- intermediate scenegraphs for use in image generation (`tmp_sg_for_generation.json`)
- A recording of the shared lighting & camera positions used by each variant (`variants.json`). This file  shouldn't be needed.

2. Generate the final images & final scene graph .json files using `render_images_from_sg.py`:

You can generate the first 500 images by running:

```commandline
mkdir output_atom_ho_exists/tmp
blender --background --python render_images_from_sg.py  -- --start_idx 0 --num_images 500 --use_gpu 1 --output_scene_file ./output_atom_ho_exists/tmp/rendered_scenes.json --output_scene_dir ./output_atom_ho_exists/raw_scenes --output_image_dir ./output_atom_ho_exists/raw_images --full-spec --use_gpu 1 --scene_graphs_path output_atom_ho_exists/tmp_sg_for_generation.json
```

Note that if you are generating more than 600 or so images, then the rendering can potentially get 
stuck in an infinite loop. This, problem dates to the original CLEVR script https://github.com/facebookresearch/clevr-dataset-gen/issues/18.
If you need to generate more images, use the `--start_idx` and `--num_images` flags. For example, `--start_idx 0 --num_images 500`, then `--start_idx 500 --num_images 500`, then `--start_idx 1000 --num_images 500`, etc... 

Once generation is done, then under the subfolder `output_atom_ho_exists`, you will find:
- The `images` subdirectory containing the generated `.png` images
- The `scenes` subdirectory containing the final scenegraph for each images, stored as seperate `.json` files
- The `tmp` subdirectory containing a single .json that has merged all of the generated scenegraphs into a single file. Note that this file is typically useless, because most of the time you will need to use the `--start_idx` and `--num_images` flags. Therefore this file will only contain the scenegraphs for the 500 or so generated images, rather than all the scenegraphs in the dataset. 

For a quick, low-quality render, run:
```commandline
mkdir output_atom_ho_exists/tmp
blender --background --python render_images_from_sg.py  -- --output_scene_file ./output_atom_ho_exists/tmp/rendered_scenes.json --output_scene_dir ./output_atom_ho_exists/raw_scenes --output_image_dir ./output_atom_ho_exists/raw_images --full-spec --use_gpu 1 --render_num_samples 2 --scene_graphs_path output_atom_ho_exists/tmp_sg_for_generation.json --start_idx 0 --num_images 500
```

If the above is still too slow for testing, you can reduce the image resolution further (note that you must 
adjust `--min_pixels_per_object`, as at a lower resolution, as there will be less pixels per object -- if it's not adjusted, the script may believe that one of the images is invalid due to occlusion):
```commandline
mkdir output_atom_ho_exists/tmp
blender --background --python render_images_from_sg.py  -- --output_scene_file ./output_atom_ho_exists/tmp/rendered_scenes.json --output_scene_dir ./output_atom_ho_exists/raw_scenes --output_image_dir ./output_atom_ho_exists/raw_images --full-spec --use_gpu 1 --width 160 --height 120 --min_pixels_per_object 25 --render_num_samples 2 --scene_graphs_path output_atom_ho_exists/tmp_sg_for_generation.json --start_idx 0 --num_images 500
```

3. Merge the folder of generated individual scene graphs into a single .json file using `collect_scenes.py` 

```commandline
mkdir -p ./output_atom_ho_exists/scenes/minimal/
python3 collect_scenes.py --output_file ./output_atom_ho_exists/scenes/minimal/atom_CLEVR_val_v1.0_scenes.json --input_dir ./output_atom_ho_exists/raw_scenes 
```

4. Create the image hdf5 files, by merging all the individual .png files.

```commandline
mkdir -p ./output_atom_ho_exists/images/minimal/
python3 CLEVR_im_to_hdf5.py --input_image_dir ./output_atom_ho_exists/raw_images --output_h5_file ./output_atom_ho_exists/images/minimal/val_ims.h5
```

5. Optional: Check things have worked properly by running visualizer.py to produce a visualization of the questions, using the question .json files and the image hdf5 file.

```commandline
python visualizer.py
```

The visualizer creates composite images of the test case, for every HOP.
Each row corresponds to matched image/questions of the form
X&Y, X&not(Y), not(X)&Y, not(X) & not(Y), where X and Y are the attributes 
being asked about (i.e., "Is there an XY object?"). 

For example, for X=rubber and Y=cylinder; not(X) is any material other than 
rubber (in this case, it must be metal), and not(Y) is any shape other than 
cylinder (sphere or cube). Note that for each set of images, we fix a choice
of not(X) & not(Y). Specifically, different 'variant' images (i.e, fixed choices of
lighting & camera position) use different choices of not(X) and not(Y).

Each row in the visualization is a different choice of X & Y. Note that for 
any `atom_ho_exist_match_ho` there
can only be a single row (as there is only a single pair we are interested in).

On the other hand, the `atom_ho_exist_mismatch` will contain several
rows as there are several pairs that are NOT our ho combination.

Moving along rows, different choices of Z&W are used, where Z & W are the 
remaining 2 attributes that are not constrained. 
(e.g., for HOP="rubber cylinder", then Z & W would be size & colour).

6. Create the question hdf5 files.

```
python atom_ho_exist_template/convert_to_hdf5.py
```

8. Update `DATASET_PATH` and `SCENES_PATH` in [filter.py](filter.py), then run the script.

```
python filter.py
```

At this point you should keep only the created `val_atom_ho_exist_mismatch_ho{ho_idx}_filtered.h5` files, and delete the original `val_atom_ho_exist_mismatch_ho{ho_idx}.h5` files.

