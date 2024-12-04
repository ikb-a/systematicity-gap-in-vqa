# CLEVR-HOPE Complex HOP 6-28


[![Python Version](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This directory contains the code for creating a subset of the complex-IID and complex-OOD splits of CLEVR-HOPE (specifically, the splits for all of HOP6-28).

This code has only been tested with Python 3.7 and [Blender 2.78c](https://www.blender.org/download/releases/2-78/) is required to generate images. 

Please install the packages in [full_requirements.txt](../../full_requirements.txt) (note that you may need to adjust torch CUDA requirements)and then install the vqa-framework package (in the root directory):

```
pip install -e ./vqa-framework
```

NOTE: If blender is having difficulties importing files, then it may be necesary to update it's PATH variable via
```
echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr_hope.pth
```


## Instructions

Specify the HOP's you want in the file `ho_tuples_config.json`

Specify where the CLEVR training set data is in the file `reuse_config.json`

For HOP6-28 we used 4 randomly selected tuples: 
```
"('small', 'brown', 'sphere', 'rubber')",
"('small', 'red', 'cylinder', 'metal')",
"('large', 'gray', 'cube', 'metal')",
"('small', 'purple', 'sphere', 'rubber')"
```

This has 3 duplicated pairs:
1) (small rubber) (we've exhausted all size/material combos so we deleted this redundant HOP)
2) (small, sphere) (randomly replaced that with small cube]
3) (rubber, sphere) (randomly replaced that with rubber cube]

## Setup instructions

First, install Blender 2.78c  (NOTE: important to use this version as there are issues with later versions)
Ensure it is accessible via command line (e.g., add path to installation folder to `$PATH` via `.bashrc`)

Second, you may want to check if Blender can see any GPU(s) that may be present.
Note that if it can't then it will cause an error message (something like the below) when running some of the commands later on.

```
cycles_prefs.compute_device_type = 'CUDA'
TypeError: bpy_struct: item.attr = val: enum "CUDA" not found in ('NONE')
```

Third, set up Blender to be able to see the python files in this folder by running the following from _this_ 
folder (i.e., from the location with the python files that are executed in the instructions below):

```
BLENDER="directory where Blender is installed"
#BLENDER=/scratch/gobi2/ianberlot/blender-2.78c-linux-glibc219-x86_64/
VERSION="2.78"
echo `pwd` >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/held_out_clevr_add.pth
echo {full path to dataset_generation/atom_clevr/} >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/atom_clevr.pth
```

Fourth, edit `reuse_config.json` to contain the correct paths to the CLEVR dataset. Please use our processed CLEVR dataset
(note that you only need the subpaths in the .json -- i.e., the scenegraphs and the train images).
You should be able to automatically generate the required files by running the vqa-framework's CLEVRDataModule.
This may take some time as it will automatically download all of CLEVR.

## Generation instructions:

1. Run `generate_held_out_objects.py` to determine the number of objects & their attributes.
By default this creates the files `held_out_objects_test.json`, `held_out_objects_test.json`,
and `held_out_objects_val_iid.json`, under the folder `outputs`.

NOTE: the counts for the number of scenes to create (i.e., `train_ims_without_HO`, `val_iid_ims_without_HO`, 
`test_ims`) are all per held-out combination. Therefore, if
you have N held-out combinations, you'll have at most N*`test_ims` unique test images (note that image reuse is possible, hence
this is just the upper bound)

```
# TEST
# mkdir out_debug
# python3 generate_held_out_objects.py --ho_tuples_config ho_tuples_config_test.json --reusable_ims_config reuse_config.json --train_ims_without_HO 10 --val_iid_ims_without_HO 10 --test_ims 10 --train_output_path out_debug/held_out_objects_train.json --val_iid_output_path out_debug/held_out_objects_val_iid.json --test_output_path out_debug/held_out_objects_test.json 

mkdir output
python3 generate_held_out_objects.py --reusable_ims_config reuse_config.json
```

2. For each produced objects file, create the indices. i.e., run:
```
# TEST
# blender --background --python generate_held_out_sg_from_objects.py -- --create_index --split train --objects_path out_debug/held_out_objects_train.json --output_path out_debug/scenes_to_render/ --index_map_output_path out_debug/scenes_to_render/ --reusable_ims_config reuse_config.json 
# blender --background --python generate_held_out_sg_from_objects.py -- --create_index --split test --objects_path out_debug/held_out_objects_test.json --output_path out_debug/scenes_to_render/ --index_map_output_path out_debug/scenes_to_render/ --reusable_ims_config reuse_config.json 
# blender --background --python generate_held_out_sg_from_objects.py -- --create_index --split val_iid --objects_path out_debug/held_out_objects_val_iid.json --output_path out_debug/scenes_to_render/ --index_map_output_path out_debug/scenes_to_render/ --reusable_ims_config reuse_config.json 

blender --background --python generate_held_out_sg_from_objects.py -- --create_index --split train --objects_path output/held_out_objects_train.json --reusable_ims_config reuse_config.json 
blender --background --python generate_held_out_sg_from_objects.py -- --create_index --split test --objects_path output/held_out_objects_test.json --reusable_ims_config reuse_config.json 
blender --background --python generate_held_out_sg_from_objects.py -- --create_index --split val_iid --objects_path output/held_out_objects_val_iid.json --reusable_ims_config reuse_config.json 
```
3. For each produced objects file, create the render scenegraphs. i.e., run:

NOTE: `--num_glob_indices_to_process` is the number of scenegraphs to generate, starting at the index `--start_idx`, inclusive. Set to a very large number to do all of them.

NOTE: About 1 second per scene graph; you can parallelize need be by having different calls working on non-overlapping indices at once.

Note: you can either use a sufficiently large choice of `--num_glob_indices_to_process`, or you can
alternatively divide the indices across several GPUs.


```
# TEST
#blender --background --python generate_held_out_sg_from_objects.py -- --split train --objects_path out_debug/held_out_objects_train.json --start_idx 0 --num_glob_indices_to_process 100 --output_path out_debug/scenes_to_render/ --index_map_output_path out_debug/scenes_to_render/ --reusable_ims_config reuse_config.json 
#blender --background --python generate_held_out_sg_from_objects.py -- --split val_iid --objects_path out_debug/held_out_objects_val_iid.json --start_idx 0 --num_glob_indices_to_process 100 --output_path out_debug/scenes_to_render/ --index_map_output_path out_debug/scenes_to_render/ --reusable_ims_config reuse_config.json 
#blender --background --python generate_held_out_sg_from_objects.py -- --split test --objects_path out_debug/held_out_objects_test.json --start_idx 0 --num_glob_indices_to_process 100 --output_path out_debug/scenes_to_render/ --index_map_output_path out_debug/scenes_to_render/ --reusable_ims_config reuse_config.json 


blender --background --python generate_held_out_sg_from_objects.py -- --split train --objects_path output/held_out_objects_train.json --start_idx 0 --num_glob_indices_to_process 9999999999 --reusable_ims_config reuse_config.json 
blender --background --python generate_held_out_sg_from_objects.py -- --split val_iid --objects_path output/held_out_objects_val_iid.json --start_idx 0 --num_glob_indices_to_process 9999999999 --reusable_ims_config reuse_config.json 
blender --background --python generate_held_out_sg_from_objects.py -- --split test --objects_path output/held_out_objects_test.json --start_idx 0 --num_glob_indices_to_process 9999999999 --reusable_ims_config reuse_config.json 
```

NOTE: If blender is not properly seeing your GPU, then this step will fail. You'll need to fix blender before proceeding.

4. Now merge the render scenegraphs into a single file:
```
# TEST
#blender --background --python generate_held_out_sg_from_objects.py -- --merge_scenegraphs --split train --objects_path out_debug/held_out_objects_train.json --output_path out_debug/scenes_to_render/ --index_map_output_path out_debug/scenes_to_render/ --reusable_ims_config reuse_config.json 
#blender --background --python generate_held_out_sg_from_objects.py -- --merge_scenegraphs --split val_iid --objects_path out_debug/held_out_objects_val_iid.json --output_path out_debug/scenes_to_render/ --index_map_output_path out_debug/scenes_to_render/ --reusable_ims_config reuse_config.json 
#blender --background --python generate_held_out_sg_from_objects.py -- --merge_scenegraphs --split test --objects_path out_debug/held_out_objects_test.json --output_path out_debug/scenes_to_render/ --index_map_output_path out_debug/scenes_to_render/ --reusable_ims_config reuse_config.json 

blender --background --python generate_held_out_sg_from_objects.py -- --merge_scenegraphs --split train --objects_path output/held_out_objects_train.json --reusable_ims_config reuse_config.json 
blender --background --python generate_held_out_sg_from_objects.py -- --merge_scenegraphs --split val_iid --objects_path output/held_out_objects_val_iid.json --reusable_ims_config reuse_config.json 
blender --background --python generate_held_out_sg_from_objects.py -- --merge_scenegraphs --split test --objects_path output/held_out_objects_test.json --reusable_ims_config reuse_config.json 
```
5. Run `render_images_from_sg.py` from `dataset_generation/dataset_generation/atom_clevr`; it will skip generating
any images that are marked as being reused. Note that you can only render up to 500
images in a single go, so you will almost certainly have to divide the dataset among several
calls. You can do process a batch of 500 by setting `--start_idx` to a multiple of 500 (where to start), 
and including `--num_images 500` 
```
# TEST
#blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path out_debug/scenes_to_render/tmp_train_sg_for_generation.json --output_image_dir out_debug/images/train/ --output_scene_dir out_debug/scenes/train/ --output_scene_file out_debug/scenes/CLEVR_held_out_train.json --start_idx 0 --num_images 100 --render_num_samples 3
#blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path out_debug/scenes_to_render/tmp_val_iid_sg_for_generation.json --output_image_dir out_debug/images/val_iid/ --output_scene_dir out_debug/scenes/val_iid/ --output_scene_file out_debug/scenes/CLEVR_held_out_val_iid.json --start_idx 0 --num_images 100 --render_num_samples 3
#blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path out_debug/scenes_to_render/tmp_test_sg_for_generation.json --output_image_dir out_debug/images/test/ --output_scene_dir out_debug/scenes/test/ --output_scene_file out_debug/scenes/CLEVR_held_out_test.json --start_idx 0 --num_images 100 --render_num_samples 3

blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path output/scenes_to_render/tmp_train_sg_for_generation.json --output_image_dir output/images/train/ --output_scene_dir output/scenes/train/ --output_scene_file output/scenes/CLEVR_held_out_train.json --start_idx 0 --num_images 500
blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path output/scenes_to_render/tmp_val_iid_sg_for_generation.json --output_image_dir output/images/val_iid/ --output_scene_dir output/scenes/val_iid/ --output_scene_file output/scenes/CLEVR_held_out_val_iid.json --start_idx 0 --num_images 500
blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path output/scenes_to_render/tmp_test_sg_for_generation.json --output_image_dir output/images/test/ --output_scene_dir output/scenes/test/ --output_scene_file output/scenes/CLEVR_held_out_test.json --start_idx 0 --num_images 500
```

NOTE: for a quick & dirty render, you can append the flag & value `--render_num_samples 3`

6. Copy over any re-used training images. Note: you will need to update the paths to the CLEVR dataset inside of the file passed as `--reusable_ims_config`:

Note that this script will not work correctly unless *all* the images have been generated. Also note that
the final image numbers are not generated sequentially.

```
# TEST
# python3 copy_reused_ims.py --output_image_dir out_debug/images/train/ --scene_graphs_path out_debug/scenes_to_render/tmp_train_sg_for_generation.json --reusable_ims_config reuse_config.json 

python3 copy_reused_ims.py --output_image_dir output/images/train/ --scene_graphs_path output/scenes_to_render/tmp_train_sg_for_generation.json --reusable_ims_config reuse_config.json
```

7. At this point, you can iterate through all the images, by category, by running `python3 test_generated_ims.py output`
Note that this is only really practical for small test datasets. (for testing, run `python3 test_generated_ims.py out_debug`)

8. Next, merge all scenegraphs into a single file, rather than a folder of .json files.
Use `collect_scenes.py`, from the original CLEVR codebase (see `clevr-dataset-gen/image_generation/collect_scenes.py`)

```
#TEST
#python3 collect_scenes.py --input_dir out_debug/scenes/train/ --output_file out_debug/scenes/CLEVR_held_out_train.json --date '2023-06-09'
#python3 collect_scenes.py --input_dir out_debug/scenes/test/ --output_file out_debug/scenes/CLEVR_held_out_test.json --date '2022-06-09'
#python3 collect_scenes.py --input_dir out_debug/scenes/val_iid/ --output_file out_debug/scenes/CLEVR_held_out_val_iid.json --date '2022-06-09'


python3 collect_scenes.py --input_dir output/scenes/train/ --output_file output/scenes/CLEVR_held_out_train.json --date '2023-06-09'
python3 collect_scenes.py --input_dir output/scenes/test/ --output_file output/scenes/CLEVR_held_out_test.json --date '2022-06-09'
python3 collect_scenes.py --input_dir output/scenes/val_iid/ --output_file output/scenes/CLEVR_held_out_val_iid.json --date '2022-06-09'
```

9. Generate test split questions:

NOTE: However many held-out combinations are in ho_tuples_config.json; that's the max
number you need to reach. (note: a combination of all 4 attributes, corresponds to (4 choose 2)=6
possible held-out pairs)

```
# TEST
#for INDEX in $(seq 0 11);
#do   
#   python3 generate_held_out_questions.py test --ho_idx $INDEX --verbose --dir out_debug --ho_tuples_config ho_tuples_config_test.json
#done

python3 generate_held_out_questions.py test --ho_idx 0 --verbose
python3 generate_held_out_questions.py test --ho_idx 1 --verbose
python3 generate_held_out_questions.py test --ho_idx 2 --verbose
...
```


10. Generate train questions:

```
# TEST
#for INDEX in $(seq 0 11);
#do   
#   python3 generate_held_out_questions.py train_S1 --ho_idx $INDEX --verbose --dir out_debug --ho_tuples_config ho_tuples_config_test.json
#done

python3 generate_held_out_questions.py train_S1 --ho_idx 0 --verbose
python3 generate_held_out_questions.py train_S1 --ho_idx 1 --verbose
python3 generate_held_out_questions.py train_S1 --ho_idx 2 --verbose
...
```

11. Generate val questions:
```
# TEST
#for INDEX in $(seq 0 11);
#do   
#   python3 generate_held_out_questions.py val_iid_S1 --ho_idx $INDEX --verbose --dir out_debug --ho_tuples_config ho_tuples_config_test.json
#done

python3 generate_held_out_questions.py val_iid_S1 --ho_idx 0 --verbose
python3 generate_held_out_questions.py val_iid_S1 --ho_idx 1 --verbose
python3 generate_held_out_questions.py val_iid_S1 --ho_idx 2 --verbose
...
```

12. Merge questions into final .json files

```
# TEST
#python3 merge_questions.py --dir out_debug --ho_tuples_config ho_tuples_config_test.json

python3 merge_questions.py
```

14. Extract test, val, and test features:
```
# TEST
#export PYTHONPATH=${PYTHONPATH}:../../../vislang/:../../../
#python3 extract_img_feats_held_out.py test --dir out_debug
#python3 extract_img_feats_held_out.py val_iid --dir out_debug
#python3 extract_img_feats_held_out.py train --only_resnet --dir out_debug
#python3 extract_img_feats_held_out.py train --exclude_resnet --dir out_debug

export PYTHONPATH=${PYTHONPATH}:../../../vislang/:../../../
python3 extract_img_feats_held_out.py test
python3 extract_img_feats_held_out.py val_iid
python3 extract_img_feats_held_out.py train --only_resnet
python3 extract_img_feats_held_out.py train --exclude_resnet
```
NOTE: Should be run from this directory. There may be some problems
importing vislang; if so then just add to `PYTHONPATH` environment variable
and try again. i.e., `export PYTHONPATH=${PYTHONPATH}:../../vislang/:../../`

NOTE: you can generate the train features in one go, but only if you have ~350GB
of disk space for every 6 ho's. Otherwise you may want to generate, copy over, then generate the 
rest (hence the two separate calls, ea. requiring about 150GB per every 6 ho's)

15. Rename 'type' key to 'function' due to differences between the released CLEVR dataset and the CLEVR generation code.
```
#TEST
#for INDEX in $(seq 0 11);
#do   
#   python3 rename_ques_key.py $INDEX --dir out_debug --ho_tuples_config ho_tuples_config_test.json
#done

python3 rename_ques_key.py 0
python3 rename_ques_key.py 1
python3 rename_ques_key.py 2
...
```

15. Create test, val, and train question hdf5 files for all tests. Should not require a GPU.
```
#TEST
#python3 extract_ques_feats_held_out.py --dir out_debug --ho_tuples_config ho_tuples_config_test.json

python3 extract_ques_feats_held_out.py
```

16. Next to `hdf5_held_out_CLEVR`, create `hdf5_held_out_CLEVR_ims` and move
the subdirectories `test_v1.0_img`, `train_v1.0_img`, 
and `val_iid_v1.0_img` 

```
#TEST
#mkdir out_debug/hdf5_held_out_CLEVR_ims
#mv out_debug/hdf5_held_out_CLEVR/test_v1.0_img/ -t out_debug/hdf5_held_out_CLEVR_ims/
#mv out_debug/hdf5_held_out_CLEVR/val_iid_v1.0_img/ -t out_debug/hdf5_held_out_CLEVR_ims/
#mv out_debug/hdf5_held_out_CLEVR/train_v1.0_img/ -t out_debug/hdf5_held_out_CLEVR_ims/

mkdir output/hdf5_held_out_CLEVR_ims
mv output/hdf5_held_out_CLEVR/test_v1.0_img/ -t output/hdf5_held_out_CLEVR_ims/
mv output/hdf5_held_out_CLEVR/val_iid_v1.0_img/ -t output/hdf5_held_out_CLEVR_ims/
mv output/hdf5_held_out_CLEVR/train_v1.0_img/ -t output/hdf5_held_out_CLEVR_ims/
```

17. [Optional] To reduce the ammount of RAM memory used by the dataloader, you can trim the scenes
remove any information that is not strictly required for symbolic evaluation (i.e.,
you can remove all info on pixel coordinates, direction vectors, where the image
was reused from, etc...). This script will edit the original file in-place, so you
may want to create a copy of the scene .json files (i.e. the contents 
of `output/scenes`) first. Note that this is less information than the CLEVR
scenes contain (e.g., we also remove the `split` key for each scene)

```
# TEST
mkdir -p out_debug/intermediate_files/scenes/before_trim/
cp -r out_debug/scenes -t  out_debug/intermediate_files/scenes/before_trim/
python3 trim_scenes.py --dir out_debug --ho_tuples_config ho_tuples_config_test.json


mkdir -p output/intermediate_files/scenes/before_trim/
cp -r output/scenes -t  output/intermediate_files
python3 trim_scenes.py
```



# Final testing:

First, download the dataset files somewhere:

In [test_generated_ques_auto.py](test_generated_ques_auto.py) adjust the variable `DATASET_PATH` to point to your dataset path.
This script checks that ho is, or isn't, present as expected. It also checks that we the symbolic executor
when run on the program and scenegraph produces the correct answer.

```
python3 test_generated_ques_auto.py
```


To manually visualize some examples, again change `DATASET_PATH` (this time in [test_generated_ques_manual.py](test_generated_ques_manual.py)) and run

```
python3 test_generated_ques_manual.py
```
