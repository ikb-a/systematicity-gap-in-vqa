# CLEVR-HOPE Complex HOP 0-5


[![Python Version](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/downloads/release/python-370/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This directory contains the code for creating a subset of the complex-IID and complex-OOD splits of CLEVR-HOPE (specifically, the splits for all of HOP0-5).

This code has only been tested with Python 3.7 and [Blender 2.78c](https://www.blender.org/download/releases/2-78/) is required to generate images. 

Please install the packages in [full_requirements.txt](../../full_requirements.txt) (note that you may need to adjust torch CUDA requirements)and then install the vqa-framework package (in the root directory):

```
pip install -e ./vqa-framework
```

NOTE: If blender is having difficulties importing files, then it may be necesary to update it's PATH variable via
```
echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr_hope.pth
```

Note that this code also generates additional data splits that are not required.
These additional data splits are  not genenrated by the code in CLEVR_HOPE_complex_ho6-28; we recommend refactoring that code
to also generate HOPs 0-5 if you want to generate more data.

## Generation instructions

1. Run `generate_held_out_objects.py` to determine the number of objects & their attributes.
By default this creates the files `held_out_objects_test.json`, `held_out_objects_test.json`,
and `held_out_objects_val_iid.json`. Once you've done this, create a folder called 
`outputs` and make a copy of these files under `outputs`.

2. For each produced objects file, create the indices. i.e., run:
Note that generate_held_out_sg_from_objects may not work due to changes to held_out_utils_generate_questions (specifically, importing tqdm and using f-strings which blender does not support)
```
blender --background --python generate_held_out_sg_from_objects.py -- --create_index --split train --objects_path held_out_objects_train.json
blender --background --python generate_held_out_sg_from_objects.py -- --create_index --split test --objects_path held_out_objects_test.json
blender --background --python generate_held_out_sg_from_objects.py -- --create_index --split val_iid --objects_path held_out_objects_val_iid.json
```
3. For each produced objects file, create the render scenegraphs. i.e., run:
```
blender --background --python generate_held_out_sg_from_objects.py -- --split train --objects_path held_out_objects_train.json --start_idx 0 --num_glob_indices_to_process 1000000
blender --background --python generate_held_out_sg_from_objects.py -- --split val_iid --objects_path held_out_objects_val_iid.json --start_idx 0 --num_glob_indices_to_process 100000
blender --background --python generate_held_out_sg_from_objects.py -- --split test --objects_path held_out_objects_test.json --start_idx 0 --num_glob_indices_to_process 100000
```
Note: you can either use a sufficiently large choice of `--num_glob_indices_to_process`, or you can
alternatively divide the indices accross several GPUs.
4. Now merge the render scenegraphs into a single file:
```
blender --background --python generate_held_out_sg_from_objects.py -- --merge_scenegraphs --split train --objects_path held_out_objects_train.json
blender --background --python generate_held_out_sg_from_objects.py -- --merge_scenegraphs --split val_iid --objects_path held_out_objects_val_iid.json
blender --background --python generate_held_out_sg_from_objects.py -- --merge_scenegraphs --split test --objects_path held_out_objects_test.json
```
5. Run `render_images_from_sg.py` from `dataset_generation/dataset_generation/atom_clevr`; it will skip generating
any images that are marked as being reused. Note that you can only render up to 500
images in a single go, so you will almost certainly have to divide the dataset among several
calls.
```
blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path outputs/scenes_to_render/tmp_train_sg_for_generation.json --output_image_dir outputs/images/train/ --output_scene_dir outputs/scenes/train/ --output_scene_file outputs/scenes/CLEVR_held_out_train.json --start_idx 0 --num_images 500
blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path outputs/scenes_to_render/tmp_val_iid_sg_for_generation.json --output_image_dir outputs/images/val_iid/ --output_scene_dir outputs/scenes/val_iid/ --output_scene_file outputs/scenes/CLEVR_held_out_val_iid.json --start_idx 0 --num_images 500
blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path outputs/scenes_to_render/tmp_test_sg_for_generation.json --output_image_dir outputs/images/test/ --output_scene_dir outputs/scenes/test/ --output_scene_file outputs/scenes/CLEVR_held_out_test.json --start_idx 0 --num_images 500
```

NOTE: for a quick & dirty render, you can use the following settings instead:

```
blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path outputs/scenes_to_render/tmp_train_sg_for_generation.json --output_image_dir outputs/images/train/ --output_scene_dir outputs/scenes/train/ --output_scene_file outputs/scenes/CLEVR_held_out_train.json --start_idx 0 --num_images 500 --render_num_samples 3
blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path outputs/scenes_to_render/tmp_val_iid_sg_for_generation.json --output_image_dir outputs/images/val_iid/ --output_scene_dir outputs/scenes/val_iid/ --output_scene_file outputs/scenes/CLEVR_held_out_val_iid.json --start_idx 0 --num_images 500 --render_num_samples 3
blender --background --python render_images_from_sg.py  -- --full-spec --use_gpu 1 --scene_graphs_path outputs/scenes_to_render/tmp_test_sg_for_generation.json --output_image_dir outputs/images/test/ --output_scene_dir outputs/scenes/test/ --output_scene_file outputs/scenes/CLEVR_held_out_test.json --start_idx 0 --num_images 500 --render_num_samples 3
```

6. Copy over any re-used training images. Note: you will need to update the paths to the CLEVR dataset inside of the file passed as `--reusable_ims_config`:
```
python3 copy_reused_ims.py --output_image_dir outputs/images/train/ --scene_graphs_path outputs/scenes_to_render/tmp_train_sg_for_generation.json --reusable_ims_config dcs_reuse_config.json
```

7. At this point, you can iterate through all the images, by category, by running `test_generated.py`
Note that this is only really practical for small test datasets.

8. Next, merge all scenegraphs into a single file, rather than a folder of .json files.
Use `collect_scenes.py`, from the original CLEVR codebase (see `clevr-dataset-gen/image_generation/collect_scenes.py`)

```
python3 collect_scenes.py --input_dir outputs/scenes/train/ --output_file outputs/scenes/CLEVR_held_out_train.json --date '2022-10-26'
python3 collect_scenes.py --input_dir outputs/scenes/test/ --output_file outputs/scenes/CLEVR_held_out_test.json --date '2022-10-26'
python3 collect_scenes.py --input_dir outputs/scenes/val_iid/ --output_file outputs/scenes/CLEVR_held_out_val_iid.json --date '2022-10-26'
```

9. Generate test split questions:
```
python3 generate_held_out_questions.py test --ho_idx 0 --verbose
python3 generate_held_out_questions.py test --ho_idx 1 --verbose
python3 generate_held_out_questions.py test --ho_idx 2 --verbose
python3 generate_held_out_questions.py test --ho_idx 3 --verbose
python3 generate_held_out_questions.py test --ho_idx 4 --verbose
python3 generate_held_out_questions.py test --ho_idx 5 --verbose
```


10. Generate train questions:

```
python3 generate_held_out_questions.py train_S3 --ho_idx 0 --verbose
python3 generate_held_out_questions.py train_S3 --ho_idx 1 --verbose
python3 generate_held_out_questions.py train_S3 --ho_idx 2 --verbose
python3 generate_held_out_questions.py train_S3 --ho_idx 3 --verbose
python3 generate_held_out_questions.py train_S3 --ho_idx 4 --verbose
python3 generate_held_out_questions.py train_S3 --ho_idx 5 --verbose
python3 generate_held_out_questions.py train_S1_S2 --ho_idx 0 --verbose
python3 generate_held_out_questions.py train_S1_S2 --ho_idx 1 --verbose
python3 generate_held_out_questions.py train_S1_S2 --ho_idx 2 --verbose
python3 generate_held_out_questions.py train_S1_S2 --ho_idx 3 --verbose
python3 generate_held_out_questions.py train_S1_S2 --ho_idx 4 --verbose
python3 generate_held_out_questions.py train_S1_S2 --ho_idx 5 --verbose
```

11. Generate val questions:
```
python3 generate_held_out_questions.py val_iid_S3 --ho_idx 0 --verbose
python3 generate_held_out_questions.py val_iid_S3 --ho_idx 1 --verbose
python3 generate_held_out_questions.py val_iid_S3 --ho_idx 2 --verbose
python3 generate_held_out_questions.py val_iid_S3 --ho_idx 3 --verbose
python3 generate_held_out_questions.py val_iid_S3 --ho_idx 4 --verbose
python3 generate_held_out_questions.py val_iid_S3 --ho_idx 5 --verbose
python3 generate_held_out_questions.py val_iid_S1_S2 --ho_idx 0 --verbose
python3 generate_held_out_questions.py val_iid_S1_S2 --ho_idx 1 --verbose
python3 generate_held_out_questions.py val_iid_S1_S2 --ho_idx 2 --verbose
python3 generate_held_out_questions.py val_iid_S1_S2 --ho_idx 3 --verbose
python3 generate_held_out_questions.py val_iid_S1_S2 --ho_idx 4 --verbose
python3 generate_held_out_questions.py val_iid_S1_S2 --ho_idx 5 --verbose
```

12. Merge questions into final .json files

```
python3 merge_questions.py
```


14. Extract test, val, and test features:
```
python3 extract_img_feats_held_out.py test
python3 extract_img_feats_held_out.py val_iid
python3 extract_img_feats_held_out.py train --only_resnet
python3 extract_img_feats_held_out.py train --exclude_resnet
```
NOTE: Should be run from this directory. There may be some problems
importing vislang; if so then just add to `PYTHONPATH` environment variable
and try again. i.e., `export PYTHONPATH=${PYTHONPATH}:../../vislang/:../../`

NOTE: you can generate the train features in one go, but only if you have ~350GB
of disk space. Otherwise you may want to generate, copy over, then generate the 
rest (hence the two separate calls, ea. requiring about 150GB)

15. Rename 'type' key to 'function' because of changes to the CLEVR dataset between its creation and release (see https://github.com/facebookresearch/clevr-dataset-gen/issues/14#issuecomment-484300688).
```
python3 rename_ques_key.py 0
python3 rename_ques_key.py 1
python3 rename_ques_key.py 2
python3 rename_ques_key.py 3
python3 rename_ques_key.py 4
python3 rename_ques_key.py 5
```

15. Create test, val, and train question hdf5 files for all tests. Should not require a GPU.
```
python3 extract_ques_feats_held_out.py
```

16. Next to `hdf5_held_out_CLEVR`, create `hdf5_held_out_CLEVR_ims` and move
the subdirectories `test_v1.0_img`, `train_v1.0_img`, 
and `val_iid_v1.0_img` 

17. [Optional] To reduce the ammount of RAM memory used by the dataloader, you can trim the scenes
remove any information that is not strictly required for symbolic evaluation (i.e.,
you can remove all info on pixel coordinates, direction vectors, where the image
was reused from, etc...). This script will edit the original file in-place, so you
may want to create a copy of the scene .json files (i.e. the contents 
of `outputs/scenes`) first. Note that this is less information than the CLEVR
scenes contain (e.g., we also remove the `split` key for each scene)

```
python3 trim_scenes.py
```
