# Bees Challenge

## Challenge Description

We want to count the number of bees flying around in order to monitor the hive. You are given a sample set of images
of bees, along with ground truth labels containing dots at the centroids of each bee in the image. The goal of this
challenge is to automate the process of counting bees in a given image.

## Solution Description

The model [CenterNet](https://arxiv.org/abs/1904.07850) was chosen as the base of the solution, after modifying its bounding-box-prediciton variant to the task of detecting bee centers and setting `resnet18` as its backbone. The implementation was based on [this repo](https://github.com/tteepe/CenterNet-pytorch-lightning/tree/main) that makes use of `pytorch-lighting` to simplify the code. The logging, model selection and evalation were done with `wandb`.


## Inference

The Torchscript file of the trained model resides under `trained_models`. Follow these steps top use it directly:

1. Create the inference environment: `conda env create -f  run_environment.yaml`
2. Activate it: `conda activate bee_inference`
3. Run inference on a bee input image: 
```
python solution.py --model trained_models/model.pt --image <path/to/input/image>
```
A visualization image with the bee counter will be saved as `bee_img.png`:

![bee_img](https://github.com/user-attachments/assets/8503f7d6-0d0f-4b89-b824-05fdb2648e86)


## Unit Testing

To run the unit tests, within the inference environment and from the repo root directory do:
```
python -m test.test_bee_detector
```

## Training

### Requirements

1. The training was done with CUDA 11.8, it's recommended to use the same
2. For training and validation logging (as well as visualization), a `wandb` account
3. The data under `<dataset root>` should contain `img` and `gt-dot` subdirectories with images and ground truth dots, as well as the file lists for the data split.

The training was done on a single GPU with 8GB VRAM, smaller ones are of course usable.

### Instructions

 0. Generate COCO files from list splits under the folder `data/honeybee` by simply running
   ```
   python tools/prepare_data_from_file_lists.py --data_root data/honeybee/
   ```
1. Create the training environment: `conda env create -f  environment.yaml`
2. Activate it: `conda activate bees_env`
3. Add the base repo as submodule:
    * `git submodule add git@github.com:tteepe/CenterNet-pytorch-lightning.git lib`
    * `git submodule update --init --recursive`
    * `pip install -e lib/`
    * Due to some library versions conflict, it's necessary to replace the import line

      ```from DCN.dcn_v2 import DCN```

      by

      ```from mmcv.ops.deform_conv import DeformConv2d as DCN```

      in the files `lib/CenterNet/models/backbones/resnet_dcn.py`and `lib/CenterNet/models/backbones/pose_dla_dcn.py`

4. wandb logging: `wandb login`
5. Start training with e.g.
```
python run_bee_exp.py --dataset_root data/honeybee/ --learning_rate 0.0004
```

Here we can see how the model evolves during training on validation samples. Red dots are model detections, green dots are ground truth points and the heat map represents detection confidence (darker meaning more confident):

![val_viz](https://github.com/user-attachments/assets/27bd6309-88a0-4c9f-89c3-05222cfd649f)
