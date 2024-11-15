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

![bee_img](https://github.com/user-attachments/assets/ab240951-500f-47b9-ba53-5beaaf659712)


## Unit Testing

To run the unit tests, within the inference environment and from the repo root directory do:
```
python -m unittest discover -s test -p 'test_*.py'
```

## Training with Docker

### Requirements

1. The training was done on a single GPU with 8GB VRAM, smaller ones are of course usable.
2. For training and validation logging (as well as visualization), a `wandb` account.
3. The data under `<dataset root>` should contain `img` and `gt-dot` subdirectories with images and ground truth dots, as well as the file lists for the data split.


### Instructions
1. Build the docker:

```
docker build -t bee_docker .
```
2. Run the container:
```
docker run \
    --shm-size=8g \
    --gpus all \
    --net=host \
    -v <dataset_root>:/data \
    -v <folder_to_save_weights>:/trained_models \
    -e WANDB_API_KEY=<wandb_api_key> \
    -e WANDB_PROJECT=<wandb_project> \
    -e WANDB_ENTITY=<wandb_entity> \
    bee_docker:latest 0.0003 16 30
```
Adjust the absolute paths `<dataset_root>` and `<folder_to_save_weights>` as well as the `wandb` enviroment variables `<wandb_api_key>`, `<wandb_project>` and `<wandb_entity>` to your convinience. The last three args stand for learning rate, batch size and train epochs respectively.


Here we can see how the model evolves during training on validation samples. Red dots are model detections, green dots are ground truth points and the heat map represents detection confidence (darker meaning more confident):

![val_viz](https://github.com/user-attachments/assets/27bd6309-88a0-4c9f-89c3-05222cfd649f)

## Testing the model

You can use the same docker image as for training:

```
docker run \
  --shm-size=8g \
  --gpus all \
  --net=host \
  --entrypoint python3 \
  -v <dataset_root>:/data \
  -v <folder_of_weights>:/trained_models \
  bee_docker:latest \
  test_model.py \
  --dataset_root /data \
  --model_weights_path /trained_models/model.pth
```

with your specific absolute paths `<dataset_root>` and `<folder_of_weights>`. 

The chosed model reported the following metrics on the test split:

![image](https://github.com/user-attachments/assets/764a502c-4579-4a96-8245-a61831ec1527)


