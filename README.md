# Drone-detection-dataset
Dataset containing IR, visible and audio data that can be used to train and evaluate drone detection sensors and systems.

Video labels: Airplane, Bird, Drone and Helicopter.
Audio labels: Drone, Helicopter and Background.

The dataset contains 90 audio clips and 650 videos (365 IR and 285 visible). If all images are extracted from all the videos the dataset has a total of 203328 annotated images.

Free to download, use and edit.
Descriptions of the videos are found in "Video_dataset_description.xlsx".
The videos can be used as they are, or together with the respective label-files.
The annotations are in .mat-format and have been done using the Matlab video labeler.
Some instructions and examples are found in "Create_a_dataset_from_videos_and_labels.m"

Please cite:  
"Svanström F. (2020). Drone Detection and Classification using Machine Learning and Sensor Fusion".
[Link to thesis](https://hh.diva-portal.org/smash/get/diva2:1434532/FULLTEXT02.pdf)  
or  
"Svanström F, Englund C and Alonso-Fernandez F. (2020). Real-Time Drone Detection and Tracking With Visible, Thermal and Acoustic Sensors".
[Link to ICPR2020-paper](https://arxiv.org/pdf/2007.07396.pdf)  
or  
"Svanström F, Alonso-Fernandez F and Englund C. (2021). A Dataset for Multi-Sensor Drone Detection".
[Link to Data in Brief](https://www.sciencedirect.com/science/article/pii/S2352340921007976#!)

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.5500576.svg)](http://dx.doi.org/10.5281/zenodo.5500576)

Contact:  
DroneDetectionThesis@gmail.com

## Lighthouse Stuff

This repository contains tools to generate a YOLO dataset from the .mat and .mp4 files in the Drone-detection-dataset.

### Installation

Download poetry if you don't have it already, then:

```sh
poetry install
```

Install the submodule:

```sh
git submodule update --init --recursive
```

### Usage

#### Generating a YOLO dataset

To generate a YOLO dataset, first, follow the README in the `MatReader` submodule to build the `MatReader` .NET tool and convert the .mat files into .csv format.

Once the data is in .csv format, you can run the following command to generate a YOLO dataset:

```sh
poetry run python3 csv_to_yolo.py output_V Data/Video_V yolo_dataset -j $(nproc)
```

If you also did the IR videos, you can run:

```sh
poetry run python3 csv_to_yolo.py output_IR Data/Video_IR yolo_dataset -j $(nproc)
```

#### Splitting the dataset into train, val, and test sets

To split the generated YOLO dataset into training, validation, and test sets, you can use the following command:

```sh
python3 split_dataset.py --input yolo_dataset --train 0.8 --val 0.1 --test 0.1
```

If you are actively tuning hyperparameters, you might want to use a larger validation set:

```sh
python3 split_dataset.py --input yolo_dataset --train 0.7 --val 0.2 --test 0.1
```

This will create three folders inside the `yolo_dataset` folder: `train`, `val`, and `test`, each containing the respective images and labels.

The dataset.yaml file will also be updated to reflect the new structure.

#### Training a YOLO model

To train a YOLO model using the generated dataset, you can use the following command:

```sh
 poetry run python train_temporal_yolo.py \
   --model yolov10n \
   --frames 5 \
   --data yolo_dataset/dataset.yaml \
   --output ./runs/temporal \
   --batch 16 \
   --epochs1 10 \
   --epochs2 40 \
   --imgsz 640
```
