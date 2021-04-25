# BDNN-VehicleX
BiDirectional Neural Networks for vechicle classification

[![BDNN-VehicleX AutoTest](https://github.com/skyguidance/BDNN-VehicleX/actions/workflows/autotest.yml/badge.svg)](https://github.com/skyguidance/BDNN-VehicleX/actions/workflows/autotest.yml)
### One Click Approach

This repo has Github Actions enabled. Simply run the DevOps Task.

### Train your own model

Use this code repo could reproduce all the results that stated in the original paper.

* **Confirm your environment.** Basically any Python 3 version should be compatable as long as numpy, scipy, pyyaml, matplotlib and Pytorch installed. 

* **Download Dataset.** This project uses a synthesis dataset that genereated by VehicleX, which is a publicly available 3D engine. A total of 1,362 vehicles are annotated, which includes 45,438 samples for training, 14,936 for validation and 15,142 for testing.

  [Download Link](http://cs.anu.edu.au/~tom/datasets/vehicle-x.zip)

  Please unzip the file and place into the dataset/vehicle-x folder.

* **Specify a Task.** Some predifined task's YAML config files are placed under experiments folder. These are tasks that presented in the paper. Based on those templates, you can specify your own task.

* **Train a model.**  Once you have a YAML file, execute the following commands to produce the result.

  ```shell
  python main.py -c experiments/config.yaml
  ```

  Model weights for both Best Top-1 and Best Top-5 will saved under checkpoint folder, as long as the graph.

  Once training process done, it will automatically evaluate on Test-set.

### Pretrained Weights

Download all pretrained weight file [click here (Google Drive)](https://drive.google.com/drive/folders/19mvKVqIdI6BuqqkO-nMWtU3x9ivCeorP?usp=sharing).

### Reference

1. Some utility code adpot from previous project [Landmark-2020](https://github.com/skyguidance/ENGN8501-Landmark-2020).
2. Dataset originally from [VehicleX](https://github.com/yorkeyao/VehicleX), this project uses a pre-processed version that avilable on the link above.
3. BiDirectional Neural Networks (BDNN) is based on paper: [Bidirectional neural networks and class prototypes](https://ieeexplore.ieee.org/document/487348/), by [A.F. Nejad](https://ieeexplore.ieee.org/author/38156108300); [T.D. Gedeon](https://ieeexplore.ieee.org/author/37271327900).

