# checkbox-classifier
A simple CNN classifier, determining the state of a check box

## Setup

### Local

Requires Python 3.9.

1. Create a virtual environment:

`python -m venv env`

2. Activate the environment

3. Install requirements

`pip install -r requirements.txt`

`pip install -r requirements_dev.txt`

### Docker

#### Training

`docker build --pull --rm -f "Dockerfile.train" -t checkboxclassifier:train "."`

`docker run --shm-size=256m checkboxclassifier:train --model=mobilenet --batch_size=32 --num_workers=2`

`docker container ls`

`docker cp <<CONTAINER ID>>:/usr/src/app/logs <<LOCAL DESTINATION PATH>>`

#### Inference

`docker build --pull --rm -f "Dockerfile.infer" -t checkboxclassifier:infer "."`

`docker run -d -p 8000:8000 checkboxclassifier:infer`

`curl -X POST http://127.0.0.1:8000/predict/ -H "Content-Type: multipart/form-data" -F "file=@<<FILEPATH>>"`

## Dataset preprocessing and EDA

The dataset is downloaded in the following structure:

`splits.csv` contains a mapping from the relative location of each image in the dataset to its correspondent split. This gives more flexibility and can easily be extended to use the dataset in k fold cross validation setting.

It is split into train, valid and test subsets with ratios of 0.6, 0.2, 0.2 respectively. Since the dataset is imbalanced, stratified sampling is used to ensure the dataset splits have the same distribution of labels.

Check the jupyter notebook `EDA.ipynb` for some data exploration.


## Models



### Baseline CNN architecture

* 239 K trainable parameters, 239 K total parameters, model size 0.959 MB

### MobileNetV2 (Finetuing)

* 1.1 M trainable parameters, 2.5 M total parameters, model size 9.922 MB

### Resnet50 (Finetuning)

* 3.8 M trainable parameters, 23.9 M total parameters,  model size 95.673 MB

## Hyperparameter search
Using Optuna, 10 epochs, Bayesian search, model, batch size, learning rate, weight decay

## Training

## Inference
