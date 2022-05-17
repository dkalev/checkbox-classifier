# checkbox-classifier

A simple CNN classifier, determining the state of a check box

## Setup

```bash
checkbox-classifier
├── ./data
│   ├── ./data/checked
│   ├── ./data/ds_stats.json
│   ├── ./data/label_mapping.json
│   ├── ./data/other
│   ├── ./data/splits.csv
│   └── ./data/unchecked
├── ./dataset
│   ├── ./dataset/dataset.py
│   ├── ./dataset/preprocess.py
│   └── ./dataset/__init__.py
├── ./Dockerfile.infer
├── ./Dockerfile.train
├── ./EDA.ipynb
├── ./hparam_search.py
├── ./inference.py
├── ./models
│   ├── ./models/base.py
│   ├── ./models/baseline.py
│   ├── ./models/mobilenet.py
│   ├── ./models/resnet.py
│   └── ./models/__init__.py
├── ./README.md
├── ./requirements.txt
├── ./requirements_dev.txt
└── ./train.py
```

### Local

Requires Python 3.9

1. Create a virtual environment:

   `python -m venv env`

2. Activate the environment and install requirements

   `src env/Scripts/activate` (Windows)

   `pip install -r requirements.txt`

   `pip install -r requirements_dev.txt` (Optional for EDA notebook)

### Docker

Alternatively for training and inference you can use the docker images provided.

#### Training

1. Build image

   `docker build --pull --rm -f "Dockerfile.train" -t checkboxclassifier:train "."`

2. Run training

   `docker run --shm-size=256m --name cbtrain checkboxclassifier:train --model=mobilenet --batch_size=32 --num_workers=2`

3. Copy results locally

   `docker container ls`

   `docker cp cbtrain:/usr/src/app/logs <<LOCAL DESTINATION PATH>>`

#### Inference

1. Build image

   `docker build --pull --rm -f "Dockerfile.infer" -t checkboxclassifier:infer "."`

2. Run container with inference REST API

   `docker run -d -p 8000:8000 checkboxclassifier:infer`

3. Make predictions by passing a local image file

   `curl -X POST http://127.0.0.1:8000/predict/ -H "Content-Type: multipart/form-data" -F "file=@<<FILEPATH>>"`

## Dataset preprocessing and EDA

The dataset is downloaded in the following structure:

`splits.csv` contains a mapping from the relative location of each image in the dataset to its correspondent split. This gives more flexibility and can easily be extended to use the dataset in k fold cross validation setting.

It is split into train, valid and test subsets with ratios of 0.6, 0.2, 0.2 respectively. Since the dataset is imbalanced, stratified sampling is used to ensure the dataset splits have the same distribution of labels.

Check the jupyter notebook `EDA.ipynb` for some data exploration.

## Models

We compare 3 models: a custom baseline CNN model, a pretrained MobilenetV2 and a pretrained Resnet50. All the three models share the same classification head.

### Classifier head

### Baseline CNN architecture

- 239 K trainable parameters, 239 K total parameters, model size 0.959 MB

A simple CNN architecture using three blocks of Convolutional layer followed by BatchNorm and ReLU activation. Instead of pooling layers, we use strided convolutions to downsample the images, which gives the model a bit more expressive power.

### MobileNetV2 (Finetuing)

- 1.1 M trainable parameters, 2.5 M total parameters, model size 9.922 MB

The architecture is optimized for performance, while maintaining high performance. We are using weights pretrained on the ImageNet dataset and freezing all the layers except the last two, which are trained further with the classification head.

### Resnet50 (Finetuning)

- 3.8 M trainable parameters, 23.9 M total parameters, model size 95.673 MB

The biggest and most expressive architecture, can provide better results, at the expense of slower inference and bigger hardware requirements. Again, we are using weights pretrained on the ImageNet dataset and freezing all the layers except the last two, which are trained further with the classification head.

## Hyperparameter search

Using Optuna with a [TPESampler](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.samplers.TPESampler.html) to explore the hyperparameter space of the following:

- model [baseline, mobilenet, resnet50]
- batch size [8, 32, 64]
- learning rate [1e-5: 1e-2] (log scale)
- weight decay [1e-6: 1e-3] (log scale)

Conducted 50 trials of 10 epochs each.

## Training

### Data augmentations

Since a lot of the images in the dataset have width larger than height, simply resizing them to a square format can cause some issues. For example, sliders might look like a square, or a checkbox might look like a vertical slider. To mitigate these issues, we first pad the images to a square by reflecting the image until it fills a square. Then, only if necessary resize the image, which preserves the original aspect ratio. All images are normalized, with using the dataset statistics for the baseline model and ImageNet statistics for the pretrained models. The set of training augmentations is small as most actually had a negative impact on performance. The augmentations used are random changes in the brightness and hue of the photos and random translations.

### Experimental setup

For training we used the Adam optimizer with CrossEntropy loss. For metrics we report Accuracy and Weighted F1 score (similar to macro, however weighted by the support of each class). The hyperparameters used are the following:

- model: MobileNetV2
- batch size: 8
- learning rate: 0.00093
- weight decay: 0.000001514

|       | Accuracy | Weighted F1 score |     |     |
| ----- | -------- | ----------------- | --- | --- |
| Train | 0.945    | 0.969             |     |     |
| Valid | 0.882    | 0.921             |     |     |
| Test  | 0.833    | 0.888             |     |     |

### Model weights

The trained model weights can be accessed from:

https://drive.google.com/uc?export=download&id=11dbv1XSgiR1QI50fzXwbZnRnfOPy_Ts3

They are already used in the inference script.

## Next steps

- More extensive hyperparameter search
- Improving dataset quality
- Reducing the impact of the dataset class imbalances by weighted sampling or loss function
- Reducing size of Docker images
- Improving code quality by adding python docs and testing, extracting env variable
- Preparing inference REST API for production
