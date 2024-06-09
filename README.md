# Edusign Model - American Sign Languange Translation

This project aims to translate American Sign Language (ASL) videos into text using landmark detection and machine learning models in Python and TensorFlow. The project involves extracting frames from the video, applying landmark detection to these frames, and predicting the sign language based on the processed video frames.

# Dataset
For the Edusign ASL letter model, we are combining two datasets: the Google ASL dataset and the WASL dataset. The combined dataset includes 78,501 videos with 201 labels. This merger enhances the model's accuracy and robustness by leveraging the strengths of both datasets.

- https://www.kaggle.com/competitions/asl-signs
- https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

For the Edusign ASL alphabet model, we are using a dataset that contains 27,000 images, each with a size of 512x512 pixels. The data is divided into training and testing sets, with each set containing 27 folders representing the 27 labels.

- https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data 

# Research Method

# Model Architecture
For the Edusign ASL letter model, ..

For the Edusign ASL alphabet model, we're using EfficientNetV2B1 as a base model, which has been pretrained with ImageNet weights. After the base model, we add a batch normalization layer, a flatten layer, two dense layers, and one dropout layer, followed by the output layer. 

Since the model is too large for this repository, here is a link to the model: https://drive.google.com/drive/folders/17nspZZdu3bRWVUu10_xaDrBFMEO8_FC0?usp=sharing

# Requirements
- Python
- Pandas
- Numpy
- Sklearn (scikit-learn)
- MediaPipe
- TensorFlow
- OpenCV
- Matplotlib
- Seaborn
- Tqdm

# Usage

# References

# Authors

This project is developed by C241-PS015 Team Bangkit as part of Bangkit Product Capstone.
1. M006D4KY2955 – Yehezkiel Stephanus Austin
2. M006D4KY2954 – Juan Christopher Young
3. M006D4KY2953 – Haikal Irfano
