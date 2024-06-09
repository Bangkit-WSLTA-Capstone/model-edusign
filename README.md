# Edusign Model - American Sign Languange Translation

This project aims to translate American Sign Language (ASL) videos into text using landmark detection and machine learning models in Python and TensorFlow. The project involves extracting frames from the video, applying landmark detection to these frames, and predicting the sign language based on the processed video frames.

# Dataset
For Edusign ASL letter model, we are using two datasets: one is the google asl dataset, and the other is a wasl dataset. We are taking this approach to test the model with a real world video application. The google asl dataset contains 94,477 rows of xyz landmark and 250 labels.  

- **Train Validation Dataset** = https://www.kaggle.com/competitions/asl-signs
- **Additional Test Dataset** = https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed

For the Edusign ASL alphabet model, we are using a dataset that contains 27,000 images, each with a size of 512x512 pixels. The data is divided into training and testing sets, with each set containing 27 folders representing the 27 labels.

- https://www.kaggle.com/datasets/lexset/synthetic-asl-alphabet/data 

# Research Method

# Model Architecture

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
