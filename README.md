# Edusign Model - American Sign Languange Translation

This project aims to translate American Sign Language (ASL) videos into text using landmark detection and machine learning models in Python and TensorFlow. The project involves extracting frames from the video, applying landmark detection to these frames, and predicting the sign language based on the processed video frames.

# Dataset
For the Edusign ASL letter model, we are combining three datasets: the Google ASL dataset and two WLASL datasets. The combined dataset includes 78,501 videos with 201 labels. This merger enhances the model's accuracy and robustness by leveraging the strengths of all three datasets.

- https://www.kaggle.com/competitions/asl-signs
- https://www.kaggle.com/datasets/risangbaskoro/wlasl-processed
- https://www.kaggle.com/datasets/sttaseen/wlasl2000-resized

# Research Method

# Model Architecture
For the Edusign WLASL letter model, we use a model architecture structured as follows:

- Input Layer: The initial layer to receive input images.
- Masking Layer: A layer to handle masking operations.
- Channel Addition Layer: A layer to add new channels to the input data.
- Base Model: EfficientNetV2B1 pretrained on ImageNet.
- Global Average Pooling 2D: A pooling layer to reduce each feature map to a single value.
- Early Late Dropout: Dropout layers for regularization, applied both before and after certain epoch.
- Output Layer: The final layer to produce the classification output.

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
