# Prediction of Radiological Decision Errors 

This repository contains the code for the paper titled "Prediction of Radiological Decision Errors from Longitudinal Analysis of Gaze and Image Features". The study investigates the potential of using eye-tracking technology to predict diagnostic errors in radiology.

## Overview

Radiologists often face the challenge of making accurate diagnoses based on radiographic images, where errors can lead to significant consequences for patient care. This project explores how gaze patterns and image analysis can be combined using deep learning techniques to better predict errors in radiological decisions.

By harnessing eye-tracking data and image features, this research aims to develop a predictive model that identifies potential inaccuracies in radiologist interpretations, thereby enhancing the diagnostic process.

## Usage

### Dataset

To test on your own data, you need to prepare the data as follows:

1. The **"data_folder"** must contain three files: **train.csv, test.csv,** and **val.csv**. Each of these files (**train.csv, test.csv, val.csv**) should store the path to **"gaze_features.csv"**.

```
data_folder/
├── case_study_A/
│   ├── radA/
│   │   ├── a/
│   │   │   ├── raw_gaze.csv
│   │   │   ├── gru/
│   │   │   │   ├── gaze_features.csv
│   │   ├── b/
│   │   ├── c/
│   ├── radB/
├── train.csv
├── test.csv
├── val.csv
```


