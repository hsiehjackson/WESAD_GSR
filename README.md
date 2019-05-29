# Stress detection with GSR on WESAD dataset

* Download Raw Data

Datasource:https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29
1. ``wget https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download``
2. ``unzip download.zip``
3. Put the folder ``WESAD`` data into ``./data `` folder

* Segmentation

``python src/segmentation/seg_signal.py [seg_pkl_name]``

* Data Preprocessing + Feature extraction

`` python src/preprocess/gsr_extraction.py ./data/seg_data/[seg_pkl_name]``

* Train XGB Binary Class

``python src/train/trainGSR_xgb_binary.py [feature_csv_name]``

* Train XGB Three Class

``python src/train/trainGSR_xgb_three.py [feature_csv_name]``

* Train WESAD Baseline Three Class

``python src/train/trainGSR_Baseline.py [feature_csv_name]``