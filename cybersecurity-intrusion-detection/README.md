# Cybersecurity Intrusion Detection with Machine Learning

## 1. Project Overview

This project aims to build a machine learning model to predict malicious cyber attacks based on network traffic and session data. The goal is to create a reliable classifier that can distinguish between normal and malicious activity, helping to enhance network security.

The prediction is based on the following features:
- `session_id`
- `network_packet_size`
- `protocol_type`
- `login_attempts`
- `session_duration`
- `encryption_used`
- `ip_reputation_score`
- `failed_logins`
- `browser_type`
- `unusual_time_access`

---

## 2. Dataset

The dataset used for this project is the "CyberSecurity Intrusion Detection Dataset," publicly available on Kaggle.

- **Source**: [Kaggle Dataset Link](https://www.kaggle.com/datasets/dnkumars/cybersecurity-intrusion-detection-dataset/data)

---

## 3. Methodology

Several baseline machine learning models were trained to tackle this binary classification problem. To optimize performance, hyperparameter tuning was performed using **Grid Search with Cross-Validation** (`GridSearchCV`).

The models were evaluated based on four key metrics: **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

---

## 4. Results

After training and tuning, the **Random Forest Classifier** achieved the best performance among the tested models.

The final evaluation metrics for the optimized Random Forest model on the test set are as follows:

| Metric    | Score   |
| :-------- | :------ |
| Accuracy  | 89.5%   |
| Precision | 99.4%   |
| Recall    | 77.4%   |
| F1-Score  | 87.0%   |