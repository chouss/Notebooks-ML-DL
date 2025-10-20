# Brain Tumor Segmentation using Fine-Tuned nn-U-Netv2

This project enhances brain tumor segmentation by fine-tuning the powerful **nn-U-Netv2** architecture. By leveraging deep learning on MRI scans, we achieve highly accurate and reliable tumor detection, providing a valuable tool for medical professionals.

A simple, user-friendly interface has also been developed, making the model accessible for quick MRI analysis and tumor visualization.

---

## Key Features

* **Advanced Architecture:** Utilizes both standard and ResEnc configurations of nn-U-Netv2 for robust performance.
* **Ensemble Methods:** Employs Max Probability and Probability Averaging techniques to combine model outputs, leading to smoother and more precise segmentation.
* **User-Friendly Interface:** A simple GUI allows healthcare professionals to easily upload MRI scans and visualize segmentation results.
* **High-Quality Training Data:** Trained and evaluated on the BraTS 2020 and BraTS 2021 datasets.

---

## Results

The model was evaluated on the **BraTS 2021 dataset**, demonstrating strong performance in identifying and outlining tumor regions.

* **Dice Score:** Achieved a mean Dice score of **84.45%**, indicating excellent overlap between predicted and ground truth tumors.
* **Intersection over Union (IoU):** Reached a mean IoU of **76.78%**, confirming high volumetric accuracy.
* **Sensitivity:** Demonstrated a high sensitivity of approximately **88.6%**, showing its effectiveness in correctly detecting actual tumor tissue.

---

## Dataset

The model was trained using the BraTS 2021 Task 1 dataset, which can be downloaded from Kaggle:

* [**BraTS 2021 Task 1 Dataset**](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)