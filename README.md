# Flower-Species-Classification-CNN

A deep learning project using a Convolutional Neural Network (CNN) to classify flower species from images. Trained on the **TensorFlow Flowers dataset**, this model learns to accurately identify five different flower types using end-to-end supervised learning.

---

## 🌸 Project Overview

This project implements a custom CNN architecture to classify color images of five flower species. It leverages the TensorFlow/Keras framework to build, train, and evaluate a model using image data sourced from the publicly available `tf_flowers` dataset.

The goal of this project is to demonstrate practical image classification using CNNs and gain experience in managing image preprocessing, model tuning, and performance evaluation in a computer vision context.

---

## 🧠 Model Summary

- **Architecture**: Custom CNN built using `tf.keras.Sequential`
- **Layers**: Convolutional layers with ReLU activation, MaxPooling, Dropout, Fully Connected (Dense) layers, and Softmax output
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Evaluation**: Accuracy, Training/Validation Loss Curves

---

## 📈 Results

The final trained model achieved high classification accuracy on the validation set, demonstrating strong generalization across all five flower species.

Visual outputs include:
- Training vs. validation accuracy/loss plots
- Impact of Hyperparameter Tuning 

---

## 📂 Contents

- `notebooks/`: Jupyter notebooks for preprocessing, model training, evaluation
- `dataset/`: Dataset information and download links
- `reports/`:  Final report with problem statement, methodology, and results

---

## 📊 Dataset

The dataset contains **3,670 color images** across five categories:

- Daisy  
- Dandelion  
- Roses  
- Sunflowers  
- Tulips  

📥 Download from TensorFlow:  
[http://download.tensorflow.org/example_images/flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)

For more details, see [`dataset/dataset-info.md`](./Dataset/dataset-info.md)

---

## 🛠 Tools & Technologies

- Python, TensorFlow, Keras
- NumPy, Matplotlib, Seaborn
- Jupyter

---

## 📘 License

This project is intended for academic and educational purposes only. Please refer to the dataset’s [license and usage terms](https://www.tensorflow.org/datasets/catalog/tf_flowers) before using.

