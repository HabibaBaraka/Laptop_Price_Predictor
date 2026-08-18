# 💻 Laptop Price Predictor

A Machine Learning web application that predicts the price of a laptop based on its specifications.

## 🚀 Project Overview

Laptop prices depend on many factors such as brand, processor, RAM, storage, GPU, screen resolution, and other specifications.

This project uses Machine Learning to estimate the price of a laptop based on the specifications entered by the user through a simple web interface.

## ✨ Features

- Predict laptop prices using Machine Learning
- Interactive web interface
- Supports different laptop brands and types
- Takes hardware specifications as input
- Displays the predicted price directly
- Clean and responsive UI

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS
- Jupyter Notebook

## 🤖 Machine Learning

The project includes data preprocessing, feature engineering, encoding, scaling, model training, and evaluation.

Several Machine Learning models were evaluated, including:

- Random Forest
- XGBoost
- Gradient Boosting

Hyperparameter tuning was also performed using `RandomizedSearchCV`.

### Model Performance

The final model achieved approximately:

- **R² Score:** 0.82
- **MAE:** 11,356
- **RMSE:** 18,278

> Performance may vary depending on the train/test split and preprocessing configuration.

## 📊 Dataset Features

The model uses laptop specifications such as:

- Company
- Type
- Screen Size
- RAM
- Operating System
- Weight
- Screen Resolution
- Processor
- GPU
- SSD
- HDD
- Flash Storage
- Hybrid Storage
- Touchscreen
- IPS Display
- Retina Display

## 🌐 Web Application

The Flask application provides a user-friendly interface where users can enter laptop specifications and receive an estimated price.

## 📁 Project Structure

```text
Laptop_Price_Predictor/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
├── templates/
├── static/
├── data/
├── screenshots/
│
└── notebooks/
