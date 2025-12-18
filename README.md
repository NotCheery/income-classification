# 💰 Income Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

## 📌 Overview
This project leverages machine learning to analyze census data and predict financial demographics. Specifically, it uses **Logistic Regression** to classify whether an individual earns **> $50k/year** based on features like age, education, occupation, and capital gain.

The goal is to build a robust pipeline that processes raw census data and outputs binary classifications with high accuracy.

## 📂 Dataset
* **Source:** U.S. Census Bureau (commonly known as the "Adult" dataset).
* **Target Variable:** `income` (Binary: `<=50K` or `>50K`).
* **Features:** Age, Workclass, Education, Marital Status, Occupation, Relationship, Race, Sex, Capital Gain/Loss, Hours per week, Native Country.

## 🛠️ Tech Stack
* **Language:** Python
* **Environment:** Jupyter Notebook
* **Libraries:**
    * `pandas` & `numpy` (Data Manipulation)
    * `scikit-learn` (Modeling & Preprocessing)
    * `matplotlib` / `seaborn` (Visualization)

## ⚙️ Methodology

### 1. Data Preprocessing
Raw census data is rarely model-ready. Key steps included:
* **Handling Nulls:** Imputing or removing missing values.
* **Encoding:** Converting categorical variables (like `Education` or `Marital Status`) into numerical formats.
* **Scaling:** Applied `StandardScaler` to normalize numerical features.
    > *Note: Scaling is critical for Logistic Regression convergence and correct regularization (C parameter).*

### 2. Model Architecture
We implemented a **Logistic Regression** classifier.
* **Splitting:** Data was split into training and testing sets to prevent overfitting.
* **Regularization:** Adjusted the `C` parameter to control model complexity.
* **Validation:** Used `train_test_split` with a fixed random state for reproducibility.

### 3. Evaluation
The model is evaluated using:
* **Log Loss:** To measure the uncertainty of the probabilities.
* **Accuracy Score:** The percentage of correct predictions.

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/NotCheery/income-classification.git
    cd income_prediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib
    ```
3.  **Launch the Notebook:**
    ```bash
    jupyter notebook
    ```
    Open `Lab 8.ipynb` and run all cells.

## 📈 Results
* **Current Accuracy:** 79%


---