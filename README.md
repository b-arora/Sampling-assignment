# Sampling Techniques for Imbalanced Datasets

## Project Objective
The objective of this assignment is to understand the importance of sampling techniques in handling imbalanced datasets and to analyze how different sampling strategies affect the performance of various machine learning models.

In real-world scenarios, datasets like Credit Card Fraud detection are highly imbalanced (e.g., 99% legitimate transactions vs. 1% fraud). This project balances the dataset and evaluates five different sampling techniques across five different machine learning models to determine the optimal combination for accuracy.

## Dataset
* **Source:** [Creditcard_data.csv](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv)
* **Description:** The dataset contains transactions made by credit cards. It is highly imbalanced, with the target class `Class` indicating whether a transaction is fraudulent (1) or valid (0).

## Methodology

### 1. Data Balancing
Since the original dataset is skewed, we first balance the data to ensure fair training.
* **Technique Used:** Random Over Sampling (`RandomOverSampler` from `imbalanced-learn`).
* **Result:** Both classes (0 and 1) have an equal number of records in the training set.

### 2. Sample Size Calculation
We determined the sample size using **Cochran’s Formula** with the following parameters:
* Confidence Level: 95%
* Margin of Error: 5%
* Proportion (p): 0.5

### 3. Sampling Techniques Applied
We created five distinct samples from the balanced dataset using the following methods:
1.  **Sampling1:** Simple Random Sampling
2.  **Sampling2:** Systematic Sampling
3.  **Sampling3:** Stratified Sampling
4.  **Sampling4:** Cluster Sampling
5.  **Sampling5:** Bootstrap Sampling (Replacement)

### 4. Machine Learning Models
We evaluated the performance of the samples on the following models:
* **M1:** Logistic Regression
* **M2:** Decision Tree Classifier
* **M3:** Random Forest Classifier
* **M4:** Support Vector Machine (SVM)
* **M5:** Naive Bayes (GaussianNB)

## Installation & Usage

### Prerequisites
To run the code, you need Python installed along with the following libraries:

```bash
pip install pandas numpy scikit-learn imbalanced-learn
