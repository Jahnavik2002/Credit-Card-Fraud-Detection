# Credit Card Fraud Detection

## Project Overview

This project aims to identify fraudulent credit card transactions using machine learning techniques. The dataset consists of anonymized transaction records with labeled data indicating whether a transaction is fraudulent. The objective is to train a model that accurately distinguishes between legitimate and fraudulent transactions to help financial institutions prevent fraud.

## Dataset

The dataset used in this project is a collection of credit card transactions, with features representing various attributes of each transaction. Here are some details:

- **Features**: Anonymized numerical variables.
- **Target**: A binary variable indicating whether a transaction is fraudulent (`1`) or legitimate (`0`).

## Project Structure

The project consists of the following components:

- **Data Preprocessing**: Includes handling missing values, scaling features, and balancing classes (fraudulent transactions are typically very rare).
- **Exploratory Data Analysis (EDA)**: Visualizations and statistics to understand the characteristics of both fraudulent and non-fraudulent transactions.
- **Modeling**: Training various machine learning models (e.g., Logistic Regression, Decision Trees, Random Forest, XGBoost) to classify transactions as fraud or non-fraud.
- **Evaluation**: Assessing the model's performance using metrics such as accuracy, precision, recall, F1 score, and AUC-ROC curve.

## Requirements

- Python 3.x
- Jupyter Notebook
- Libraries:
  - NumPy
  - Pandas
  - Scikit-Learn
  - Matplotlib
  - Seaborn
  - XGBoost (if used in modeling)

To install the required libraries, run:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. **Run the Jupyter Notebook**:
   Open the `Credit Card Fraud Detection.ipynb` file to view the code, execute cells, and interact with the project.

## Model Training and Evaluation

- **Training**: The models are trained on the preprocessed dataset, using various techniques to address the imbalance between fraud and non-fraud transactions.
- **Evaluation**: Each model is evaluated using cross-validation and confusion matrix analysis. The metrics focus on the model’s ability to correctly classify fraudulent transactions, emphasizing precision and recall to minimize false positives and false negatives.

## Results

The final model achieved the following scores:

- **Accuracy**: _e.g., 99%_
- **Precision**: _e.g., 98%_
- **Recall**: _e.g., 92%_
- **F1 Score**: _e.g., 95%_

(Replace these values with the actual metrics from your model.)

## Acknowledgments

This project is based on publicly available credit card transaction data, anonymized for privacy reasons. The notebook leverages common machine learning practices to enhance model performance and optimize for high-stakes detection scenarios.



