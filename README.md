# Amazon Reviews Sentiment Analysis

**Dataset**: Amazon Reviews Dataset  
**Dataset URL**: https://www.kaggle.com/datasets/bittlingmayer/amazonreviews

## Project Overview

This notebook provides a step-by-step guide for downloading, processing, and performing sentiment analysis on the Amazon Reviews dataset. The main tasks in this notebook include data ingestion, preprocessing, model training, and evaluation. Below are the key steps covered:

1. **Library Imports**:
   - Essential libraries like `pandas`, `numpy`, `matplotlib`, `seaborn`, `nltk`, and `sklearn` are imported for data manipulation, visualization, and model evaluation.

2. **Setup Kaggle API**:
   - Configuring Kaggle API credentials using `kaggle.json` to enable downloading datasets from Kaggle.

3. **Downloading the Dataset**:
   - The Amazon Reviews dataset is downloaded directly from Kaggle using the Kaggle API.

4. **Unzipping the Dataset**:
   - The downloaded `.zip` file is extracted into the environment.

5. **Extracting Compressed Files**:
   - The `.bz2` compressed files (`train.ft.txt.bz2` and `test.ft.txt.bz2`) are extracted and saved in their original `.txt` formats.

6. **Loading Data into a DataFrame**:
   - The extracted `.txt` files are loaded into a pandas DataFrame with two columns: **label** (sentiment label) and **review** (text of the review).

7. **Preprocessing**:
   - Data preprocessing includes text cleaning, tokenization, and transformation steps for preparing the reviews for machine learning models.

8. **Model Training**:
   - Machine learning models, including **Multinomial Naive Bayes (MNB)** and **Logistic Regression (LR)**, are trained on the preprocessed review data.

9. **Model Evaluation**:
   - The models' performance is evaluated using the **Receiver Operating Characteristic (ROC)** curve, and the **Area Under the Curve (AUC)** is calculated to assess the accuracy of the models.

## Requirements

To run this notebook, you will need the following Python libraries:
- `pandas` (for data manipulation)
- `numpy` (for numeric operations)
- `matplotlib` (for plotting and visualization)
- `seaborn` (for statistical data visualization)
- `nltk` (for text processing)
- `sklearn` (for machine learning tasks)

You can install these libraries using `pip`:

```
pip install pandas numpy matplotlib seaborn nltk scikit-learn
```



## Steps to Run the Notebook

1. Set up your Kaggle API credentials by uploading the `kaggle.json` file.
2. Run the code in the second cell to set up the Kaggle API.
3. Use the code in the third cell to download the dataset from Kaggle.
4. Extract and load the data as shown in the subsequent cells.
5. After loading the dataset, you can begin preprocessing the text data for machine learning tasks.
6. Train the models and evaluate their performance using the ROC curve.

s
