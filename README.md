
# Sentiment Analysis on IMDB Dataset

## Overview

This project is a sentiment analysis task performed on the IMDB dataset. The notebook applies several machine learning models like Logistic Regression, SVM, and Random Forest to predict whether a review is positive or negative. The analysis involves text preprocessing, vectorization techniques (CountVectorizer and TF-IDF), and evaluation using confusion matrices and classification reports.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Running the Notebook](#running-the-notebook)
- [Models Used](#models-used)
- [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Prerequisites

Make sure you have the following libraries installed:
- Python 3.10+
- pandas
- scikit-learn
- spacy
- nltk
- seaborn
- matplotlib

### Required Commands:
- Install Spacy model:
  ```bash
  !pip install spacy
  !python -m spacy download en_core_web_sm
  ```
- Install NLTK:
  ```bash
  !pip install nltk
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/NLPL_Assignment1.git
   ```

2. Open the notebook in Google Colab or Jupyter Notebook:
   ```bash
   NLPL_Assignment_1_095.ipynb
   ```

3. Install the required packages (already in Google Colab):
   ```bash
   !pip install -r requirements.txt
   ```

## Dataset

The dataset used in this project is the IMDB movie reviews dataset. It contains 50,000 labeled reviews, with an even distribution of positive and negative sentiments.

- Columns:
  - **review**: The actual movie review text.
  - **sentiment**: The label (positive or negative).

The dataset can be downloaded from [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

## Running the Notebook

1. Load the dataset:
   ```python
   df = pd.read_csv('/content/drive/MyDrive/IMDB Dataset.csv')
   ```

2. Preprocess the text data using Spacy:
   ```python
   def clean_text(text):
       text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
       text = re.sub(r'[^A-Za-z\s]', '', text)
       doc = nlp(text.lower())
       tokens = [token.lemma_ for token in doc if not token.is_stop]
       return ' '.join(tokens)
   df['cleaned_review'] = df['review'].apply(clean_text)
   ```

3. Split the dataset into training and testing sets:
   ```python
   X_train, X_test, y_train, y_test = train_test_split(df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42)
   ```

4. Apply vectorization (CountVectorizer and TF-IDF):
   ```python
   count_vectorizer = CountVectorizer()
   tfidf_vectorizer = TfidfVectorizer()
   ```

5. Train and evaluate models using the `evaluate_model` function:
   ```python
   def evaluate_model(model, X_train, X_test, y_train, y_test, vectorizer_name):
       model.fit(X_train, y_train)
       y_pred = model.predict(X_test)
       print(classification_report(y_test, y_pred))
   ```

## Models Used

The following models are implemented:
- **Logistic Regression**: A simple linear model for classification.
- **SVM (Support Vector Machine)**: A robust classifier for binary sentiment classification.
- **Random Forest Classifier**: An ensemble model based on decision trees.

## Evaluation

The models were evaluated based on the following metrics:
- **Confusion Matrix**: Visualized using Seaborn to compare predicted vs. actual results.
- **Classification Report**: Precision, recall, f1-score, and accuracy for both positive and negative classes.
- **Model Comparison**: Bar plots comparing precision, recall, f1-score, and accuracy across different models.

## Results

- **Logistic Regression** (CountVectorizer): 
   - Accuracy: 88%
   - Precision: 87%
   - Recall: 89%
   
- **SVC** (TF-IDF):
   - Accuracy: 90%
   - Precision: 88%
   - Recall: 91%

- **Random Forest Classifier** (TF-IDF):
   - Accuracy: 85%
   - Precision: 85%
   - Recall: 85%

## Conclusion

Among the models tested, the SVM model using TF-IDF Vectorizer yielded the best results with an accuracy of 90%, followed closely by Logistic Regression. Random Forest underperformed in comparison.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
