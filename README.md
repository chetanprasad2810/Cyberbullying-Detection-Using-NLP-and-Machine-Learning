# Cyberbullying-Detection-Using-NLP-and-Machine-Learning
Cyberbullying Detection using NLP and Machine Learning
This repository contains a Google Colab notebook that demonstrates a machine learning-based approach to detect and classify cyberbullying text. The project leverages Natural Language Processing (NLP) techniques and traditional machine learning algorithms to identify harmful and abusive language in user-generated content.

Project Objective
The primary goal of this project is to build an efficient and accurate classification model that can automatically detect and flag cyberbullying content. By applying this model, online platforms can be better equipped to maintain safer and more respectful digital spaces.

Technologies and Libraries
The project is built using Python and several key libraries for data manipulation, NLP, and machine learning:

Pandas: For data loading and manipulation.

NumPy: For numerical operations.

Matplotlib & Seaborn: For data visualization.

NLTK (Natural Language Toolkit): For text preprocessing, including stop word removal, tokenization, and stemming.

Scikit-learn: The primary library for machine learning, used for:

Preprocessing and feature extraction (TfidfVectorizer).

Model training (LogisticRegression, MultinomialNB, SVC).

Model evaluation (accuracy_score, confusion_matrix, classification_report).

Methodology
The notebook follows a standard machine learning pipeline for text classification:

Data Loading & Exploration: The project starts by loading a CSV dataset containing user-generated text and their corresponding labels (bullying or not bullying). It then performs an initial analysis to understand the data distribution.

Text Preprocessing: The raw text data is cleaned and prepared for the model. This includes:

Removing punctuation and special characters.

Converting text to lowercase.

Removing common English stop words (e.g., "the," "is," "a").

Stemming words to their root form.

Feature Extraction: The preprocessed text is transformed into numerical feature vectors using TF-IDF (Term Frequency-Inverse Document Frequency). This technique weighs the importance of a word in a document relative to its frequency in the entire dataset.

Model Training & Evaluation: The dataset is split into training and testing sets. Three different classification models are trained on the TF-IDF features and then evaluated on the test set:

Logistic Regression: A linear model for binary classification.

Multinomial Naive Bayes: A probabilistic classifier that is highly effective for text classification.

Support Vector Machine (SVM): A powerful model that finds the optimal hyperplane to separate data classes.

Results and Analysis: The performance of each model is measured using metrics like accuracy, precision, recall, and the F1-score. A confusion matrix is also generated to visualize the models' performance in classifying positive and negative instances. The best-performing model is identified for the task.

How to Use the Notebook
Open the Notebook: Click on the link below to open the project directly in Google Colab.

Open in Google Colab

Run All Cells: Once the notebook is open, you can run all the cells sequentially by going to the menu and selecting Runtime > Run all.

Explore the Code: You can also run individual cells to see the output and modify the code as you wish. The notebook is well-commented to guide you through each step of the process.

Summary
This project serves as a practical example of applying machine learning to a real-world problem. By following the steps in the notebook, you can gain a deeper understanding of the entire workflow, from data preprocessing to model evaluation, and see how different algorithms perform on the task of cyberbullying detection.
