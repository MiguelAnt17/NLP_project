# Natural Language Processing: Sentiment Analysis of Portuguese Twitter Dialogues

This repository contains a comprehensive project on Natural Language Processing (NLP), developed for the NLP course at the University of Coimbra. The project is divided into three main parts (Meta 1, Meta 2, and Meta 3), each building upon the last. The primary goal is to perform sentiment analysis and regression on a dataset of Portuguese Twitter dialogues, exploring a variety of Machine Learning (ML) and state-of-the-art NLP techniques.

**Project Overview**

The project follows a structured, three-stage approach:
- Meta 1: Data Annotation and Corpus Analysis: Annotation of a Twitter dialogue dataset for sentiment, followed by a statistical and linguistic analysis of the corpus.
- Meta 2: Supervised Machine Learning for Regression: Development and evaluation of classic regression models (such as KNN, Decision Tree, Random Forest, and SVM) using different text representations (TF, TF-IDF, and Word2Vec).
- Meta 3: Prompt Engineering with Large Language Models (LLMs): Application of pre-trained transformer-based models (Llama 3.2 and Mistral) to perform the same regression task using Zero-Shot and Few-Shot learning techniques.

Throughout the project, we compare the performance, methodologies, and outcomes of traditional ML approaches versus modern LLMs.


**Meta 1: Data Annotation and Corpus Analysis**

In the first stage, our focus was on preparing the dataset for analysis. We manually annotated a corpus of Twitter dialogues, assigning sentiment scores to each message. This phase was crucial for creating the ground truth for our subsequent regression tasks.

Key Activities:
- Dataset Description: The project uses the TwitterDialogueSAPT dataset, which contains dialogues from Twitter (now X) on various topics, including customer service interactions with the company MEO, discussions about TV series, and everyday conversations.
- Sentiment Annotation: Messages were annotated using a four-category system {-2, -1, 0, 1} to capture both the polarity (positive, negative, neutral) and intensity of the sentiment.
Methodology: We employed PoS (Part-of-Speech) Tagging and Lemmatization to pre-process the text, enabling a richer and more standardized analysis.
- Inter-Annotator Agreement: We measured the consistency of our annotations using metrics like Cohen's Kappa, Fleiss' Kappa, and Krippendorff's Alpha, achieving a moderate to substantial level of agreement.

Files:
- Code: Code_Grupo3.ipynb

**Meta 2: Supervised Machine Learning for Regression**

The second stage involved developing a regression system using supervised ML methods. We trained and evaluated several models to predict the sentiment scores annotated in Meta 1.

Key Activities:
- Text Representation: We experimented with three different text vectorization techniques:
    - Term Frequency (TF): A sparse matrix representation based on word counts.
    - Term Frequency-Inverse Document Frequency (TF-IDF): A sparse matrix that weights words by their importance in the corpus.
    - Word2Vec: A dense vector representation that captures semantic relationships between words.
- Models: We implemented and evaluated the following ML models:
    - K-Nearest Neighbors (KNN)
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
- Evaluation: Model performance was assessed using MAE (Mean Absolute Error), MSE (Mean Squared Error), and RMSE (Root Mean Squared Error). The SVM model with TF representation yielded the best overall performance.

Files:
- Code: CODE_LAST.ipynb

**Meta 3: Prompt Engineering with Large Language Models**

In the final stage, we shifted our focus to modern, transformer-based Large Language Models (LLMs) to tackle the same regression task. This allowed us to compare their performance against traditional ML models, especially in scenarios with limited or no specific training.

Key Activities:
- Prompt Engineering Techniques: We utilized two primary approaches:
    - Zero-Shot Learning: The models were asked to predict sentiment scores without any prior examples, relying solely on their general language understanding.
    - Few-Shot Learning: The models were provided with three annotated examples to help them understand the task format and context before making predictions.
- Models: We used the following pre-trained models from Ollama:
    - Llama 3.2 (3B & 8B parameters): Developed by Meta AI, known for its efficiency and strong performance.
    - Mistral (7B parameters): A high-performance model known for its optimized architecture.
- Results:
    - In the Zero-Shot setting, Llama 3.2 (8B) demonstrated the best performance, showcasing the power of increased parameter size for generalization.
    - In the Few-Shot setting, Mistral showed significant improvement and delivered performance close to Llama 3.2 (8B), proving to be a highly effective and economical alternative when training examples are available.
 
Files:
- Code: Final_Code.ipynb
