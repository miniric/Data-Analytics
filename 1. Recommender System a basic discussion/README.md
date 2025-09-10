# Comparative Analysis of Recommender Systems
*A Focus on Matrix Completion and Deep Learning Approaches*

## 📖 Project Overview
The project compares **traditional recommender systems** (Content-Based Filtering, User-Based Collaborative Filtering) with **modern approaches** (Matrix Completion, Neural Network models) under **limited data conditions** — using only a user–movie rating matrix as input.

The study focuses on addressing:
- **Data Sparsity**: Very few ratings per user.
- **Cold Start Problem**: New users or items without historical ratings.

---

## 🎯 Objectives
- Review and implement multiple recommender system algorithms.
- Experiment with hyperparameters and architectures for different models.
- Compare models using **accuracy, precision, recall, F1-score, and RMSE**.
- Suggest suitable RS models for **small businesses or startups** with limited data.

---

## 🛠 Models Implemented
### **Conventional Methods**
- Content-Based Filtering
- User-Based Collaborative Filtering (Cosine Similarity, Pearson Correlation)

### **Matrix Completion**
- Singular Value Thresholding (SVT)
- Alternating Least Squares (ALS)

### **Neural Network Models**
- Multi-Layer Perceptron (MLP)
- Autoencoder (User-Based and Item-Based)
- Matrix Factorization Networks with Bias

---

## 📊 Dataset
- **MovieLens 20M** dataset by GroupLens Research.

---

## 📈 Evaluation Metrics
- **Classification Metrics**: Accuracy, Precision, Recall, F1-score 
- **Regression Metric**: Root Mean Squared Error (RMSE)
- Two test scenarios:
  1. **Normal User**: 30% ratings removed.
  2. **Cold Start User**: 100% ratings removed.

---
