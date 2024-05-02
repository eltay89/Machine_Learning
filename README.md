# Machine_Learning
This comprehensive cheat sheet covers the basics, algorithms, evaluation metrics, challenges, best practices, tools, resources, and glossary terms related to Machine Learning.

**Machine Learning Basics**
======================

### What is Machine Learning?
-----------------------------

* A subset of Artificial Intelligence
* enables computers to learn from data without being explicitly programmed

 Types of Machine Learning
-------------------------

* **Supervised Learning**: algorithm learns from labeled examples, predicts new instances based on patterns learned.
	+ Example: Image classification (e.g., cat vs. dog)
* **Unsupervised Learning**: algorithm discovers hidden patterns or structure in unlabeled data.
	+ Example: Clustering similar customers based on purchase history
* **Reinforcement Learning**: algorithm learns by interacting with environment, receiving rewards or penalties for actions taken.

 Machine Learning Algorithms
---------------------------

### Supervised

* Linear Regression (LR)
* Logistic Regression (LogReg)
* Decision Trees (DT)
* Random Forests (RF)
* Support Vector Machines (SVM)

### Unsupervised

* K-Means Clustering (KMC)
* Hierarchical Clustering (HC)
* Principal Component Analysis (PCA)
* t-Distributed Stochastic Neighbor Embedding (t-SNE)

### Reinforcement Learning

* Q-Learning
* SARSA
* Deep Q-Networks (DQN)

 Machine Learning Evaluation Metrics
-----------------------------------

### Regression

* Mean Squared Error (MSE)
* Root Mean Squared Percentage Error (RMSPE)
* Coefficient of Determination (R-squared)

### Classification

* Accuracy
* Precision
* Recall
* F1 Score
* Area Under the ROC Curve (AUC-ROC)

 Machine Learning Challenges and Best Practices
-------------------------------------------------

### Common Issues

* **Overfitting**: model performs well on training data but poorly on new, unseen data.
	+ Solution: Regularization techniques (e.g., dropout), early stopping, or ensemble methods.

### Responsible AI Practices
-------------------------------

### Fairness and Bias

* **Data bias**: algorithms learn from biased datasets, perpetuating unfair outcomes.
	+ Solution: Collect diverse, representative training data; use fairness metrics (e.g., demographic parity).

### Transparency and Explainability

* **Model interpretability**: understand how models make predictions to ensure trustworthiness.
	+ Solution: Use techniques like feature importance, partial dependence plots, or SHAP values.

 Machine Learning Tools and Resources
--------------------------------------

### Frameworks

* TensorFlow
* PyTorch
* Scikit-Learn
* Keras

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn

### Platforms

* Kaggle (https://kaggle.com)
* Google Colab (https://colab.research.google.com/)
* AWS SageMaker (https://aws.amazon.com/sagemaker/)

 Machine Learning Glossary
-------------------------

### Terms and Concepts

* **Bias-Variance Tradeoff**: balance between model's ability to fit training data and its generalizability.
* **Gradient Descent**: optimization algorithm used in many machine learning models.
* **Hyperparameter Tuning**: process of adjusting model parameters for optimal performance.
