# Spam Classification

The code is written in Spam_Classification.py file. 

1. Read, clean, and organize this dataset into easy-to-read format for Machine Learning 
(ML) models.
- The Spambase dataset is loaded, with features separated for model training 
and labels indicating spam or non-spam. Standard scaling is applied to make 
features more comparable for machine learning.

2. Experiment with the following ML models:  
• Logistic Regression 
• Decision Tree 
• Gradient Boosting 
- Three models (Logistic Regression, Decision Tree, and Gradient Boosting) are 
tested to classify emails. Each model is trained and evaluated for accuracy 
using the scaled data.

3. Use Principal Component Analysis (PCA) to compress the data (i.e. dimensionality 
reduction) into k = 10dimensions and report your classification results with Logistic 
Regression, Decision Tree and Gradient Boosting on the PCA features (i.e. compressed 
data).  
- Principal Component Analysis (PCA) reduces the feature space to 10 
dimensions, capturing most of the data's variance. The models are then re
evaluated on this reduced dataset to compare performance.

4. Visualize the embeddings with PCA and t-SNE.  
- The dataset is visualized in 2D using PCA and t-SNE, showing class 
separations for spam and non-spam emails and helping us understand the 
clustering and separability in reduced dimensions.

![image](https://github.com/user-attachments/assets/4a4cfc9f-8df0-4c0c-8704-10d9e4a107a2)

![image](https://github.com/user-attachments/assets/553a939c-5e5b-4fba-b9ea-7d866fdc293c)

![image](https://github.com/user-attachments/assets/6010be48-3c00-4d03-905f-fd974d88cba2)

- The initial model accuracies without PCA are 93.04% for Logistic Regression, 99.93% for 
Decision Tree, and 96.28% for Gradient Boosting. When the data is compressed to 10 
dimensions using PCA where compression factor of about 5.7, the accuracy of Logistic 
Regression drops slightly to 88.96%, and Gradient Boosting decreases to 93.20%, while the 
Decision Tree retains its accuracy at 99.93%. This indicates that the Decision Tree model is 
robust to dimensionality reduction, maintaining its high performance even with fewer 
features.

