## Breast Cancer Prediction
This project aims to predict whether a breast tumor is malignant or benign using a machine learning model. The dataset used is the famous Breast Cancer Wisconsin (Diagnostic) dataset. The project implements multiple machine learning models and allows for predictions based on user-provided inputs.

### Overview
This project uses the Breast Cancer Wisconsin (Diagnostic) dataset, which consists of features like radius, perimeter, area, and texture measurements from images of breast cancer biopsies. The goal is to build a model that can predict whether a tumor is malignant or benign based on these features. The project uses several machine learning algorithms to classify the data: Random Forest Classifier, Decision Tree Classifier, Support Vector Classifier (SVC), K-Nearest Neighbors (KNN), KMeans (Clustering) The performance of each model is evaluated using accuracy scores, precision, recall, and f1-score.

### Technologies and Libraries Used
- Python 3.10+
- pandas
- matplotlib

- seaborn

- scikit-learn

- plotly

- numpy

### Setup and Usage
- Clone the repository:

      git clone https://github.com/yourusername/breast-cancer-prediction.git
      cd breast-cancer-prediction
- Install required dependencies:

      pip install -r requirements.txt
- You can create the requirements.txt file using:

      pip freeze > requirements.txt
- Ensure that you have the necessary dataset. You can download it from UCI Machine Learning Repository - Breast Cancer Wisconsin Dataset.

- Run the script to train and evaluate the models:

      python breast_cancer_predictor.py
- When prompted, enter the feature values (such as radius_mean, perimeter_mean, etc.) for the tumor you wish to classify.

### Example:

- Radius_mean[6.981,28.11] : 50
- Perimeter_mean[43.79,188.5] : 200
- Area_mean[143.5,2501.0] : 1000 \
- Symmetry_mean[0.106,0.304] : 0.5
- Compactness_mean[0.01938,0.3454] : 0.6
- Concave_points_mean[0.0,0.2012] : 0.5

The model will output whether the tumor is Malignant or Benign based on the input features.

### Models Implemented
 - RandomForestClassifier: Best model based on accuracy and general performance. Achieved an accuracy of 92.02%.

 - DecisionTreeClassifier: A simpler tree-based model with a slightly lower accuracy of 90.43%.

- SVC (Support Vector Classifier): An SVM-based model with accuracy around 88.30%.

- KNN (K-Nearest Neighbors): A distance-based model with an accuracy of 87.23%.

- KMeans: Clustering model that is not ideally suited for classification tasks. It gave an accuracy of 61.17%.

### Contributions
Feel free to fork this repository, make improvements, or add new features to enhance the translation capabilities. Contributions are welcome!

### Hope this helps! Happy learning!!
