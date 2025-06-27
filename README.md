# ✈️ Predicting Airline Customer Satisfaction Using ML Models

A machine learning project analyzing airline passenger satisfaction using demographic, travel, and service-related features. Developed multiple classification models and performed EDA to identify key drivers of satisfaction for different travel classes.

## 👥 Authors
- Pavan Kumar Chintala

---

## 📊 Dataset Overview

- 📁 **Rows**: 129,880  
- 🧬 **Features**: 23 (Demographics, Service Ratings, Delays)  
- 🏷️ **Target**: `Satisfaction` (Satisfied / Neutral or Dissatisfied)  
- 📌 Source: [Kaggle - Airline Customer Satisfaction Dataset](https://www.kaggle.com/datasets)

---

## 🔍 Project Goals

- Identify key factors influencing airline passenger satisfaction.
- Predict customer satisfaction using machine learning.
- Compare model performance across different travel classes (Business vs. Economy).

---

## 🧪 Models Used

Separate models were trained for Business and Economy class passengers:

| Model               | Business Class Accuracy | Economy Class Accuracy |
|--------------------|-------------------------|------------------------|
| Logistic Regression | 87.1%                  | 87.9%                 |
| Decision Tree       | 96.1%                  | 92.4%                 |
| Random Forest       | **97.3%**              | **94.1%**             |

- ROC AUC (Business): 0.97  
- ROC AUC (Economy): 0.89

---

## 🔧 Tools & Techniques

- **EDA**: Univariate, Bivariate, and Correlation Analysis
- **Feature Engineering**: Outlier removal, Encoding categorical variables
- **Sklearn Models**: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`
- **Train/Test Split**: 80% / 20%
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📈 Key Insights

- **Top Satisfaction Drivers**:
  - Inflight Wi-Fi Service
  - Online Boarding Experience
  - Inflight Entertainment
- **Delays** negatively impacted satisfaction.
- Business class passengers prioritize convenience; Economy class focuses more on comfort and cleanliness.

---

## 📌 Conclusion

Random Forest achieved the best results with ~97% accuracy for Business class and ~94% for Economy class. Airlines are recommended to focus on **improving inflight Wi-Fi** and **streamlining online booking** to enhance overall satisfaction.

---

## 📚 References

- Kaggle Dataset
- Academic papers on customer satisfaction in airlines
- Google Scholar

---
