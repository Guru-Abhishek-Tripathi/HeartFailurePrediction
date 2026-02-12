# Heart Disease Prediction - ML Classification Models

## Problem Statement

Heart disease is one of the leading causes of mortality worldwide. Early and accurate prediction of heart disease can significantly improve patient outcomes by enabling timely medical intervention. This project addresses the binary classification problem of predicting whether a patient has heart disease (1) or not (0) based on clinical and diagnostic features.

The goal is to implement, evaluate, and compare six machine learning classification models on the Heart Failure Prediction dataset and deploy an interactive Streamlit web application to demonstrate the models.

---

## Dataset Description

**Dataset:** [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)  
**Source:** Kaggle (fedesoriano)  
**File:** `heart.csv`

| Property | Value |
|---|---|
| Total Instances | 918 |
| Number of Features | 11 (+ 1 target) |
| Target Column | `HeartDisease` (0 = No Disease, 1 = Disease) |
| Class Distribution (Train) | Class 0: 328 samples, Class 1: 406 samples |
| Train / Test Split | 734 / 184 (80/20, stratified) |
| Preprocessing | StandardScaler for feature scaling |

### Features

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Age of the patient (years) |
| Sex | Categorical | Sex of the patient (M/F) |
| ChestPainType | Categorical | Chest pain type (ATA, NAP, ASY, TA) |
| RestingBP | Numeric | Resting blood pressure (mm Hg) |
| Cholesterol | Numeric | Serum cholesterol (mm/dl) |
| FastingBS | Binary | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| RestingECG | Categorical | Resting electrocardiogram results (Normal, ST, LVH) |
| MaxHR | Numeric | Maximum heart rate achieved |
| ExerciseAngina | Categorical | Exercise-induced angina (Y/N) |
| Oldpeak | Numeric | ST depression induced by exercise |
| ST_Slope | Categorical | Slope of peak exercise ST segment (Up, Flat, Down) |
| **HeartDisease** | **Binary (Target)** | **Output class (0 = No Disease, 1 = Disease)** |

---

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 86.96% | 0.8971 | 84.82% | 93.14% | 88.79% | 0.7374 |
| Decision Tree | 79.35% | 0.7958 | 81.37% | 81.37% | 81.37% | 0.5820 |
| kNN | 89.13% | 0.9192 | 89.42% | 91.18% | 90.29% | 0.7797 |
| Naive Bayes | 89.13% | 0.9280 | 89.42% | 91.18% | 90.29% | 0.7797 |
| Random Forest (Ensemble) | 88.04% | 0.9273 | 87.04% | 92.16% | 89.52% | 0.7579 |
| XGBoost (Ensemble) | 86.96% | 0.9209 | 89.00% | 87.25% | 88.12% | 0.7368 |

> **Setup:** All models trained on 734 samples (80% split), evaluated on 184 test samples. Features scaled using `StandardScaler`. Hyperparameters: Logistic Regression (C=1.0, L2, max_iter=1000), Decision Tree (max_depth=10), KNN (k=5), Random Forest (n_estimators=300, max_depth=20), XGBoost (n_estimators=300, max_depth=7, lr=0.1).

---

### Model Observations

| ML Model Name | Observation about Model Performance |
|---|---|
| Logistic Regression | Achieved a solid accuracy of **86.96%** with a strong recall of **93.14%**, meaning it correctly identifies the majority of true heart disease cases — a critical property in medical diagnosis. The AUC of **0.8971** confirms good discriminative ability. However, the relatively lower precision (84.82%) compared to other models suggests it produces more false positives. The MCC of **0.7374** indicates a reliable overall performance despite the asymmetry between precision and recall. This model serves as a strong, interpretable baseline. |
| Decision Tree | The Decision Tree was the weakest performer overall, with the lowest accuracy (**79.35%**), AUC (**0.7958**), and MCC (**0.5820**). Its precision and recall are both at **81.37%**, indicating a balanced but mediocre performance. The low MCC suggests the model struggles to effectively separate the two classes. Decision Trees are prone to overfitting, and even with `max_depth=10`, it could not match the performance of other models on this dataset. It may benefit from pruning or hyperparameter tuning. |
| kNN | Tied with Naive Bayes for the highest accuracy (**89.13%**), precision (**89.42%**), recall (**91.18%**), F1 (**90.29%**), and MCC (**0.7797**). The AUC of **0.9192** reflects strong probabilistic ranking ability. KNN performs well on this relatively small, well-scaled dataset, as StandardScaler ensures distance-based calculations are meaningful. With k=5, it captures local patterns effectively. However, its performance may degrade on larger datasets due to the computational cost of distance calculations at prediction time. |
| Naive Bayes | Achieved identical accuracy, precision, recall, F1, and MCC to KNN (**89.13%, 89.42%, 91.18%, 90.29%, 0.7797**), but posted the **highest AUC of 0.9280** among all models. This is notable because the Gaussian Naive Bayes assumes feature independence, which clearly holds well enough on this dataset after scaling. Its high AUC indicates excellent class separation in terms of predicted probabilities. It is the fastest model to train and is highly suitable for this problem given the relatively continuous numerical features. |
| Random Forest (Ensemble) | Delivered well-rounded performance with **88.04% accuracy**, the **highest recall of 92.16%**, and a strong AUC of **0.9273**. The ensemble approach smooths out the variance seen in individual decision trees, resulting in significantly better generalization than the standalone Decision Tree. The MCC of **0.7579** is solid. The slight gap in accuracy compared to KNN/Naive Bayes may be attributed to the hyperparameter configuration (min_samples_split=10, min_samples_leaf=4), which prevents overfitting but might slightly underfit on this smaller dataset. |
| XGBoost (Ensemble) | Achieved **86.96% accuracy** — equal to Logistic Regression — but with the **highest precision of 89.00%** and a notably high AUC of **0.9209**. It has the lowest recall (**87.25%**) among all models, meaning it is more conservative in predicting positive cases, which can result in more missed diagnoses. In a medical context, this trade-off is important to consider. The MCC of **0.7368** is competitive. XGBoost's strength lies in its robustness and potential for further improvement with hyperparameter tuning (e.g., via GridSearchCV). |

---

## Repository Structure

```
project-folder/
│── app.py                  # Streamlit web application
│── requirements.txt        # Python dependencies
│── README.md               # Project documentation
│── MFML_Assignment2.ipynb  # Jupyter notebook with full ML pipeline
│── model/
│   ├── Logistic_Regression.pkl
│   ├── Decision_Tree.pkl
│   ├── KNN.pkl
│   ├── Gaussian_Naive_Bayes.pkl
│   ├── Random_Forest.pkl
│   ├── XGBoost.pkl
│   └── scaler.pkl
```

---

## Streamlit App Features

- **Dataset Upload** – Upload a CSV file of test data for batch predictions
- **Model Selection** – Dropdown to choose from all 6 trained classification models
- **Evaluation Metrics** – Displays Accuracy, AUC, Precision, Recall, F1, and MCC
- **Confusion Matrix** – Visual confusion matrix for the selected model's predictions
- **Classification Report** – Detailed per-class precision, recall, and F1 scores

---

## Requirements

```
streamlit
scikit-learn
xgboost
imbalanced-learn
numpy
pandas
matplotlib
seaborn
joblib
kagglehub
```

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

---

## Deployment

The app is deployed on **Streamlit Community Cloud**.  
Live App: 

---

*Assignment 2 | Machine Learning | M.Tech (AIML/DSE) | BITS Pilani – WILP Division*