# 🐾 PetFinder.my - Adoption Speed Prediction

***Predicting how quickly pets find homes using Ensemble Gradient Boosting***
[Competition](https://www.kaggle.com/competitions/petfinder-adoption-prediction/overview)

## 📖 Overview
This project is a solution for the PetFinder.my Adoption Prediction challenge. The goal is to predict the speed at which a pet is adopted based on their online profile metadata, text descriptions, and photo properties.


## 📈 Target Distribution
The dataset is imbalanced, with a high concentration of pets in Class 2 (1st month) and Class 4 (100+ days).
<img width="712" height="479" alt="image" src="https://github.com/user-attachments/assets/3ccc933c-60a9-4f61-8501-735e616d7883" />

Figure 1: Distribution of adoption speeds in the training set.

The solution utilizes a **Stacked Ensemble** of three powerful gradient boosting models **(LightGBM, CatBoost, XGBoost)**, treating the problem as a regression task followed by an optimized thresholding step to classify the results into 5 ordinal categories (0-4).

## 📊 Key Results
The model achieves a **Quadratic Weighted Kappa (QWK) score of 0.4112**, placing it competitively within the problem space ***(Top 800)***.

### 1. Model Performance
The ensemble approach successfully reduces variance. As shown below, combining the three models yields a higher score than any single model alone.

<img width="709" height="458" alt="image" src="https://github.com/user-attachments/assets/8094f1b8-4afa-4cc1-8273-a46a414b9d40" />

Figure 2: The Ensemble (Red) outperforms individual models, reaching a QWK of 0.4112.


### 2. Confusion Matrix Analysis
The model excels at identifying pets that will take a long time to be adopted (Class 4), with a normalized accuracy of 47% for that class. It struggles most with the intermediate classes (1 vs 2), often confusing pets adopted in the first month vs. the second month, which is expected given the subtle differences in those categories.

<img width="690" height="553" alt="image" src="https://github.com/user-attachments/assets/5165ce63-5359-479d-a209-6ac68e1cbc16" />

Figure 3: Normalized Confusion Matrix showing class-wise performance.

## 🛠️ Methodology
### 1. Feature Engineering
The pipeline creates rich features from multiple data modalities:
- **Text Data:** TF-IDF vectorization followed by TruncatedSVD (LSA) to extract semantic meaning from pet descriptions into 12 dense features.
- **Metadata:** Extraction of Google Vision API data (Image Label Scores, Dominant Colors) and Google Natural Language API sentiment scores.
- **Rescuer Profiling (High Impact):** Utilizing RescuerID to capture shelter-specific adoption habits. This was handled using GroupKFold to prevent data leakage and Target Encoding (for LightGBM/XGBoost) and Ordered Target Encoding (CatBoost).
- **Aggregations:** Statistical features such as "Average Fee per Breed" and "Rescuer Volume" to provide context.

### 2. Modeling Strategy
Three distinct gradient boosting architectures were trained to ensure diversity:
- **LightGBM:** Optimized for speed and leaf-wise growth.
- **CatBoost:** Utilized for its superior handling of categorical features (Breed, Color, RescuerID).
- **XGBoost:** Added for stability and diversity in the ensemble.

All models were trained using **5-Fold GroupKFold Cross-Validation (grouped by RescuerID)** to ensure the model generalizes to new shelters/rescuers.

### 3. Threshold Optimization
Since the target is ordinal (0-4), treating it as a standard multi-class classification problem ignores the order (e.g., predicting 4 when the truth is 0 is worse than predicting 1).

Instead, I used Regression to predict a continuous score and then applied Nelder-Mead optimization via scipy.optimize to find the perfect cut-off points (thresholds) that maximize the QWK metric.

## 💻 How to Run
**Prerequisites**
- Python 3.x
- GPU (Recommended for faster training)

**Installation**

```python
pip install pandas numpy scikit-learn lightgbm catboost xgboost matplotlib seaborn scipy
```
