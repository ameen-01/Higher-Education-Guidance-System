# 🎓 Intelligent Course Recommendation System using LSTM & Ensemble Learning

## 📘 Overview
This project presents a **Course Recommendation System** that predicts the most suitable postgraduate program for students based on their academic and demographic profiles.  
It leverages both **Deep Learning (LSTM)** and traditional **Ensemble Models (Random Forest, AdaBoost, and Stacking)** to improve recommendation accuracy and reliability.  
The system analyzes features such as GATE scores, UG background, university tiers, and category to recommend programs aligned with a student’s strengths and eligibility.

---

## 🚀 Motivation
With thousands of students applying across multiple universities, manual shortlisting becomes inconsistent and inefficient.  
This project aims to **automate course recommendations** using AI, ensuring fairness, transparency, and higher accuracy in admission predictions.

---

## 🎯 Objectives
- Develop a predictive model that recommends the most suitable course for each student.  
- Compare the performance of **LSTM (RNN)** with **Ensemble Learning techniques**.  
- Incorporate explainability and visualization to interpret prediction factors.  
- Improve institutional admission efficiency using data-driven recommendations.

---

## 🧹 Data Preprocessing
- **Missing Values:** Imputed missing numerical values (`GATE_score`, `UG_percentage`) using mean or median strategies.  
- **Encoding:** Applied Label and One-Hot Encoding to categorical features such as `UG_branch`, `Category`, and `Location_state`.  
- **Normalization:** Scaled continuous variables between 0 and 1 for uniform model convergence.  
- **Feature Selection:** Retained only correlated features affecting admission results (e.g., GATE score, cutoff difference, university tier).

---

## 🔍 Feature Extraction
Key extracted and engineered features include:  
- **Academic:** `UG_percentage`, `GATE_score`, `GATE_rank`, `Year_of_passing`  
- **Demographic:** `Category`, `Location_state`  
- **Institutional:** `University_tier`, `Dept_specialization`, `difference_from_cutoff`  

These features capture a student’s performance, category-based eligibility, and competitiveness of applied programs.

---

## 🧠 Model Development

### 🔹 LSTM (Recurrent Neural Network)
- Processes **sequential admission data** for each student.  
- Captures **temporal dependencies** like how UG performance and GATE trends influence admission success.  
- Best suited for datasets where **student progress forms a sequence** over time.  

### 🔹 Ensemble Models
- **AdaBoost:** Sequentially corrects errors of weak learners for better classification.  
- **Random Forest:** Uses multiple decision trees to reduce variance and overfitting.  
- **Stacking (ID3 + Naïve Bayes + SVM):** Combines heterogeneous models for improved generalization.

---

## 🧩 Methodology
1. Data Collection and Cleaning  
2. Preprocessing and Feature Encoding  
3. Model Development (LSTM and Ensemble Models)  
4. Model Training and Hyperparameter Tuning  
5. Evaluation using Accuracy, Confusion Matrix, and Visualization  
6. Streamlit Deployment for real-time student recommendation

---

## 📊 Results
- **Random Forest Classifier** achieved the highest accuracy (~94%), outperforming **AdaBoost (92.6%)** and **Stacking (87.5%)**.  
- **LSTM** showed better adaptability for sequential and temporal student data.  
- Key influencing factors included **GATE Score**, **UG Percentage**, and **Difference from Cutoff**.

---

## ⚖️ Why LSTM Over Ensemble
- Ensemble models treat data as **independent instances**, ignoring order or dependencies.  
- LSTM captures **temporal relationships** across academic features, such as trends in performance over years.  
- Ideal for modeling **student progression**, where previous academic behavior impacts future program suitability.

---

## 🎨 Visualizations
- Comparative model performance charts using **Plotly**.  
- Feature importance visualization for ensemble models.  
- Streamlit dashboard for input-based course recommendations with explainability.

---

## 🧰 Technologies Used
| Category | Tools / Libraries |
|-----------|------------------|
| Programming | Python 3.x |
| ML Frameworks | PyTorch, Scikit-learn |
| Visualization | Plotly, Matplotlib, Seaborn |
| Data Handling | Pandas, NumPy |
| Web App | Streamlit |

---

## 🧭 Future Scope
- Integrate **Explainable AI (XAI)** for transparent reasoning behind recommendations.  
- Include **student feedback loops** for dynamic model improvement.  
- Expand to **cross-institutional recommendation systems** using collaborative learning.  
- Deploy as a **cloud-based API** for integration with university admission portals.

---


