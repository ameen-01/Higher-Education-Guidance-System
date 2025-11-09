🎓 Intelligent Course Recommendation System using LSTM & Ensemble Learning
📘 Overview
In today’s data-driven education ecosystem, recommending the right academic program to a student is essential for improving success rates and institutional efficiency.
This project leverages Recurrent Neural Networks (LSTM) for sequence-based learning of student profiles and compares its performance with traditional Ensemble Models like Random Forest, AdaBoost, and Stacking.
By analyzing historical admission data (GATE scores, UG background, cutoffs, and other academic factors), the system predicts the most suitable program or specialization for each student.
🚀 Motivation
With thousands of students applying to various postgraduate programs, manual shortlisting becomes inefficient and inconsistent.
This system aims to automate course recommendations, helping students identify programs that match their academic strengths and eligibility patterns while assisting institutions in efficient candidate screening.
🎯 Objectives
To build a predictive recommendation system that aligns student profiles with the most appropriate programs.
To explore and compare deep learning (LSTM) with ensemble-based models for accuracy and interpretability.
To enhance decision transparency through explainable AI and visual analytics.
To improve admission prediction and recommendation efficiency for academic institutions.
🧹 Data Preprocessing
Handling Missing Values: Null entries in key features like GATE_score and UG_percentage were imputed using mean or median values.
Encoding Categorical Variables: Label Encoding and One-Hot Encoding applied to fields like UG_branch, Category, and Location_state.
Normalization: Numerical features (scores, percentages, ranks) scaled to [0,1] range for faster convergence.
Feature Selection: Removed redundant columns and retained features most correlated with admission outcomes.
🔍 Feature Extraction
Key extracted features:
Academic: UG_percentage, GATE_score, GATE_rank, Year_of_passing
Demographic: Category, Location_state
Institutional: University_tier, Dept_specialization, Previous_cutoff_diff
These features capture both student competency and program competitiveness, forming the base for model learning.
🧠 Model Development
🔹 LSTM (Recurrent Neural Network)
Designed to process sequential admission patterns per student.
Captures temporal dependencies — e.g., how UG performance and GATE score trends influence future program choices.
Justified for this use case since the student–program sequence behaves like a time-ordered learning journey, unlike static ensemble models.
🔹 Ensemble Models
AdaBoost: Sequentially focuses on misclassified samples, improving weak learners.
Random Forest: Aggregates multiple decision trees to reduce variance and prevent overfitting.
Stacking (ID3 + Naïve Bayes + SVM): Combines probabilistic and margin-based classifiers for hybrid learning.
📊 Methodology
Step 1: Data Cleaning & Preprocessing
Step 2: Exploratory Data Analysis & Feature Correlation
Step 3: Model Building (LSTM + Ensemble)
Step 4: Model Evaluation (Accuracy, Confusion Matrix, ROC Curve)
Step 5: Explainability & Visualization (Feature Importance, SHAP values)
Step 6: Streamlit Interface for student input and course recommendations
📈 Results
Random Forest Classifier achieved the highest accuracy (~94%), outperforming AdaBoost (92.6%) and Stacking (87.5%).
LSTM, while slightly less accurate, showed stronger generalization in sequential prediction tasks.
Key predictive features include GATE score, UG percentage, and difference_from_cutoff, indicating their strong impact on admission success.
⚖️ Why LSTM Over Ensemble?
Ensemble models treat inputs as static, ignoring order or dependencies among student attributes.
LSTM captures temporal and sequential relationships, vital when past academic trends influence future admission outcomes.
It adapts to variable-length inputs (different numbers of past records per student), making it more flexible for real-world academic data.
🎨 Visualizations
Comparative Bar Charts for Supervised, Unsupervised, and Ensemble models using Plotly.
Feature Importance and Correlation Heatmaps.
Streamlit dashboard for live input and model explainability.
🧩 Technologies Used
Python, PyTorch, Scikit-learn, Pandas, NumPy
Plotly, Matplotlib, Seaborn
Streamlit – Interactive recommendation interface
💡 Future Scope
Integrate Explainable AI (XAI) for transparent decision reasoning.
Incorporate real-time student feedback for model refinement.
Extend to cross-institutional recommendations using collaborative filtering and federated data.
