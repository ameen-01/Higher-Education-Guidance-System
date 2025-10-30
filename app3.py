import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")


# -------------------------------
# Load dataset (for encoding, not training)
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("")
    return df

df = load_data()

# -------------------------------
# Label Encoders
# -------------------------------
categorical_cols = ['UG_institute_tier', 'UG_branch', 'Category', 'GATE_paper',
                    'university_tier', 'location_state', 'dept_specialization', 'university_name']

encoders = {}
for col in categorical_cols:
    enc = LabelEncoder()
    df[col] = enc.fit_transform(df[col])
    encoders[col] = enc

scaler = StandardScaler()
num_cols = ['UG_percentage', 'GATE_score', 'GATE_rank', 'Work_exp_years', 'difference_from_cutoff']
df[num_cols] = scaler.fit_transform(df[num_cols])

# -------------------------------
# Define Model
# -------------------------------


class RNNRecommender(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNRecommender, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

input_size = 13  # 8 categorical + 5 numeric
hidden_size = 64
output_size = len(df['program_id'].unique())

model = RNNRecommender(input_size, hidden_size, output_size)
state_dict = torch.load("best_rnn_model.pt", map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()

# -------------------------------
# Streamlit UI
# -------------------------------

st.title("🎓 AI-Powered PG Program Recommender")
st.markdown("#### Enter your details to get the best-matched university programs based on your profile.")

col1, col2 = st.columns(2)

with col1:
    UG_institute_tier = st.selectbox("Undergraduate Institute Tier", encoders['UG_institute_tier'].classes_)
    UG_branch = st.selectbox("Undergraduate Branch", encoders['UG_branch'].classes_)
    Category = st.selectbox("Category", encoders['Category'].classes_)
    GATE_paper = st.selectbox("GATE Paper", encoders['GATE_paper'].classes_)
    university_tier = st.selectbox("University Tier", encoders['university_tier'].classes_)
    dept_specialization = st.selectbox("Department Specialization", encoders['dept_specialization'].classes_)
    university_name = st.selectbox("University Name", encoders['university_name'].classes_)
    location_state = st.selectbox("Preferred State", encoders['location_state'].classes_)

with col2:
    UG_percentage = st.number_input("UG Percentage", 50.0, 100.0, 75.0)
    GATE_score = st.number_input("GATE Score", 15.0, 40.0, 25.0)
    GATE_rank = st.number_input("GATE Rank", 1, 10000, 500)
    Work_exp_years = st.number_input("Work Experience (Years)", 0.0, 5.0, 0.0)
    difference_from_cutoff = st.number_input("Difference from Cutoff", -10.0, 10.0, 0.0)

# -------------------------------
# Prediction Logic
# -------------------------------
if st.button("🔍 Recommend Programs"):
    try:
        # Encode categorical features
        encoded_input = [
            encoders['UG_institute_tier'].transform([UG_institute_tier])[0],
            encoders['UG_branch'].transform([UG_branch])[0],
            encoders['Category'].transform([Category])[0],
            encoders['GATE_paper'].transform([GATE_paper])[0],
            encoders['university_tier'].transform([university_tier])[0],
            encoders['location_state'].transform([location_state])[0],
            encoders['dept_specialization'].transform([dept_specialization])[0],
            encoders['university_name'].transform([university_name])[0]
        ]

        # Scale numeric features (5 total)
        numeric_scaled = scaler.transform([[UG_percentage, GATE_score, GATE_rank, Work_exp_years, difference_from_cutoff]])[0]

        # Combine 8 encoded + 5 numeric
        input_vector = np.concatenate((encoded_input, numeric_scaled)).reshape(1, 1, -1)

        # Convert to tensor
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)

        # Model prediction
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).numpy().flatten()

        # Top 5 recommendations
        top_indices = np.argsort(probs)[::-1][:5]
        top_programs = df['program_id'].unique()[top_indices]
        top_scores = probs[top_indices]

        st.subheader("🎯 Recommended Programs")
        results = []
        for pid, score in zip(top_programs, top_scores):
            program_data = df[df['program_id'] == pid].iloc[0]
            results.append({
                "Program ID": pid,
                "University": encoders['university_name'].inverse_transform([program_data['university_name']])[0],
                "Department": encoders['dept_specialization'].inverse_transform([program_data['dept_specialization']])[0],
                "University Tier": encoders['university_tier'].inverse_transform([program_data['university_tier']])[0],
                "Location": encoders['location_state'].inverse_transform([program_data['location_state']])[0]
                
            })

        st.dataframe(pd.DataFrame(results))

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
