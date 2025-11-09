import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import io

# Set page config
st.set_page_config(page_title="PG Admission Recommender", layout="wide", page_icon="🎓")

# Neural Network Model
class AdmissionPredictor(nn.Module):
    def __init__(self, input_size):
        super(AdmissionPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.network(x)

@st.cache_data
def load_data():
    # Read the CSV file
    df = pd.read_csv('/')
    return df

@st.cache_resource
def train_model(df):
    # Prepare features
    categorical_cols = ['UG_institute_tier', 'UG_branch', 'GATE_paper', 'Category', 
                       'university_name', 'dept_specialization', 'university_tier', 'location_state']
    
    numerical_cols = ['UG_percentage', 'Year_of_passing', 'GATE_score', 'GATE_rank', 
                     'Work_exp_years', 'previous_cutoff_gen', 'previous_cutoff_obc',
                     'previous_cutoff_sc', 'previous_cutoff_st', 'min_UG_required', 
                     'difference_from_cutoff']
    
    # Encode categorical variables
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Prepare features and target
    feature_cols = categorical_cols + numerical_cols
    X = df_encoded[feature_cols].values
    y = df_encoded['admitted_flag'].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)
    
    # Initialize model
    model = AdmissionPredictor(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training
    epochs = 100
    batch_size = 64
    
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train_tensor), batch_size):
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = (test_outputs > 0.5).float()
        accuracy = (test_predictions == y_test_tensor).float().mean()
    
    return model, scaler, label_encoders, feature_cols, accuracy.item()

def predict_admission(model, scaler, label_encoders, feature_cols, student_data):
    # Encode categorical features
    encoded_data = {}
    for col in student_data:
        if col in label_encoders:
            try:
                encoded_data[col] = label_encoders[col].transform([str(student_data[col])])[0]
            except:
                encoded_data[col] = 0  # Default for unknown categories
        else:
            encoded_data[col] = student_data[col]
    
    # Create feature vector
    feature_vector = np.array([encoded_data[col] for col in feature_cols]).reshape(1, -1)
    
    # Scale
    feature_scaled = scaler.transform(feature_vector)
    
    # Predict
    model.eval()
    with torch.no_grad():
        tensor_input = torch.FloatTensor(feature_scaled)
        probability = model(tensor_input).item()
    
    return probability

def get_recommendations(df, student_profile, model, scaler, label_encoders, feature_cols, top_n=10):
    # Filter programs based on basic eligibility
    eligible_programs = df[
        (df['min_UG_required'] <= student_profile['UG_percentage']) &
        (df['GATE_paper'] == student_profile['GATE_paper'])
    ].copy()
    
    if len(eligible_programs) == 0:
        return pd.DataFrame()
    
    # Predict admission probability for each program
    probabilities = []
    for _, program in eligible_programs.iterrows():
        test_data = {
            'UG_institute_tier': student_profile['UG_institute_tier'],
            'UG_branch': student_profile['UG_branch'],
            'UG_percentage': student_profile['UG_percentage'],
            'Year_of_passing': student_profile['Year_of_passing'],
            'GATE_paper': student_profile['GATE_paper'],
            'GATE_score': student_profile['GATE_score'],
            'GATE_rank': student_profile['GATE_rank'],
            'Category': student_profile['Category'],
            'Work_exp_years': student_profile['Work_exp_years'],
            'university_name': program['university_name'],
            'dept_specialization': program['dept_specialization'],
            'previous_cutoff_gen': program['previous_cutoff_gen'],
            'previous_cutoff_obc': program['previous_cutoff_obc'],
            'previous_cutoff_sc': program['previous_cutoff_sc'],
            'previous_cutoff_st': program['previous_cutoff_st'],
            'min_UG_required': program['min_UG_required'],
            'university_tier': program['university_tier'],
            'location_state': program['location_state'],
            'difference_from_cutoff': program['difference_from_cutoff']
        }
        
        prob = predict_admission(model, scaler, label_encoders, feature_cols, test_data)
        probabilities.append(prob)
    
    eligible_programs['admission_probability'] = probabilities
    
    # Sort by probability and get top N
    recommendations = eligible_programs.nlargest(top_n, 'admission_probability')
    
    return recommendations[['university_name', 'dept_specialization', 'university_tier', 
                           'location_state', 'previous_cutoff_gen', 'previous_cutoff_obc',
                           'previous_cutoff_sc', 'previous_cutoff_st', 'admission_probability']]

# Main App
def main():
    st.title("🎓 PG Admission Course Recommendation System")
    
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Train model
    with st.spinner("Training neural network model..."):
        model, scaler, label_encoders, feature_cols, accuracy = train_model(df)
    
    st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")
    
    # Sidebar for student input
    st.sidebar.header("📋 Student Profile")
    
    # Input fields
    ug_tier = st.sidebar.selectbox("UG Institute Tier", sorted(df['UG_institute_tier'].unique()))
    ug_branch = st.sidebar.selectbox("UG Branch", sorted(df['UG_branch'].unique()))
    ug_percentage = st.sidebar.number_input("UG Percentage", min_value=50.0, max_value=100.0, value=75.0, step=0.1)
    year_passing = st.sidebar.selectbox("Year of Passing", sorted(df['Year_of_passing'].unique(), reverse=True))
    gate_paper = st.sidebar.selectbox("GATE Paper", sorted(df['GATE_paper'].unique()))
    gate_score = st.sidebar.number_input("GATE Score", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    gate_rank = st.sidebar.number_input("GATE Rank", min_value=1, max_value=2000, value=500, step=1)
    category = st.sidebar.selectbox("Category", sorted(df['Category'].unique()))
    work_exp = st.sidebar.number_input("Work Experience (years)", min_value=0, max_value=10, value=0, step=1)
    
    # Create student profile
    student_profile = {
        'UG_institute_tier': ug_tier,
        'UG_branch': ug_branch,
        'UG_percentage': ug_percentage,
        'Year_of_passing': year_passing,
        'GATE_paper': gate_paper,
        'GATE_score': gate_score,
        'GATE_rank': gate_rank,
        'Category': category,
        'Work_exp_years': work_exp
    }
    
    if st.sidebar.button("🔍 Get Recommendations", type="primary"):
        with st.spinner("Analyzing your profile and generating recommendations..."):
            recommendations = get_recommendations(
                df, student_profile, model, scaler, label_encoders, feature_cols, top_n=10
            )
        
        if len(recommendations) == 0:
            st.warning("⚠️ No eligible programs found.")
        else:
            st.header("🎯 Top Recommended Programs")
            
            # Display recommendations
            for idx, row in recommendations.iterrows():
                col1, col2= st.columns([2, 2])
                
                with col1:
                    st.markdown(f"**{row['university_name']}**")
                    st.caption(f"Specialization: {row['dept_specialization']}")
                
                with col2:
                    st.metric("University Tier", row['university_tier'])
                    st.caption(f"Location: {row['location_state']}")
                
                # with col3:
                #     prob_percent = row['admission_probability'] * 100
                #     if prob_percent >= 70:
                #         st.success(f"**{prob_percent:.1f}%**")
                #         st.caption("High Chance")
                #     elif prob_percent >= 50:
                #         st.warning(f"**{prob_percent:.1f}%**")
                #         st.caption("Moderate")
                #     else:
                #         st.info(f"**{prob_percent:.1f}%**")
                #         st.caption("Low Chance")
                
                # # Show cutoffs
                # cutoff_col1, cutoff_col2, cutoff_col3, cutoff_col4 = st.columns(4)
                # cutoff_col1.caption(f"Gen: {row['previous_cutoff_gen']}")
                # cutoff_col2.caption(f"OBC: {row['previous_cutoff_obc']}")
                # cutoff_col3.caption(f"SC: {row['previous_cutoff_sc']}")
                # cutoff_col4.caption(f"ST: {row['previous_cutoff_st']}")
                
                st.divider()
    
    # Statistics section
    with st.expander("📊 Dataset Statistics"):
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", len(df))
        col2.metric("Unique Students", df['student_id'].nunique())
        col3.metric("Unique Programs", df['program_id'].nunique())
        col4.metric("Admission Rate", f"{df['admitted_flag'].mean()*100:.1f}%")
        
        st.subheader("University Distribution")
        university_counts = df['university_tier'].value_counts()
        st.bar_chart(university_counts)

if __name__ == "__main__":
    main()
