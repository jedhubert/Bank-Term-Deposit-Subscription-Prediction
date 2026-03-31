import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Page configuration
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .positive {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .negative {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Feature Engineering Class
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Duration features
        if 'duration' in X.columns:
            X['duration_log'] = np.log1p(X['duration'])
        
        # Balance features
        if 'balance' in X.columns:
            X['balance_positive'] = (X['balance'] > 0).astype(int)
        
        # Age groups
        if 'age' in X.columns:
            X['age_group'] = pd.cut(X['age'], bins=[0, 30, 40, 50, 60, 100], 
                                    labels=['young', 'middle', 'senior', 'old', 'elderly'])
        
        # Campaign intensity
        if 'campaign' in X.columns:
            X['high_campaign'] = (X['campaign'] > 3).astype(int)
        
        # Previous contact
        if 'pdays' in X.columns:
            X['was_contacted_before'] = (X['pdays'] != -1).astype(int)
        
        return X

@st.cache_resource
def train_model(train_path):
    """Train the model from scratch using the training data"""
    
    with st.spinner("🔄 Training model... This may take a few minutes..."):
        # Load data
        train_df = pd.read_csv(train_path)
        
        # Prepare features
        X = train_df.drop(['y'], axis=1, errors='ignore')
        if 'id' in X.columns:
            X = X.drop(['id'], axis=1)
        
        # Convert target
        if train_df['y'].dtype == 'object':
            y = train_df['y'].map({'no': 0, 'yes': 1})
        else:
            y = train_df['y'].astype(int)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature engineering
        fe = FeatureEngineer()
        X_train_fe = fe.fit_transform(X_train)
        X_val_fe = fe.transform(X_val)
        
        # Define features
        numeric_features = X_train_fe.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X_train_fe.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Preprocessing
        numeric_transformer = Pipeline(steps=[('scaler', RobustScaler())])
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop'
        )
        
        # Create complete pipeline
        pipeline = Pipeline([
            ('feature_engineering', fe),
            ('preprocessing', preprocessor),
            ('classifier', XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
            ))
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_val_pred = pipeline.predict(X_val)
        y_val_proba = pipeline.predict_proba(X_val)[:, 1]
        
        metadata = {
            'model': 'XGBoost',
            'train_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'accuracy': float(accuracy_score(y_val, y_val_pred)),
            'precision': float(precision_score(y_val, y_val_pred)),
            'recall': float(recall_score(y_val, y_val_pred)),
            'f1': float(f1_score(y_val, y_val_pred)),
            'roc_auc': float(roc_auc_score(y_val, y_val_proba))
        }
        
        return pipeline, metadata

# Load or train model
@st.cache_resource
def get_model():
    """Get the trained model - train if necessary"""
    
    # Check if training data exists
    possible_paths = ['train (3).csv']
    train_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            train_path = path
            break
    
    if train_path is None:
        st.error("❌ Training data not found. Please ensure train.csv is in the same folder as app.py")
        return None, None
    
    # Train model
    pipeline, metadata = train_model(train_path)
    
    st.success("✅ Model trained successfully!")
    
    return pipeline, metadata

# Initialize model
pipeline, metadata = get_model()

# Header
st.markdown('<p class="main-header">🏦 Bank Term Deposit Prediction System</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x150/1f77b4/ffffff?text=Bank+Logo", use_container_width=True)
    st.markdown("### Navigation")
    page = st.radio("Choose a page:", ["🎯 Single Prediction", "📊 Batch Prediction", "📈 Model Info"])
    
    st.markdown("---")
    st.markdown("### Model Information")
    if metadata:
        st.markdown(f"**Model:** {metadata['model']}")
        st.markdown(f"**Accuracy:** {metadata['accuracy']:.2%}")
        st.markdown(f"**ROC-AUC:** {metadata['roc_auc']:.4f}")

# Page: Single Prediction
if page == "🎯 Single Prediction":
    st.markdown("### Single Customer Prediction")
    st.write("Enter customer details to predict term deposit subscription probability")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**👤 Personal Information**")
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        job = st.selectbox("Job", [
            "admin.", "technician", "services", "management", "retired",
            "blue-collar", "unemployed", "entrepreneur", "housemaid",
            "unknown", "self-employed", "student"
        ])
        marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
        education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    
    with col2:
        st.markdown("**💰 Financial Information**")
        balance = st.number_input("Account Balance (€)", min_value=-10000, max_value=100000, value=1000, step=100)
        default = st.selectbox("Has Credit Default?", ["no", "yes"])
        housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
        loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
    
    with col3:
        st.markdown("**📞 Campaign Information**")
        contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
        day = st.number_input("Day of Month", min_value=1, max_value=31, value=15, step=1)
        month = st.selectbox("Month", [
            "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec"
        ])
        duration = st.number_input("Call Duration (seconds)", min_value=0, max_value=5000, value=180, step=10)
        campaign = st.number_input("Number of Contacts (this campaign)", min_value=1, max_value=50, value=2, step=1)
        pdays = st.number_input("Days Since Last Contact (-1 if never)", min_value=-1, max_value=1000, value=-1, step=1)
        previous = st.number_input("Previous Contacts", min_value=0, max_value=50, value=0, step=1)
        poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "success", "other"])
    
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        if pipeline:
            # Create input dataframe
            input_data = pd.DataFrame({
                'age': [age],
                'job': [job],
                'marital': [marital],
                'education': [education],
                'default': [default],
                'balance': [balance],
                'housing': [housing],
                'loan': [loan],
                'contact': [contact],
                'day': [day],
                'month': [month],
                'duration': [duration],
                'campaign': [campaign],
                'pdays': [pdays],
                'previous': [previous],
                'poutcome': [poutcome]
            })
            
            # Make prediction
            try:
                prediction_proba = pipeline.predict_proba(input_data)[0]
                prediction = pipeline.predict(input_data)[0]
                
                # Display results
                st.markdown("---")
                st.markdown("### Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown(
                            f'<div class="prediction-box positive">✅ LIKELY TO SUBSCRIBE<br>Probability: {prediction_proba[1]:.1%}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="prediction-box negative">❌ UNLIKELY TO SUBSCRIBE<br>Probability: {prediction_proba[0]:.1%}</div>',
                            unsafe_allow_html=True
                        )
                
                with col2:
                    # Probability gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction_proba[1] * 100,
                        title={'text': "Subscription Probability"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkgreen" if prediction == 1 else "darkred"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "gray"},
                                {'range': [70, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation
                st.markdown("---")
                st.markdown("### 💡 Recommendation")
                
                if prediction == 1:
                    st.success(f"""
                    **Action:** Prioritize this customer for term deposit offers.
                    
                    **Strategy:**
                    - Schedule follow-up call at optimal time
                    - Prepare personalized offer based on profile
                    - Allocate experienced agent for conversion
                    - Expected conversion probability: {prediction_proba[1]:.1%}
                    """)
                else:
                    st.warning(f"""
                    **Action:** Low priority for immediate contact.
                    
                    **Strategy:**
                    - Consider alternative products first
                    - Nurture relationship through other channels
                    - Revisit after 3-6 months
                    - Current conversion probability: {prediction_proba[1]:.1%}
                    """)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
        else:
            st.error("Model not loaded. Please check training data.")

# Page: Batch Prediction
elif page == "📊 Batch Prediction":
    st.markdown("### Batch Prediction")
    st.write("Upload a CSV file with multiple customers for batch predictions")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} records")
            
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head(10))
            
            if st.button("🚀 Generate Predictions", type="primary"):
                if pipeline:
                    with st.spinner("Generating predictions..."):
                        # Make predictions
                        predictions_proba = pipeline.predict_proba(df)[:, 1]
                        predictions = (predictions_proba > 0.5).astype(int)
                        
                        # Add to dataframe
                        results_df = df.copy()
                        results_df['prediction'] = predictions
                        results_df['probability'] = predictions_proba
                        results_df['recommendation'] = results_df['prediction'].map({
                            1: 'High Priority',
                            0: 'Low Priority'
                        })
                        
                        # Display summary
                        st.markdown("---")
                        st.markdown("### 📊 Prediction Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Customers", len(results_df))
                        
                        with col2:
                            likely_count = (predictions == 1).sum()
                            st.metric("Likely to Subscribe", likely_count, 
                                     delta=f"{likely_count/len(results_df):.1%}")
                        
                        with col3:
                            avg_prob = predictions_proba.mean()
                            st.metric("Average Probability", f"{avg_prob:.1%}")
                        
                        # Distribution chart
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.histogram(
                                results_df, x='probability', nbins=30,
                                title='Probability Distribution',
                                labels={'probability': 'Subscription Probability'},
                                color_discrete_sequence=['#1f77b4']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.pie(
                                results_df, names='recommendation',
                                title='Priority Distribution',
                                color_discrete_map={
                                    'High Priority': '#2ecc71',
                                    'Low Priority': '#e74c3c'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show results
                        st.markdown("---")
                        st.markdown("### 📋 Detailed Results")
                        st.dataframe(
                            results_df.sort_values('probability', ascending=False),
                            use_container_width=True
                        )
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            type="primary"
                        )
                else:
                    st.error("Model not loaded.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("👆 Upload a CSV file to get started")

# Page: Model Info
elif page == "📈 Model Info":
    st.markdown("### Model Performance Metrics")
    
    if metadata:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metadata['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metadata['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metadata['recall']:.2%}")
        with col4:
            st.metric("ROC-AUC", f"{metadata['roc_auc']:.4f}")
        
        st.markdown("---")
        st.markdown("### 🤖 Model Information")
        
        st.info(f"""
        **Model Type:** {metadata['model']}  
        **Last Trained:** {metadata['train_date']}  
        **F1-Score:** {metadata['f1']:.4f}  
        **Status:** ✅ Active
        """)
        
        # Performance visualization
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values = [
            metadata['accuracy'],
            metadata['precision'],
            metadata['recall'],
            metadata['f1'],
            metadata['roc_auc']
        ]
        
        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line_color='#1f77b4'
        ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            title="Model Performance Metrics"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Model metadata not available")

