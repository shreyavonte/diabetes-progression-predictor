import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes

# Load dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Train model (using top 2 features: 'bmi' and 's5' for simplicity)
X = df[['bmi', 's5']]  # BMI and Glucose (s5)
y = df['target']
model = Lasso(alpha=0.1)
model.fit(X, y)

# Streamlit UI
st.set_page_config(page_title="Diabetes Progression Predictor", page_icon="🩺")
st.title("🩺 Diabetes Progression Predictor")
st.markdown("""
Predict disease progression (1-year change) based on **BMI** and **Glucose** levels.
*Data is normalized (mean=0, std=1).*
""")

# Interactive sliders
col1, col2 = st.columns(2)
with col1:
    bmi = st.slider("BMI", min_value=-0.1, max_value=0.2, value=0.0, step=0.01, 
                   help="Normalized BMI (higher values indicate higher risk)")
with col2:
    glucose = st.slider("Glucose (s5)", min_value=-0.1, max_value=0.2, value=0.0, step=0.01,
                       help="Normalized serum glucose level")

# Predict button
if st.button("Predict Progression"):
    prediction = model.predict([[bmi, glucose]])[0]
    st.subheader(f"Predicted 1-year progression: `{prediction:.2f}`")
    st.progress(abs(prediction / 350))  # Visualize magnitude (adjust divisor as needed)
    
    # Interpretation
    if prediction > 100:
        st.warning("⚠️ High risk: Significant progression expected. Consult a doctor.")
    elif prediction > 50:
        st.info("ℹ️ Moderate risk: Monitor lifestyle factors.")
    else:
        st.success("✅ Low risk: Stable progression.")

# Key insights
st.markdown("---")
st.subheader("🔍 Model Insights")
st.write("""
- **Top Features**: 
  - `BMI` (Weight-to-height ratio) 
  - `Glucose (s5)` (Blood sugar level)
- **Interpretation**: Higher values → Worse disease progression.
""")

# Optional: Show raw data
if st.checkbox("Show raw data"):
    st.dataframe(df.head(10))