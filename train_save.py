import pandas as pd
import pickle
from src.data_preprocessing import preprocess_data
from src.model_training import train_model

# Load data
df = pd.read_csv(r"C:\Users\cp055\Desktop\a-p-p\data\student_placement_predictor_dataset.csv")

# Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Train model
model = train_model(X_train, y_train)

# Save model and scaler
pickle.dump(model, open("app/model.pkl", "wb"))
pickle.dump(scaler, open("app/scaler.pkl", "wb"))

print("Model and scaler saved successfully!")