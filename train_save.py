import pandas as pd
import pickle
from src.data_preprocessing import preprocess_data
from src.model_training import train_model

df = pd.read_csv(r".......\data\student_placement_predictor_dataset.csv")


X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

model = train_model(X_train, y_train)

from src.model_training import evaluate_model
accuracy, report, confusion_mat = evaluate_model(model, X_test, y_test)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", confusion_mat)


pickle.dump(model, open("app/model.pkl", "wb"))
pickle.dump(scaler, open("app/scaler.pkl", "wb"))

print("\nModel, scaler, and evaluation metrics generated successfully!")
