import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):

    # Standardize the column text (lowercase & strip spaces)
    df["Placed"] = df["Placed"].astype(str).str.lower().str.strip()

    # Map yes/no to 1/0
    df["Placed"] = df["Placed"].map({"no": 0, "yes": 1})

    # Remove rows where placed is NaN
    df = df.dropna(subset=["Placed"])

    # Convert to integer
    df["Placed"] = df["Placed"].astype(int)

    X = df[["Maths", "Python", "SQL", "Attendance", "Mini_Projects",
            "Communication_Score", "Placement_Readiness_Score"]]
    y = df["Placed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
