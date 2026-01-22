🎓 AI Placement Predictor
An end-to-end Machine Learning + Streamlit project that predicts whether a student will be placed or not based on academic performance, skills, and readiness indicators.

This project demonstrates the complete ML workflow:

• Data preprocessing
• Exploratory Data Analysis (EDA)
• Model training & evaluation
• Model serialization
• Web app deployment using Streamlit


📌 Problem Statement
Campus placement decisions depend on multiple factors such as:

• Academic scores
• Programming skills
• Attendance
• Mini projects
• Communication skills
• Placement readiness

This project builds a machine learning model to predict placement outcomes (Yes / No) using these factors.


📊 Dataset Description

The dataset contains the following columns:

Feature	        	            Description
•Maths	        	            Maths marks
•Python	        	            Python marks
•SQL	            	        SQL marks
•Attendance	    	            Attendance percentage
•Mini_Projects		            Mini project score (0–5)
•Communication_Score		    Communication skill score (0–10)
•Placement_Readiness_Score      Overall readiness score
•Placed	                        Target variable (Yes / No)


🧠 Machine Learning Pipeline

1.Data Cleaning
 • Handle missing values
 • Encode target labels

2.Feature Selection
 • Numerical academic & skill features

3.Preprocessing
 • Feature scaling using StandardScaler

4.Model Training
 • Classification model using scikit-learn

5.Model Evaluation
 • Accuracy, classification report

6.Model Saving
 • model.pkl
 • scaler.pkl


🖥️ Web Application (Streamlit)

A user-friendly web interface allows users to:
 • Enter student details
 • Predict placement outcome in real time


📂 Project Structure

A-P-P/
│
├── app/
│   ├── app.py
│   ├── model.pkl
│   └── scaler.pkl
│
├── data/
│   └── student_placement_predictor_dataset.csv
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
│
├── eda_and_model_training.ipynb
├── train_save.py
├── requirements.txt
├── .gitignore
└── README.md


⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/cp1928/AI-Placement-Predictor.git
cd A-P-P


2️⃣ Create virtual environment
python -m venv .venv
.venv\Scripts\activate


3️⃣ Install dependencies
pip install -r requirements.txt


🚀 Run the Application
• streamlit run app/app.py


The app will be available at:

http://localhost:8501



📈 Model Output

• Placed ✅
• Not Placed ❌

Predictions are based on the trained ML model and scaled inputs.


🛠️ Technologies Used

• Python
• Pandas, NumPy
• Scikit-learn
• Matplotlib, Seaborn
• Streamlit
• VS Code
• Git & GitHub


📌 Future Improvements

• Improve model accuracy with advanced algorithms
• Add feature importance visualization
• Deploy app on Streamlit Cloud
• Add user authentication
• Collect real-world data

Note:
The dataset’s communication score ranges from 0 to 100, while the app accepts input from 0 to 10. To match the input format, we normalize the communication score by dividing it by 10 during preprocessing before training and prediction.