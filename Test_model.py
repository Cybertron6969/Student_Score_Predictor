# 1. Import required libraries
import joblib
import pandas as pd

# 2. Load the saved model
model = joblib.load("score_prediction_model.pkl")

# 3. Predict scores for new student
# Ensure the columns exactly match the ones you used for training
data=pd.DataFrame([[4.0, 85.0, 75.0, 7.5, 3.0, 1.0, 1.0]],
                  columns=['study_hrs_per_day', 'attendance', 'previous_score', 'sleep_hours', 'practice_tests_completed', 'internet_access', 'extracurricular_activity'])

predicted_score = model.predict(data)

print(f"Predicted Score: {predicted_score[0]:0.2f} marks")
