# 1. Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 2. Load dataset
df = pd.read_csv("data.csv")

# Delete/Drop unnecessary column (As 'id' does not affect the final_score)
df = df.drop('id', axis=1)

# Handle missing values by replacing them with mean value
df.fillna(df[['internet_access','extracurricular_activity']].mean().round(), inplace=True) # Use round as these categories require only 1/0 for True/False 
df.fillna(df.mean(), inplace=True)

# 3. Seperate into Features(X) and Target(y)
X = df.drop('final_score', axis=1)
y = df['final_score']

# 4. Perform the Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # Setting random_state makes the experiments reproducable


# 5. Initialize and Train Machine Learning Model
model=LinearRegression()
model.fit(X_train, y_train) # In ".fit()" command the model uses math to find the perfect weight for every single feature in X_train to calculate y_train.
print("Model Trained Successfully")

# 6. Use model to predict final scores for hidden test set
y_predicted = model.predict(X_test)

# 7. Calculate Errors
mae = mean_absolute_error(y_test, y_predicted)
mse = mean_squared_error(y_test, y_predicted)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae} marks")
print(f"Root Mean Squared Error: {rmse} marks")

# 8. Visualizing Accuracy of model

# Plot Actual vs Predicted scores
plt.scatter(y_test, y_predicted, alpha=0.8)

# Draw the "Perfect Prediction" line
plt.plot([y_test.min(), y_test.max()], [y_predicted.min(), y_predicted.max()], lw=2, linestyle="--", alpha=0.5, color='g', label='Perfect Prediction')

plt.title("Actual vs Predicted Scores")
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()

# 9. Save the trained model
joblib.dump(model, 'Score_prediction_model.pkl')
print("Model Saved Successfully")