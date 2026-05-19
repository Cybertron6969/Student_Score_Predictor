# 🎓 Student Performance Predictor

This is a beginner-friendly Machine Learning pipeline that tries to answer a classic question: *"If I study this much, attend my classes, and get enough sleep, what score will I get on my final?"*

## 🤔 What does this project actually do?
This project walks through the entire journey of taking raw, messy data and turning it into a smart prediction tool. Here is how I built it:

1. 🧹 **Cleaning the Data (Pandas):** Real-world data is rarely perfect. I had students missing attendance records or test scores, so I used Pandas to intelligently fill in those blanks without skewing the results.
2. 📊 **Finding the Trends (Matplotlib):** Before making the computer guess anything, I created scatter plots to visually check if studying more *actually* led to better grades in this specific dataset.
3. 🧠 **Teaching the AI (Scikit-Learn):** I used a technique called Multiple Linear Regression. Basically, I fed 80% of the student data to the computer and let it figure out the "secret mathematical formula" for a good grade.
4. 📏 **Grading the AI (Evaluation):** I took the remaining 20% of the data (which the model had never seen) and gave the AI a "final exam." I measured how many points its guesses were off by using MAE and RMSE.
5. 💾 **Saving the Brain (Joblib):** Once the model was smart enough, I saved it as a file (`score_prediction_model.pkl`). Now, anyone can use it to predict a score instantly without having to retrain it from scratch!

## 🚀 How to Run This on Your Computer

Want to see what score the model predicts for you? Follow these steps:

1. **Download the code:** Clone this repository to your computer.
2. **Install the tools:** Open your terminal and install the required Python libraries by typing:
   `pip install -r requirements.txt`
3. **Train the model:** Run `train_model.py`. This will clean the data, train the AI, and save the model to your folder.
4. **Make a prediction:** Open `test_model.py`, type in your own study habits, and run the script to see your predicted final score!

---
**Created by:** Yug Patel | Roll No: B25CH1048
