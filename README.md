# DiabetesGuard

**DiabetesGuard** is a machine learning-based application aimed at predicting the likelihood of diabetes in patients using key medical data. The app empowers healthcare providers and patients with early insights to enable proactive intervention.

## 🚀 Features

- **Predictive Model**: Utilizes machine learning algorithms for accurate diabetes prediction.
- **Interactive Web Interface**: Built using Streamlit for an easy-to-use interface where users can input data and receive predictions.
- **Detailed Visualizations**: Provides insightful visual analytics for understanding the predictions.
- **Extensibility**: The code is modular, making it easy to extend or improve the application.

## 🧠 Technologies Used

- **Python**: Core programming language.
- **Pandas & NumPy**: Data manipulation libraries.
- **Scikit-Learn**: For building and evaluating machine learning models.
- **Streamlit**: Framework for building the web interface.
- **Matplotlib & Seaborn**: For data visualizations.
  
## 📂 Project Structure

The project repository contains the following key files and directories:

### `web_streamlit.py`
This is the **main file** of the project. It contains the Streamlit web application code, including the user interface and interactions with the machine learning model. The file handles:

- **User Input**: Collecting patient medical data (e.g., glucose, insulin, etc.) through the web interface.
- **Prediction Logic**: Loading the trained machine learning model and using it to make predictions.
- **Visualization**: Displaying the prediction results and additional pictures based on the input data.

### `model.py`
This file contains the machine learning logic. It includes:
- Data preprocessing and feature engineering steps.
- The machine learning algorithm, using **RandomForestClassifier** for classification.
- The training process and evaluation metrics (accuracy, confusion matrix).
- Exporting the trained model using joblib for later use in the app.

### `requirements.txt`
Lists all the dependencies required to run the project, including:
- `pandas`
- `numpy`
- `scikit-learn`
- `streamlit`
- `matplotlib`
- `seaborn`

Make sure to install these using:
```bash
pip install -r requirements.txt
```

### `data/diabetes.csv`
This is the dataset used for training and evaluating the model. It includes the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age
- Outcome (target variable)

You can find the dataset [here](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

## 📊 Visualization Dashboard

Inside the Streamlit app, we provide several visualizations to help users better understand the data and model predictions:

1. **Correlation Heatmap**: Visualizes the correlation between features such as glucose, insulin, BMI, and the likelihood of diabetes.
2. **Prediction Output**: After submitting patient data, a chart appears that shows whether the patient is predicted to have diabetes or not (positive/negative).
3. **Feature Distribution**: Graphs that depict the distribution of important medical features for better understanding of how the dataset looks.

Below is a snapshot of the dashboard:

![Dashboard Example](url_to_dashboard_image)

## 🌐 Web Application Link

You can access the web application here:  
[DiabetesGuard Web App](your-streamlit-web-app-link)

![Streamlit App Interface](url_to_streamlit_app_image)

## 🛠️ Installation & Setup

To run **DiabetesGuard** locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Oelkreamy/DiabetesGuard.git
   cd DiabetesGuard
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/MacOS
   .\venv\Scripts\activate   # For Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 📈 How to Use

1. Open the web app, input the required patient medical data (e.g., glucose level, insulin, BMI, etc.).
2. Click the "Predict" button to get the prediction.
3. View the prediction result and explore the accompanying visualizations.

## 🤝 Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to improve the application, feel free to create a pull request or raise an issue on GitHub.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Make sure to replace `url_to_dashboard_image` and `url_to_streamlit_app_image` with actual URLs or paths to the images/screenshots you'd like to include.

Let me know if you want any further adjustments!
