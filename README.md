# DiabetesGuard

**DiabetesGuard** is a machine learning-powered web application designed to predict the likelihood of diabetes in individuals based on health metrics. The app provides healthcare professionals and users with insights derived from the **Pima Indians Diabetes Dataset** to detect early signs of diabetes.

## 🚀 Features

- **Diabetes Prediction**: Utilizes machine learning models for predicting the probability of diabetes.
- **Interactive User Interface**: Built using Streamlit for ease of use, allowing users to input personal health metrics and receive predictions.
- **Data Insights Dashboard**: A separate dashboard for visualizing data trends and feature importance using Dash and Plotly.
- **Health Metrics Visualizations**: Displays key visual analytics like feature distributions, correlation heatmaps, and breakdowns by diabetes outcome.

## 🧠 Technologies Used

- **Python**: Core programming language for model building and app development.
- **Pandas & NumPy**: Libraries for data manipulation.
- **Scikit-Learn**: Used for training the machine learning model.
- **Streamlit**: Web framework for the front-end of the prediction app.
- **Dash & Plotly**: Used to create a data visualization dashboard.
- **Matplotlib & Seaborn**: For generating data plots in the dashboard.

## 📂 Project Structure

The project contains the following key files:

### `web_streamlit.py`
This file contains the main **Streamlit web application** responsible for the user interface. It allows users to input their health data (like glucose levels, insulin, BMI) and get a prediction on the likelihood of having diabetes. The file includes:
- **Model loading**: Loads the pre-trained machine learning model (`Diabetes_prediction.sav`) and a scaler for input normalization.
- **Feature inputs**: Provides an interactive sidebar where users can input health metrics such as **Pregnancies**, **Glucose Level**, **Blood Pressure**, etc.
- **Prediction logic**: Once the inputs are submitted, it scales the data and uses the model to predict whether the user is likely to have diabetes.

### `Dashboard.py`
This file builds the **data visualization dashboard** using Dash and Plotly. It includes:
- **Data Distribution Visualizations**: Displays histograms for features like **Glucose**, **BMI**, and **Age**.
- **Correlation Heatmap**: Shows a heatmap of correlations between different health metrics.
- **Feature Importance**: A bar chart highlighting the most important features used in the model for diabetes prediction.
- **Outcome Breakdown**: Visualizes the relationship between features like **Glucose**, **BMI**, and **Age** by diabetes outcome (positive or negative).

### `diabetes_project.ipynb`
This Jupyter Notebook contains the **model training code**. It outlines the data preprocessing, training of the machine learning model using **RandomForestClassifier**, and evaluation metrics (such as accuracy and confusion matrix). The trained model is then saved as `Diabetes_prediction.sav` for use in the Streamlit app.

### `requirements.txt`
A file listing all the dependencies required to run the project, such as:
- `pandas`
- `numpy`
- `scikit-learn`
- `streamlit`
- `dash`
- `plotly`
- `matplotlib`
- `seaborn`

Make sure to install them using:
```bash
pip install -r requirements.txt
```

## 📊 Visualization Dashboard

The dashboard provides multiple visualizations to help users understand the data and how the model makes predictions:

1. **Distribution Plots**: Histograms for health features like **Glucose**, **BMI**, and **Age**.
2. **Correlation Heatmap**: Shows how features like **Glucose** and **Insulin** correlate with each other.
3. **Feature Importance Plot**: Highlights which features contribute most to the model’s predictions.
4. **Outcome Analysis**: Box plots that break down features by diabetes outcome (positive or negative).

### Dashboard Snapshot
![Dashboard Example](https://drive.google.com/uc?export=view&id=1c6XG0VTPQis8kOPU0WOsiqMEGLZ1nfxu)

## 🌐 Web Application Link

You can access the Streamlit web application here:  
[DiabetesGuard Web App](https://oelkreamy-diabetesguard-sourceweb-streamlit-vsfcxd.streamlit.app/)

### Web Application Screenshot
![Streamlit App Interface](https://drive.google.com/uc?export=view&id=1qvv7rAtlmy5UZQMAuczKssvyZcDe2rss)

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
   streamlit run web_streamlit.py
   ```

5. **Run the Dashboard:**
   ```bash
   python Dashboard.py
   ```

## 📈 How to Use

1. **For Prediction**: Open the Streamlit app, input the health data such as **Glucose**, **BMI**, and **Insulin**, and press "Predict" to see if the individual is likely to have diabetes.
2. **For Insights**: Open the dashboard to explore the data through interactive visualizations, including feature importance and correlations.

## 🤝 Contributing

Contributions are welcome! If you have suggestions, bug reports, or want to improve the application, feel free to create a pull request or raise an issue on GitHub.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
