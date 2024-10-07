import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import io
import base64

# Load the dataset
df = pd.read_csv(r'D:\OMAR\courses\depi-IBM-Data-Science\DiabetesGuard\dataset\diabetes.csv')

# Initialize the Dash app
app = dash.Dash(__name__)

# Create a function to plot distributions
def create_distribution_plot(column, color):
    fig = px.histogram(df, x=column, nbins=50, marginal="box", 
                       title=f'Distribution of {column}', color_discrete_sequence=[color])
    return fig

# Create correlation heatmap as an image
def create_correlation_heatmap():
    corr = df.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    buf.seek(0)
    image = base64.b64encode(buf.read()).decode('utf-8')
    return "data:image/png;base64,{}".format(image)

# Feature importance function
def get_feature_importances():
    X = df.drop(columns=['Outcome'])
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    feature_names = X.columns

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    return px.bar(feature_importance, x='Importance', y='Feature', orientation='h', 
                  title="Feature Importance for Diabetes Prediction", color='Importance')

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Diabetes Dataset Insights Dashboard", style={'text-align': 'center'}),
    
    # Glucose, BMI, and Age Distributions
    html.Div([
        html.Div([
            dcc.Graph(figure=create_distribution_plot('Glucose', 'skyblue')),
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=create_distribution_plot('BMI', 'lightgreen')),
        ], style={'width': '33%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(figure=create_distribution_plot('Age', 'salmon')),
        ], style={'width': '33%', 'display': 'inline-block'})
    ]),
    
    # Correlation Heatmap
    html.H2("Correlation Heatmap", style={'text-align': 'center'}),
    html.Img(src=create_correlation_heatmap(), style={'width': '100%', 'height': 'auto'}),
    
    # Feature Importance Plot
    html.H2("Feature Importance", style={'text-align': 'center'}),
    dcc.Graph(id='feature-importance', figure=get_feature_importances()),
    
    # Outcome Breakdown 
    html.H2("Breakdown by Outcome", style={'text-align': 'center'}),
    dcc.Tabs(id="tabs-outcome", value='tab-1', children=[
        dcc.Tab(label='Glucose by Outcome', value='tab-1'),
        dcc.Tab(label='BMI by Outcome', value='tab-2'),
        dcc.Tab(label='Age by Outcome', value='tab-3'),
    ]),
    html.Div(id='tabs-content-outcome')
])

# Callback for Outcome Breakdown
@app.callback(
    Output('tabs-content-outcome', 'children'),
    [Input('tabs-outcome', 'value')]
)
def render_content(tab):
    if tab == 'tab-1':
        return dcc.Graph(figure=px.box(df, x='Outcome', y='Glucose', color='Outcome',
                                       title="Glucose Levels by Outcome"))
    elif tab == 'tab-2':
        return dcc.Graph(figure=px.box(df, x='Outcome', y='BMI', color='Outcome',
                                       title="BMI by Outcome"))
    elif tab == 'tab-3':
        return dcc.Graph(figure=px.box(df, x='Outcome', y='Age', color='Outcome',
                                       title="Age by Outcome"))

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
