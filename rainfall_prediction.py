import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import gradio as gr


ds = pd.read_csv('fall.csv')
ds = ds.drop(['month', 'day'], axis=1)


x = ds.iloc[:, :7].values
y = ds.iloc[:, 7].values


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)


regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(x_train, y_train)


def predict_fall_probability(year, avgtemp, avgdp, avghumidity, avgslp, avgvisibility, avgwind):
    features = np.array([[year, avgtemp, avgdp, avghumidity, avgslp, avgvisibility, avgwind]])
    prediction = regressor.predict(features)[0]
    return float(prediction)


plt.scatter(x_train[:,1], y_train, color='blue')
plt.title('Rainfall prediction (Training set)')
plt.xlabel('Temperature')
plt.ylabel('Rainfall')
plt.show()


iface = gr.Interface(
    fn=predict_fall_probability,
    inputs=["number", "number", "number", "number", "number", "number", "number"],
    outputs="number",
    title="Fall Probability Prediction",
    description="Predicts rainfall probability based on input weather features.",
    examples=[[2035, 20, 22, 90, 1005, 4, 19]]
)

iface.launch(share=True, inline=False)
