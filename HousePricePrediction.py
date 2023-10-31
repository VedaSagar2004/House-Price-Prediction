# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geopy.geocoders import Nominatim
from tkinter import *
from tkinter import ttk
from tkinter import Tk
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

# Reading the data
df = pd.read_csv('Data.csv')
X = df.iloc[:, [1,3,-1,-2]].values
y = df.iloc[:, 0].values




# Create the GUI
root = Tk()
root.geometry('400x200')
root.title('House Price Predictor')

# Create a dropdown menu with locations from the dataframe
locations = sorted(df['Location'].unique())
location_var = StringVar()
location_dropdown = ttk.Combobox(root, textvariable=location_var, values=locations)
location_dropdown.grid(row=2, column=1, padx=5, pady=5)

geolocator = Nominatim(user_agent='ProjectExhibition.py')

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Create a function to handle the predict button
def predict_price():
    area = float(area_entry.get())
    bedrooms = int(bedrooms_entry.get())
    location = location_var.get()

    # Get the latitude and longitude for the selected location
    location_input = geolocator.geocode(location + ', Hyderabad, India', timeout=10)
    location_latitude = location_input.latitude
    location_longitude = location_input.longitude

    # create a 2D numpy array with the input values
    X_custom = np.array([[area, bedrooms, location_latitude, location_longitude]])

    # Use the trained model to predict the price
    y_custom_pred = regressor.predict(X_custom)
    predicted_price_label.configure(text="Predicted price: {} Rupees".format(round(y_custom_pred[0], 2)))

area_label = ttk.Label(root, text="Area (sq ft): ")
area_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
area_entry = ttk.Entry(root)
area_entry.grid(row=0, column=1, padx=5, pady=5)

bedrooms_label = ttk.Label(root, text="No. of Bedrooms: ")
bedrooms_label.grid(row=1, column=0, padx=5, pady=5, sticky='w')
bedrooms_entry = ttk.Entry(root)
bedrooms_entry.grid(row=1, column=1, padx=5, pady=5)

location_label = ttk.Label(root, text="Location: ")
location_label.grid(row=2, column=0, padx=5, pady=5, sticky='w')

predict_button = ttk.Button(root, text="Predict", command=predict_price)
predict_button.grid(row=3, column=0, padx=5, pady=5)

predicted_price_label = ttk.Label(root, text="Predicted price: ")
predicted_price_label.grid(row=3, column=1, padx=5, pady=5, sticky='w')

root.mainloop()

# Concatenating predicted target values and original target values
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# Evaluating the model

mse = mean_squared_error(y_test, y_pred)
print(f'MSE : {mse}')
r2 = r2_score(y_test, y_pred)
print(f'R-squared : {r2}')
pc = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE : {pc}')
