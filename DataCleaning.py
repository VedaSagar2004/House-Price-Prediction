# Importing libraries
from geopy.geocoders import Nominatim
import pandas as pd 
import numpy as np

# Reading the Dataset
df = pd.read_csv('Hyderabad.csv')

# Initialize geolocator
geolocator = Nominatim(user_agent='ProjectExhibition.py')

# Define function to get location coordinates
def get_location_coordinates(location):
    query = str(location) + ', Hyderabad, India'
    location = geolocator.geocode(query, timeout=10)
    
    if location:
        return location.latitude, location.longitude
        print(query)
    else:
        return np.nan, np.nan
    

# Add new columns for latitude and longitude coordinates
df[['Latitude', 'Longitude']] = df['Location'].apply(get_location_coordinates).apply(pd.Series)

# Remove rows where coordinates are not found
df.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Removing null values
df.replace(9, np.nan, inplace=True)
df.dropna(inplace=True)

# Save updated dataframe to new CSV file
df.to_csv('Data.csv', index=False)

# Print message to confirm file was saved
print('File saved as Data.csv')

