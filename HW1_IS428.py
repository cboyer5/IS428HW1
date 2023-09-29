#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd 
import matplotlib as plt 
import seaborn as sns
import matplotlib.pyplot as plt


# In[8]:


df=pd.read_csv('Baltimore911.csv')
df


# In[13]:


df.drop_duplicates(inplace=True)
df


# In[242]:


df['Location2'] = df['Latitude'].astype(str) + ', ' + df['Longitude'].astype(str)

# Drop the original 'Latitude' and 'Longitude' columns if needed
df.drop(['Latitude', 'Longitude'], axis=1, inplace=True)

# Display the DataFrame
print(df)


# In[235]:


df.drop(columns=['CrimeCode', 'Inside/Outside', 'Post', 'Longitude', 'Latitude', 'Location 1', 'vri_name1'], inplace=True)


# In[10]:


import pandas as pd

# Sample DataFrame (replace with your actual DataFrame)
data = {
    'CrimeDate': df.CrimeDate,
    'CrimeTime': df.CrimeTime,
    'Location': df.Location,
    'Description': df.Description,
    'Weapon': df.Weapon,
    'District': df.District,
    'Neighborhood': df.Neighborhood,
    'Premise': df.Premise,
    'Total Incidents': df['Total Incidents'],
    'CrimeCode': df.CrimeCode,
    'Inside/Outside':df['Inside/Outside'],
    'Post': df.Post,
    'Longitude':df.Longitude,
    'Latitude': df.Latitude,
    'Location 1': df['Location 1'],
    'vri_name1': df.vri_name1
}

df = pd.DataFrame(data)

# Create a profile DataFrame
profile_df = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': df.dtypes,
    'Count of Unique Values': df.nunique(),
    'Min Value': df.min(),
    'Max Value': df.max()
})

# Display the profile DataFrame
print(profile_df)


# In[121]:


import pandas as pd

# Assuming you have a DataFrame named 'df' with your data
# You should replace 'df' with the actual variable name you're using.

# Create an empty DataFrame to store the profile information
profile_df = pd.DataFrame(columns=["Column Name", "Data Type", "Count of Unique Values", "Min Value", "Max Value"])

# Iterate through each column in your DataFrame
for column_name in df.columns:
    data_type = df[column_name].dtype
    unique_count = df[column_name].nunique()
    
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(df[column_name]):
        min_value = df[column_name].min()
        max_value = df[column_name].max()
    else:
        min_value = None
        max_value = None

    # Append the information to the profile DataFrame
    profile_df = profile_df.append({
        "Column Name": column_name,
        "Data Type": data_type,
        "Count of Unique Values": unique_count,
        "Min Value": min_value,
        "Max Value": max_value
    }, ignore_index=True)

# Display the profile table
print(profile_df)


# In[ ]:





# In[183]:


import pandas as pd

# Assuming you have a DataFrame named 'df' with your data
# Replace 'df' with the actual variable name you're using.

# Extract the hour part from 'CrimeTime' and create a new column
df['Hour'] = df['CrimeTime'].str.split(':').str[0]

# Display the updated DataFrame
#print(df[['CrimeTime', 'Hour']])
df


# In[181]:


df['Premise'].value_counts()


# In[188]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with your data
# Replace 'df' with the actual variable name you're using.

# Convert 'CrimeTime' to strings


# Extract the 'CrimeTime' column
hour= df['Hour'].astype(str)

# Create a histogram using Matplotlib
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
plt.hist(hour, bins=6, color='skyblue', edgecolor='black')
plt.title('Distribution of Crime Occurrence by Time')
plt.xlabel('Time of Day (Hour)')
plt.ylabel('Frequency')
plt.xticks(range(24))  # Adjust the x-axis ticks for hours
plt.grid(axis='y', alpha=0.75)

plt.show()


# In[196]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'df' with your data
# Replace 'df' with the actual variable name you're using.

# Convert 'CrimeTime' to strings (assuming it's not already a string)
df['Hour'] = df['Hour'].astype(str)
#df.sort_values(by='Hour', inplace=True)
# Create a histogram using Matplotlib
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
plt.hist(df['Hour'], bins=300, color='skyblue', edgecolor='black')
plt.title('Distribution of Crime Occurrence by 24 hr Time')
plt.xlabel('Time of Day (24 Hour time )')
plt.ylabel('Frequency')
plt.xticks(range(24))  # Adjust the x-axis ticks for hours
plt.grid(axis='y', alpha=0.75)

plt.show()



# In[210]:


import pandas as pd

# Sample data as a Pandas Series
date_series = pd.Series(df['CrimeDate'].astype(str))

# Split the date strings using a custom delimiter (e.g., '/')
year_series = date_series.str.split('/').str[2]

print(year_series)
date_series


# In[211]:


year_series.groupby()


# In[28]:


incidentbydate=df.groupby(['CrimeDate']).sum()
incidentbydate


# In[35]:


incidentbydate2 = incidentbydate[incidentbydate['Total Incidents'] > 175]
incidentbydate2


# In[47]:


import pandas as pd

# Sample data in your existing DataFrame 'df'


# Convert 'CrimeDate' column to datetime
df['CrimeDate'] = pd.to_datetime(df['CrimeDate'])

# Extract the year from the 'CrimeDate' column
df['Year'] = df['CrimeDate'].dt.year

# Group by the year and sum the incidents
incident_by_year = df.groupby('Year')['Total Incidents'].sum().reset_index()

print(incident_by_year)


# In[36]:


crimesbymonth = df['Total Incidents'].groupby(df.CrimeDate).sum()
incidentbyday=pd.DataFrame(crimesbymonth,)
incidentbyday.tail(50)
#notice how up until 2010s ther is more crime


# In[153]:


crimes_by_month = df['Total Incidents'].groupby(df['CrimeDate']).sum()

# Convert the resulting Series to a DataFrame
incident_by_day = pd.DataFrame(crimes_by_month)

# Filter the DataFrame to include dates with incident count > 5
incident_by_day_filtered = incident_by_day[incident_by_day['Total Incidents'] > 50]

incident_by_day_filtered.mean()
incident_by_day_filtered


# In[166]:


df.drop_duplicates()


# In[17]:


missing_values = df.isna()

# Step 4: Count the missing values in each column
missing_counts = missing_values.sum()

# Step 5: Calculate the percentage of missing values
total_rows = df.shape[0]
missing_percentage = (missing_counts / total_rows) * 100

print(missing_percentage)


# In[23]:


profile_df = pd.DataFrame(columns=["Column Name", "Mean", "Median", "Std Deviation"])

# Iterate through each column in your DataFrame
for column_name in df.columns:
  #  data_type = df[column_name].dtype
   # unique_count = df[column_name].nunique()
    
    # Check if the column contains numeric data
    if pd.api.types.is_numeric_dtype(df[column_name]):
        #min_value = df[column_name].min()
        #max_value = df[column_name].max()
        mean_value = df[column_name].mean()
        median_value = df[column_name].median()
        std_deviation = df[column_name].std()
    else:
      #  min_value = None
      #  max_value = None
        mean_value = None
        median_value = None
        std_deviation = None

    # Append the information to the profile DataFrame
    profile_df = profile_df.append({
        "Column Name": column_name,
       # "Data Type": data_type,
        #"Count of Unique Values": unique_count,
        #"Min Value": min_value,
        #"Max Value": max_value,
        "Mean": mean_value,
        "Median": median_value,
        "Std Deviation": std_deviation
    }, ignore_index=True)

# Display the profile table
print(profile_df)


# In[155]:


df['CrimeDate'] = pd.to_datetime(df['CrimeDate'])

# Extract the year from the 'CrimeDate' column
df['Year'] = df['CrimeDate'].dt.year
incident_by_year = df.groupby('Year')['Total Incidents'].sum().reset_index()

# Create a line plot to show the trend of incidents by year
plt.figure(figsize=(10, 6))
plt.plot(incident_by_year['Year'], incident_by_year['Total Incidents'], marker='o', linestyle='-')
plt.title('Incidents by Year')
plt.xlabel('Year')
plt.ylabel('Total Incidents')
plt.grid(True)
plt.show()



# In[ ]:





# In[156]:


popular_crimes= df['Description'].value_counts()
popular_crimes
#most common crimes


# In[158]:


cross_tab=pd.crosstab(df['Description'], df['District'])
cross_tab.idxmax() 
#most common crimes by district


# In[163]:


cross_tab.plot(kind='bar', title='Most popular crimes by district')

# Customize labels and axis
plt.xlabel('Crimes')
plt.ylabel('Total Incidents')


# Show the plot
plt.show()


# In[21]:


cross_tab2=pd.crosstab(df['Hour'], df['District'])
cross_tab2.idxmax() 
#most common times crimes were reported by district


# In[22]:


cross_tab3=pd.crosstab(df.District, df.Hour )
cross_tab3.head()


# In[23]:


df['District'].value_counts()


# In[ ]:





# In[24]:


hourly24 =pd.Series([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23])
crimetime= pd.DataFrame(cross_tab3)
crimetime
#calls per district at each hour


# In[177]:


df


# In[30]:


#cross_tab4 = pd.crosstab(df['District'], hourly24, values=df['Hour'], aggfunc='mean')
cross_tab3=pd.crosstab(df.District, df.Hour )

# Create a heatmap
plt.figure(figsize=(50, 10))
sns.heatmap(cross_tab3, cmap='YlGnBu', annot=True, fmt=".1f", cbar=True,xticklabels=hourly24)
plt.title('Crime by Region and Time')
plt.xlabel('Hour')
plt.ylabel('Region')
plt.show()


# In[ ]:


# Create a cross-tabulation (crosstab) of District and Total Incidents by Year
cross_tab5 = pd.crosstab(df['District'], df['Year'], values=df['Total Incidents'], aggfunc='sum')

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab5, cmap='YlGnBu', annot=True, fmt=".1f", cbar=True, xticklabels= df['Year'])
plt.title('Crime by Region and Year')
plt.xlabel('Year')
plt.ylabel('Region')
plt.show()


# In[26]:


df['CrimeDate'] = pd.to_datetime(df['CrimeDate'])
df['Month'] = df['CrimeDate'].dt.month

# Group data by year and calculate the sum of incidents
monthy_incidents = df.groupby('Month')['Total Incidents'].sum()

monthy_incidents.plot(kind='bar',title ='Incident trends aggregated by month')
monthy_incidents


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Sample data (replace with your data)
locations = df['Location2']
values = df['Total Incidents']

# Define a scaling factor (you can adjust this based on your data)
scaling_factor = 10  # Adjust as needed

# Create a Matplotlib figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Plot each point as a circle with size proportional to the value
for loc, value in zip(locations, values):
    lat_str, lon_str = loc.split(', ')  # Split the 'loc' string into latitude and longitude
    lat, lon = float(lat_str), float(lon_str)  # Convert latitude and longitude to float
    circle = plt.Circle((lon, lat), value / scaling_factor, alpha=0.5)
    ax.add_patch(circle)

# Set axis limits (you can adjust this based on your data)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

# Customize the map appearance (e.g., title, labels, etc.)
plt.title('Proportional Symbol Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)

# Show the map
plt.show()



# In[239]:


df


# In[ ]:




