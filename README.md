# Crime-Forecasting-in-INDIA

## Objective
• To provide aggregate statistics from the dataset such as the highest crime areas.
• To represent key data and findings using a suitable visualization method and tool.
• To cluster applicable data.
• To forecast the trend of crimes for all the states for the next 9 years.

##  MOTIVATION
High or increase crime levels make communities decline, as crimes reduce house prices, neighborhood satisfaction, and the desire to move in a negative manner. To reduce and prevent crimes, and prescribe solutions, Due to large volumes of data and the number of algorithms needed to be applied on crime data, it is unrealistic to do a manual analysis, Therefore, it is necessary to have a platform which is capable of applying any algorithm required to a descriptive, predictive and prescriptive analysis on large volume crime data Through those three methodologies law enforcement weities will be able to take suitable actions to prevent the crimes. Moreover, by predicting the highly likely targets to be attacked, during a specific period of time and specific geographical location, price willbe able to identify better ways to deploy the limited resources and also to find and fix the problems leading to crimes.
Several applications are already developed for crime analysis. Most of these tools are developed to help the 
police to identify difference crime patterns and even to predict criminal activities. They are complex software which needs a lot of training before use. Designing a tool which is easy to use with minimum training would help law-enforcing bodies all around the world to reduce crimes.

## Data Collection
 To download the data: https://www.kaggle.com/datasets/rajanand/crime-in-india .

This dataset contains complete information about various aspects of crimes happened in India from 2001. There are many factors that can be analysed from this dataset. Over all, I hope this dataset helps us to understand better about India.

## DATA EXPLORATION
Steps involved:
• Importing the data, from data.goi website, as a CSV file.
• Importing all the necessary libraries for the study.
• Checking for all the information such as the number of columns and rows present in this dataset 
there are 33 columns present.
• Types of variables: 2 categorical and 30 numerical.
• Checking the count of all the crimes and states.
• Checking for the NULL or NAN value in the dataset.
• Merging the crime data with geographical data that was collected from the website GisDataCollection.

## Models & Training

The output from model training might be utilized for the deduction, which means making expectations on 
new data. A model is a refined portrayal of what a machine learning framework has learned. Machine 
learning models are likened to mathematical capacities they take permission in the form of input data, make 
a forecast or prediction on that input data, and then serve a reaction. In supervised and unsupervised 
machine learning, the model depicts the sign in the noise or the example distinguished from the training 
data.
A machine learning model that was used to forecast the crimes in India for the next 6 years for every state 
was ARIMA. The auto-regressive integrated moving average (ARIMA) is a time series-based model, the 
ARIMA model uses its variable on itself to predict the outcome. In the study, we used total IPC crimes 
committed and years to forecast the crime rates for the future.
Before stating the ML model training, the data is split into 70-30 i.e., 7000 training samples & 3000 testing samples. From the dataset, it is clear that this is a supervised machine learning task. There are two major types of supervised machine learning problems, called classification and regression.


## End Results
From the obtained results of the above model, ARIMA model shows performance of 74%.
