![Human Activity Recognition Using Smartphones Sensor DataSet](https://user-images.githubusercontent.com/49862149/91008443-082e6900-e5fc-11ea-9099-caa8d9a8d071.jpg)

## Overview
Smart phones have become a most useful tool in our daily life for communication with advanced technology provided intelligent assistance to the user in their everyday activities. The portable working framework with computing ability and interconnectivity, application programming interfaces for executing outsiders’ tools and applications, mobile phones have highlights such as cameras, GPS, web browsers so on., and implanted sensors such as **accelerometers** and **gyroscope** which permits the improvement of applications in view of client’s specific area, movement and context.

__Activity Recognition__ (AR) is monitoring the liveliness of a person by using smart phone. Smart phones are used in a wider manner and it becomes one of the ways to identify the human’s environmental changes by using the sensors in smart mobiles. *Smart phones are equipped in detecting sensors like gyroscope and accelerometer*. The contraption is demonstrated to examine the state of an individual. 

__Human Activity Recognition__ (HAR) framework *collects the raw data from sensors and observes the human movement using different deep learning approach*. Deep learning models are proposed to identify motions of humans with plausible high accuracy by using sensed data. 

__HAR Dataset from UCI dataset storehouse is utilized__. This dataset is collected from 30 persons (referred as subjects in this dataset), performing different activities with a smartphone to their waists. The data is recorded with the help of sensors (*accelerometer and Gyroscope*) in that smartphone. This experiment was video recorded to label the data manually.

This project is to build a model that *predicts the human activities* such as __Walking, Walking_Upstairs, Walking_Downstairs, Sitting, Standing__ and __Laying__.

## Sources/Useful Links
- __Blog 1__ : https://www.ijrte.org/wp-content/uploads/papers/v8i1/A1385058119.pdf

- __Blog 2__ : https://machinelearningmastery.com/how-to-model-human-activity-from-smartphone-data/

- __For HAR Data_Set__ : https://archive.ics.uci.edu/ml/datasets/Smartphone+Dataset+for+Human+Activity+Recognition+%28HAR%29+in+Ambient+Assisted+Living+%28AAL%29

## Problem Statement
![#FF5733](https://via.placeholder.com/7x24/FF5733/000000?text=+) Given a new datapoint we have to predict the Human Activity.

## Solution
We have a fairly small data set of Human Activity Recognition that has been labeled as “**Walking**”, “**Walking Upstairs**”, “**Walking Downstairs**”, “**Standing**”, “**Sitting**” and “**Lying**”.  We had downloaded HAR Dataset from UCI dataset storehouse and we know that the data set is define in two part first is RAW data set and second is pre-engineered by domain or signal expert engineer. **So first**, we use pre-engineered dataset with classical machine learning (ML) to learn from the data, and predict the human Activity. **Second**, we could then use RAW dataset with Deep learning model to learn from the data, and predict the human Activity.

## Which type of ML Problem is this?
Human activity recognition, **is a challenging time series classification task.**
It involves predicting the movement of a person based on sensor data and traditionally involves deep domain expertise and methods from signal processing to correctly engineer features from the raw data in order to fit a machine learning model.

OR in other words you can call, it is a **multiclass classification problem**, for given a new datapoint we have to predict the Human Activity. And *Each datapoint corresponds one of the 6 Activities*.

## What is the best performance metric for this Problem?
- **Accuracy** : For any model we have printed the over all accuracy with this simple   “Accuracy” metric.

- **Confusion Matrix** : The very important thing that the confusion Matrix had told us what type of errors and what types of confusion are happening. 
    * Simply for understanding this metric for this project view, we know that we have 6 class label and often times it could so happen that our Model will be confused between sitting or standing, and walking upstairs or walking downstairs. 
    * So, the confusion Matrix is a very-very important way of understanding which class is your Algorithm or ML model is doing very well or for which classes your Algorithm or ML model is getting confused. 
![har1](https://user-images.githubusercontent.com/49862149/91036384-de3d6c80-e624-11ea-8ee5-182bf904df35.png)

        We can see clearly in this confusion matrix plot our model is doing very well for class Laying and Walking and good for Standing, Walking_Downstairs and Walking_Upstairs but our model getting confused with Sitting Class. 

- **Multi-class log-loss**: We know that Multi-class log-loss is very important metric for multiclass ML problem. 

## Business Objectives and Constraints
- These days, in addition to Smartphones, we are also using Smart-Watches like Fitbit or Apple-Watch, which help us to track our health. They monitor our each activity throughout the day check how many calories we have burnt. How many hours have we slept. However, in addition to Accelerometer and Gyroscope, they also use Heart-Rate data to monitor our activity. Since, we only have Smartphone data so just by using Accelerometer and Gyroscope data we will monitor the activity of a person. This software can then be converted into an App which can be downloaded in Smartphone. Hence, a person who has Smartphone can monitor his/her health using this App.

- **The cost of a mis-classification can be very high**.

- **No strict latency concerns**.
