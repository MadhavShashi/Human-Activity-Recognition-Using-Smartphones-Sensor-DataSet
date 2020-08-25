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

## Data Overview
### 1. How data was recorded
-  30 participants (*referred as subjects in this dataset*) performed activities of daily living while *carrying a waist-mounted smartphone*. The phone was configured to record two implemented sensors (**accelerometer and gyroscope**). For these time series the directors of the underlying study performed feature generation and generated the dataset by moving a **fixed-width window of 2.56s** over the series. Since the **windows had 50% overlap** the resulting points are **equally spaced (1.28s)**.This experiment was video recorded to label the data manually.

-  By using the sensors(**Gyroscope and accelerometer**) in a smartphone, they have captured '**3-axial linear acceleration'(_tAcc-XYZ_) from accelerometer and '3-axial angular velocity' (_tGyro-XYZ_) from Gyroscope** with several variations.
   *  prefix '**t**' in those metrics denotes time.
   *  suffix '**XYZ**' represents 3-axial signals in **X** , **Y**, and **Z** directions.
   *  Let’s understand above information in graphical way below:

      ![npic1](https://user-images.githubusercontent.com/49862149/91124530-09739a80-e6bd-11ea-9bb9-d4e4b437f819.jpg)

### 2. How is the Data preprocessed?
-  After getting Raw Sensor Data the Expert (**Domain Expert, Signal Engineer Expert**) are preprocessed this data and make some useful feature. I am not expert but what I understand I explain here how these data are preprocessed. 
-  These sensor signals are preprocessed by applying **noise filters** and then *sampled in fixed-width windows (sliding windows) of 2.56 seconds each with 50% overlap*. ie., each window has *128* readings.
-  From Each window, a feature vector was obtained by calculating variables from the **time and frequency domain**.
   ![npic2](https://user-images.githubusercontent.com/49862149/91124992-37a5aa00-e6be-11ea-87c6-43acd7ac8b04.jpg)
   
-  The accelertion signal was saperated into Body and Gravity acceleration signals(**tBodyAcc-XYZ** and **tGravityAcc-XYZ**) using some low pass filter with corner frequecy of 0.3Hz.
-  After that, the body linear acceleration and angular velocity were derived in time to obtian *jerk signals* (**tBodyAccJerk-XYZ** and **tBodyGyroJerk-XYZ**).
-  The magnitude of these 3-dimensional signals were calculated using the Euclidian norm. This magnitudes are represented as features with names like **tBodyAccMag_, _tGravityAccMag_, _tBodyAccJerkMag_, _tBodyGyroMag and tBodyGyroJerkMag.**
-  Finally, We've got frequency domain signals from some of the available signals by applying a FFT (**Fast Fourier Transform**). These signals obtained were labeled with **prefix 'f'** just like original signals with **prefix 't'**. These signals are labeled as **fBodyAcc-XYZ, fBodyGyroMag** etc.,.

-  These are the signals that we got so far.
   *  tBodyAcc-XYZ
   *  tGravityAcc-XYZ
   *  tBodyAccJerk-XYZ
   *  tBodyGyro-XYZ
   *  tBodyGyroJerk-XYZ
   *  tBodyAccMag
   *  tGravityAccMag
   *  tBodyAccJerkMag
   *  tBodyGyroMag
   *  tBodyGyroJerkMag
   *  fBodyAcc-XYZ
   *  fBodyAccJerk-XYZ
   *  fBodyGyro-XYZ
   *  fBodyAccMag
   *  fBodyAccJerkMag
   *  fBodyGyroMag
   *  fBodyGyroJerkMag

-  We can esitmate some set of variables from the above signals. ie., We will estimate the following properties on each and every signal that we recoreded so far.
-  For better remember : we can see above image, EXPERTS apply some filter on each window and get 1st vector, 2nd vector and…….. so on. On top of these vector they computed below listed function.
   *  **mean()**: Mean value
   *  **std()**: Standard deviation
   *  **mad()**: Median absolute deviation
   *  **max()**: Largest value in array
   *  **min()**: Smallest value in array
   *  **sma()**: Signal magnitude area
   *  **energy()**: Energy measure. Sum of the squares divided by the number of values.
   *  **iqr()**: Interquartile range
   *  **entropy()**: Signal entropy
   *  **arCoeff()**: Autorregresion coefficients with Burg order equal to 4
   *  **correlation()**: correlation coefficient between two signals
   *  **maxInds()**: index of the frequency component with largest magnitude
   *  **meanFreq()**: Weighted average of the frequency components to obtain a mean frequency
   *  **skewness()**: skewness of the frequency domain signal
   *  **kurtosis()**: kurtosis of the frequency domain signal
   *  **bandsEnergy()**: Energy of a frequency interval within the 64 bins of the FFT of each window.
   *  **angle()**: Angle between to vectors.
   
-  We can obtain some other vectors by taking the average of signals in a single window sample. These are used on the angle() variable' `
   *  gravityMean
   *  tBodyAccMean
   *  tBodyAccJerkMean
   *  tBodyGyroMean
   *  tBodyGyroJerkMean
   
### 3. Y_Labels(Encoded)
-  In the dataset, Y_labels are represented as numbers from 1 to 6 as their identifiers.
   *  WALKING as __1__
   *  WALKING_UPSTAIRS as __2__
   *  WALKING_DOWNSTAIRS as __3__
   *  SITTING as __4__
   *  STANDING as __5__
   *  LAYING as __6__
   
### 4. Data Directory
         *  ![npic03](https://user-images.githubusercontent.com/49862149/91126397-7c7f1000-e6c1-11ea-94e9-c909b34df502.jpg)

-  ![#FF5733](https://via.placeholder.com/7x24/FF5733/000000?text=+) __Important Note__: When I am applying Machine learning algorithm, I use these experts created feature data. When we are applying Deep learning algorithm, I use RAW sensors DATA for predicting Human Activity.
   ![npic4](https://user-images.githubusercontent.com/49862149/91137130-25326d00-e6cc-11ea-99a0-1cee55d314c0.jpg)
 
-  The data is provided as a single zip file that is about **58 megabytes** in size. The direct link for this download is below: [**UCI HAR Dataset.zip**](https://archive.ics.uci.edu/ml/machine-learning-databases/00364/dataset_uci.zip)
   
   







