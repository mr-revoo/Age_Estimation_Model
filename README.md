# NUMERICAL CODE EXPALINTION
## Dataset Selection / Importing libraries
![image](https://github.com/user-attachments/assets/93d6f630-bffc-4e6a-ae85-455f762079da)
The code imports libraries like pandas and scikit-learn, loads data from a CSV file, and prepares data by selecting features and separating numerical and categorical variables. It likely involves splitting data, preprocessing, training models (e.g., linear regression, KNN), and evaluating their performance.
 ![image](https://github.com/user-attachments/assets/a99691d0-27d2-4dc8-abec-8b258f7df630)

## Dataset information
![image](https://github.com/user-attachments/assets/e72af51c-541a-45d5-b186-32f1ea51f524)

The code df.info() is used to display a concise summary of a DataFrame, including column names, data types, non-null values, and memory usage.
![image](https://github.com/user-attachments/assets/122c32b6-4aff-44f1-bac5-5638872430c3)

## Detecting Null Values
![image](https://github.com/user-attachments/assets/c7fa7c0f-c7ed-4e4c-8487-3117da9bac7a)
## Handle Null Values
![image](https://github.com/user-attachments/assets/65728d16-5d39-4823-8901-6e8c477b2cc6)

## Handling Outliers
 ![image](https://github.com/user-attachments/assets/068ca6a3-3346-4bbd-950b-af2a804174b6)
The code defines a function remove_outliers_iqr to remove outliers from a DataFrame using the Interquartile Range (IQR) method. It calculates Q1, Q3, and IQR for each numerical column, determines lower and upper bounds, and filters the DataFrame to exclude values outside these bounds. Finally, it applies this function to specific columns in the DataFrame and prints the new shape.

## Scale Numerical Columns and Encode Categorical Columns
![image](https://github.com/user-attachments/assets/694cf3fc-91f0-4c07-9343-accb4701656d)
The code scales numerical features using StandardScaler and encodes categorical features using LabelEncoder. It then displays the first 20 rows of the preprocessed DataFrame.
![image](https://github.com/user-attachments/assets/c010fd95-e091-4d4b-bd87-41af0bf121bc)

## Split The Data into Training and Testing
![image](https://github.com/user-attachments/assets/f0d2507f-7be7-4b3d-9dd2-acb179456bf9)
The code snippet splits the DataFrame into features (X) and target variable (y). It then prints the shapes of X and y and splits the data into training and testing sets using train_test_split with a 0.3 test size and a random state of 0.
![image](https://github.com/user-attachments/assets/8bd09f18-f419-4e80-8c45-1cdad621d5e3)

## Getting the best number of Neighbors to train the knn model
 ![image](https://github.com/user-attachments/assets/a01cb57e-d567-48e0-9841-180d723a2e47)
The code sets up a grid search to find the best n_neighbors parameter for a KNeighborsRegressor model. It defines a parameter grid with values from 1 to 20, creates a GridSearchCV object, fits it to the training data, and prints the best n_neighbors value found.
![image](https://github.com/user-attachments/assets/5a473c30-7fc0-4931-8e81-4f7948abc784)

## Train the models
![image](https://github.com/user-attachments/assets/29d7c08b-4d17-44d2-a879-d41fcb42d919)
This code initializes a LinearRegression model and fits it to the training data (X_train, y_train), training the model to learn the relationship between the features and the target variable.

 ![image](https://github.com/user-attachments/assets/0e8c485b-75eb-4120-8ce0-e48797af727c)
This code snippet creates a KNeighborsRegressor model with 11 neighbors and fits it to the training data. This trains the model to predict values based on the nearest neighbors in the training data

## Calculate The MSE, RMSE and R2 Score
 ### Linear Regression
![image](https://github.com/user-attachments/assets/5ad9e152-e8c7-4b41-8d3a-770f86e82705)
This code snippet calculates the Mean Squared Error (MSE) and R-squared (R2) scores for a linear regression model on a test set. It also calculates the Root Mean Squared Error (RMSE) and prints the results, along with the model's training and test accuracies.
 ![image](https://github.com/user-attachments/assets/57f09ad5-1050-4428-a834-fa7eb77abc79)

 ### KNN Regression
![image](https://github.com/user-attachments/assets/0b81dc57-a9f6-4106-ad06-a3e09639d386)
This code calculates evaluation metrics for a K-Nearest Neighbors (KNN) regression model:
1.	It predicts values on the test set using the trained KNN model.
2.	It computes the Mean Squared Error (MSE), R-squared (R2), and Root Mean Squared Error (RMSE) between the true and predicted values.
3.	It prints the calculated metrics for the KNN model.
4.	It also prints the training and testing accuracies of the model.
![image](https://github.com/user-attachments/assets/7b653152-8687-4662-bcc9-6a88d44b5b59)


## Machine Learning Project
---
#### Team Members:
- Refaat Mohamed
- Ali Gomaa
- Sara Ashraf
- Ahmed Maghawry
- Omar Ahmed 
