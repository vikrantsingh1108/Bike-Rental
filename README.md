# Bike-Rental
Implemented Regression model on Bike Rental sharing dataset

1. Business case
The prediction of Bikes sharing on hourly basis helps us understand the requirements of the
bike at a particular station. We can learn if the allotted bike at that station is enough for
example, if the bike returned at a particular station is more than the slot then there is a need to
build another station nearby. The data can help us understand the number of bikes being rented
from the different station and the prediction can help us to refill new bikes. The task focuses on
predicting the hourly usage of bike considering day hours, weather conditions and seasons.

2. Data Analysis
The data consists of 17379 rows and 17 columns of which some of them are dropped as it does
not contains important information and would not contribute much towards model prediction. In
the missing value analysis the data does not consist of any NULL or NAN values. The matrix
shows data with 14 features and the line plot represents the hourly pattern. The weekday and
weekend plot shows that the bike is rented more during office hours and the weather plot shows
the bikes are rented more in summer and fall and less during the winter and spring.

3. Model Selection
The data consists of both categorical and numerical features, the task is to further predict the
bikes count which can be categorized as a regression problem. The data is small and as few
features are significant. I have used Random forest Classifier for feature selection and Random
forest regressor and evaluated the model performances. The metrics used are ranking, mean
accuracy, mean absolute error and root mean square log error. The reason for using Random
Forest is that it uses decision trees, it can easily handle categorical and numerical data with littlepreprocessing. I have tried Logistic regression and random forest regression of which random
forest performed better on the Bike Sharing Dataset and hence I chose it for my final model.
The random forest model consists of 300 decision trees trained on the subsamples of the
dataset. To generalize and prevent overfitting I have used cross-validation and used mean to
improve the predictive accuracy.

5. Source file
The source files consists of ProCode folder and ipython notebook.
ProCode​ : The ProCode folder consists of python files and the dataset. The python file
does the same analysis as one can find in the ipython notebook. The objective is to display the
reusability of the code that can be maintained in the production environment. The analysis are
broken into functions and can be used for daily prediction task. It makes it easier to understand
and debug. The code is reusable and can be maintained and also can be extended by the
potential colleagues. It also consists of documentation, submission file and the unittest file.
Bike_Rental_Task.ipynb​ : This file consists of stepwise analysis such as descriptive
analysis, missing value analysis, feature importance, outliers analysis, model selection and
conclusion. For better visualization I would suggest using notebook. The notebook consists of
detailed explanation of each result.
Requirements:
IPython Notebook
Python 3.7, Python notebook 3
Sklearn, Pandas, NumPy, missingno, seaborn
To run the notebook use following command:
jupyter notebook
To run the python code use the following command:
python main.py
To run the unittest use the following command:
python -m unittest
