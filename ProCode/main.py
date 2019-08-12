# This is main file 
import pandas as pd
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dataSetSpecifics import *
from Models import *
from plots import *
import pickle

class dataSet_reader():

	def __init__(self,dataSet_path= 'hour.csv'):
		self.raw_data = 0
		self.feature_data = ['season', 'mnth', 'hr', 'holiday', 'weekday', 
								'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
		self.target =  ['cnt']
		self.dataSet_path = dataSet_path
		self.plot = plots()
		self.load_dataSet()

		self.classifierModels = AbstractModelClass.populate_classifier_models()
		self.classifier = {}
			
		# Load the data
	def load_dataSet(self):
		self.raw_data = pd.read_csv(self.dataSet_path)
		#renaming columns 
	def data_rename_columns(self,data, renameList):
		data.rename(columns=renameList ,inplace=True)

	def data_displayHead(self,data,n_cols):
		return data.head(n_cols)

	def typeCaseConversion(self,data,DtypeConv):
		return  data.astype(DtypeConv)

	def data_typeDescribe(self,data):
		return data.dtypes

	def data_tabular_describe(self,data):
		return data.describe()

	def data_ColumnDrop(self,data, col):
		return data.drop(col,axis=1)

	def data_extractFeaturesTarget(self,data,tar_name):
		target = data[tar_name]
		feature_data = self.data_ColumnDrop(data,tar_name)
		return feature_data,target

	def data_setFeature_And_Target(self,data,tar_name):
		self.feature_data , self.target  = self.data_extractFeaturesTarget(data,tar_name)

	def data_isNullAny(self,data):
		return data.isnull().any()

	def data_logTransformation(self,data):
		return data.transform(lambda x: math.log(x))

	def getDataShape(self , data):
		return data.shape

	def addClassifier(self, classification_model ,classifier):
		self.classifier[classification_model]= classifier
		

	def generate_dummies(self , data, dummy_column):
		dummies = pd.get_dummies(data[dummy_column], prefix=dummy_column)
		data = pd.concat([data, dummies], axis=1)
		return data

def packPlotDict(**kwargs):
	Dplot_specs = {}
	Dplot_specs["figsize"] = kwargs["figsize"] if "figsize" in kwargs.keys() else None
	Dplot_specs["data"] = kwargs["data"] if "data" in kwargs.keys() else None
	Dplot_specs["DataCols"] = (kwargs["data"])[kwargs["DataCols"]] if "DataCols" in kwargs.keys() else None 
	Dplot_specs["x_axis"] = kwargs["x_axis"] if "x_axis" in kwargs.keys() else None
	Dplot_specs["y_axis"] = kwargs["y_axis"] if "y_axis" in kwargs.keys() else None 
	Dplot_specs["hue"] = kwargs["hue"] if "hue" in kwargs.keys() else None
	Dplot_specs["title"] = kwargs["title"] 
	Dplot_specs["feature_ranked"] = kwargs["feature_ranked"] if "feature_ranked" in kwargs.keys() else None
	Dplot_specs["ranked_index"] = kwargs["ranked_index"] if "ranked_index" in kwargs.keys() else None
	Dplot_specs["color"] =  kwargs["color"] if "color" in  kwargs.keys() else None
	Dplot_specs["align"] =  kwargs["align"] if "align" in kwargs.keys() else None
	return Dplot_specs

def setUpDataSetObj():
	DataSetObj = dataSet_reader(C_DATASET_PATH)
	DataSetObj.data_rename_columns(DataSetObj.raw_data , DcolRename)
	DataSetObj.raw_data = DataSetObj.data_ColumnDrop(DataSetObj.raw_data , LcolDrop)
	DataSetObj.raw_data = DataSetObj.typeCaseConversion(DataSetObj.raw_data, DtypeCaseConv)
	return DataSetObj

def visualizeData(DataSetObj):
	#line plots analysing usage
	#These plots represents the usage of bikes during normal days, different weather conditions
	Dplot_specs = packPlotDict(figsize = (20,10),data = DataSetObj.raw_data, DataCols = ['hour','count','weekday'],
								x_axis = "hour", y_axis = "count", hue = "weekday",title = "Use of the system: Weather condition")
	ptr = DataSetObj.plot.line_plot(Dplot_specs)

	Dplot_specs = packPlotDict(figsize = (20,10), data = DataSetObj.raw_data, DataCols = ['hour','count','weather'], 
								x_axis = "hour", y_axis = "count",hue = "weather", title ="Bike usage during different weather condition")
	ptr1 = DataSetObj.plot.line_plot(Dplot_specs)

	Dplot_specs = packPlotDict(figsize = (20,10),data = DataSetObj.raw_data,DataCols =['hour','count','season'],
								x_axis = "hour" ,y_axis = "count",hue = "season",title = "Bike usage during different season")
	ptr2 = DataSetObj.plot.line_plot(Dplot_specs)

	Dplot_specs = packPlotDict(figsize = (20,10),data = DataSetObj.raw_data,DataCols =['month','count'], 
								x_axis = "month" ,y_axis = "count", hue="none", title = "Monthly distribution")
	DataSetObj.plot.hist_plot(Dplot_specs)
	
	Dplot_specs = packPlotDict(figsize = (20,10),data = DataSetObj.raw_data,DataCols =['weekday','count'],
								 x_axis = "weekday" ,y_axis = "count", hue="none", title = "Weekly distribution")
	DataSetObj.plot.hist_plot(Dplot_specs)

def randaomForestClassification(DataSetObj, X_train, X_test, y_train, y_test):

	# The random forest classifier is used for feature selection
	RndFrstClassifier = DataSetObj.classifierModels["random_forestClassifier"](100)
	DataSetObj.addClassifier("random_forestClassifier",RndFrstClassifier)
	DataSetObj.classifier["random_forestClassifier"].fit(X_train,y_train)

	ranked_feature_indices , feature_ranked = DataSetObj.classifier["random_forestClassifier"].get_FeatureImportance()

def barBoxPlot_visualization(DataSetObj):
	# Box plots used to detecting outliers

	# Boxplot to plot count and other parameters
	Dplot_specs = packPlotDict(data = DataSetObj.raw_data , y_axis = "count" , 
								title = "Count Data Box Plot")
	DataSetObj.plot.box_plot(Dplot_specs)

	Dplot_specs = packPlotDict(data = DataSetObj.raw_data , x_axis = "season" , 
								y_axis ="count" ,title = "Season Count Data BoxPlot")
	DataSetObj.plot.box_plot(Dplot_specs)

	Dplot_specs = packPlotDict(data = DataSetObj.raw_data , x_axis = "hour" , 
								y_axis ="count" ,title = "hour Count Data BoxPlot")
	DataSetObj.plot.box_plot(Dplot_specs)

	Dplot_specs = packPlotDict(data = DataSetObj.raw_data , x_axis = "temp" , 
								y_axis ="count" ,title = "temperature Count Data BoxPlot")
	DataSetObj.plot.box_plot(Dplot_specs)

	Dplot_specs = packPlotDict(data = DataSetObj.raw_data , x_axis = "weather" , 
								y_axis ="count" ,title = "weather Count Data BoxPlot")
	DataSetObj.plot.box_plot(Dplot_specs)

	Dplot_specs = packPlotDict(data = DataSetObj.raw_data , x_axis = "workingday" , 
								y_axis ="count" ,title = "workingday Count Data BoxPlot")
	DataSetObj.plot.box_plot(Dplot_specs)

def distPlotVisualization(DataSetObj):

	Dplot_specs = packPlotDict(data = DataSetObj.raw_data , DataCols = "count" ,
								title = " Count Data Dist Plot")
	DataSetObj.plot.dist_plot(Dplot_specs)
	
	tData = DataSetObj.data_logTransformation(DataSetObj.raw_data["count"])
	Dplot_specs = packPlotDict(title = " Count Normalized Data Dist Plot")
	Dplot_specs["DataCols"] = tData
	DataSetObj.plot.dist_plot(Dplot_specs)

def heatMapVisualization(DataSetObj):
	Dplot_specs =  packPlotDict(data = DataSetObj.raw_data.corr(),figsize=(20,10),
									font_scale=1.0,color = 'Reds', title = "heat map")
	DataSetObj.plot.heat_map(Dplot_specs)

def one_hot_encoding(DataSetObj):
	
	data_dummy  = DataSetObj.data_ColumnDrop(DataSetObj.raw_data , ['atemp','casual','registered'])

	
	dummy_data = pd.DataFrame.copy(data_dummy)
	dummy_columns = ["season", "month", "hour", "holiday", "weekday",'workingday',"weather"]

	for dummy_column in dummy_columns:
		dummy_data = DataSetObj.generate_dummies(dummy_data,dummy_column)

	for dummy_column in dummy_columns:
		del dummy_data[dummy_column]

	y = dummy_data['count']
	X = dummy_data.drop(['count'],axis=1)
	return X,y
	
def linear_regressionModel(X_train, X_test, y_train, y_test):

	''' Linear Regression Model '''
	regressionModel =  DataSetObj.classifierModels["linear_regressionModel"]()
	DataSetObj.addClassifier("linear_regressionModel",regressionModel)

	regressionModel.fit(X_train,y_train)

	y_predicted = regressionModel.classifier_prediction(X_test)
	y_pred = y_predicted.ravel()
	print (y_pred)
	CompFrame = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
	return CompFrame

def random_forestClassifier(DataSetObj,X_train, X_test, y_train):

	''' Random Forest Regressor '''
	regressor = DataSetObj.classifierModels["random_forestRegressorClassifier"](n_estimators = 300)
	DataSetObj.addClassifier("random_forestRegressorClassifier",regressor)	

	DataSetObj.classifier["random_forestRegressorClassifier"].fit(X_train,y_train)

	DataSetObj.classifier["random_forestRegressorClassifier"].save_model(model_file='model.pth')

	# Predicting the values 
	y_pred = regressor.classifier_prediction(X_test)
	return regressor.classifier , y_pred



if __name__ == "__main__":

	DataSetObj = setUpDataSetObj()
	#Remove the comment from below functions for visualizing different plots

	#barBoxPlot_visualization(DataSetObj)
	#distPlotVisualization(DataSetObj)
	#heatMapVisualization(DataSetObj)
	#visualizeData(DataSetObj)

	X,y = one_hot_encoding(DataSetObj)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
	y_test = y_test.ravel()
	estimator , y_pred = random_forestClassifier(DataSetObj,X_train, X_test, y_train)
	
	mean_aberr = mean_absolute_error(y_test, y_pred)
	print('Mean absolute error: ',mean_aberr)
	accuracy = cross_val_score(estimator = estimator, X = X_train, y = y_train, cv =10)
	mean_acc = accuracy.mean()
	print('Mean accuracy: ',mean_acc)
	df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
	df.to_csv("submission.csv")
	RMLSE = np.sqrt(mean_squared_log_error(y_test, y_pred))
	print('Root mean squared log error: ', RMLSE)
	plt.show()