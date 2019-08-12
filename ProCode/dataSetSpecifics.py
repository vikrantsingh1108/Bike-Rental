
#Cxyz - C indicates a constant name 
C_DATASET_PATH = '/Add data path/hour.csv'

C_PLOT_SIZE = "plot_size"
SUB_PLOT_DIM = "subplot_Dim"

#Dxyz - D indicates a dictType variable
DcolRename = {'weathersit':'weather','mnth':'month','hr':'hour','hum':'humidity','cnt':'count'}

#Lxyz - L indicates a ListType variable
LcolDrop = ['instant','dteday','yr']

DtypeCaseConv = {'season':'category', 'month':'category' , 'hour' : 'category' , 'holiday':'category' , 'weekday':'category' , 'workingday':'category' , 'weather':'category'}

hours_data=['hour','count','weekday']
