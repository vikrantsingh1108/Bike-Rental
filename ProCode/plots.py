import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import math

class plots:

	def __init__(self):
		#stores line plot objects
		self.Dline_plots = {}
		#stores hist_plot objects
		self.Dhist_Plot = {}
		#stores bar plot objects
		self.Dbar_Plot= {}
		#stores boxPlot object
		self.DBox_Plot = {}
		#dist plot  objects
		self.Ddist_plot = {}
		#heat map objects storage
		self.Dheatmap_plot = {}
		#cookie reference
		self.plot_cookie_ref = {}
		#plot counter
		self.plot_counter = 0
		

	def line_plot(self, Dplot_specs):
		fig, ax = plt.subplots(figsize=Dplot_specs["figsize"])
		self.Dline_plots[Dplot_specs["title"]] = sns.pointplot(data=Dplot_specs["DataCols"] ,
							x=Dplot_specs["x_axis"], y=Dplot_specs["y_axis"], hue=Dplot_specs["hue"], ax=ax)
		ax.set(title=Dplot_specs["title"]) #"Bike usage during weekdays and weekends"
		return self.updatePlotCounter("line",Dplot_specs["title"]) # return a cookie referebce for the line plot


	def hist_plot(self,Dplot_specs):
		fig, ax = plt.subplots(figsize=Dplot_specs["figsize"])
		self.Dhist_Plot[Dplot_specs["title"]] = sns.barplot(data=Dplot_specs["DataCols"],
						x=Dplot_specs['x_axis'], y=Dplot_specs['y_axis'])
		ax.set(title=Dplot_specs["title"])
		return self.updatePlotCounter("hist",Dplot_specs["title"]) # return a cookie referebce for the hist plot



	def barPlot(self,Dplot_specs):
		X_data = Dplot_specs["data"].shape[1]
		plt.figure()
		plt.title(Dplot_specs["title"])
		plt.bar(range(X_data), Dplot_specs["feature_ranked"][Dplot_specs["ranked_index"]],
											color=Dplot_specs["color"], align=Dplot_specs["align"])
		plt.xticks(range(X_data), Dplot_specs["ranked_index"])
		plt.xlim([-1, X_data])
		self.Dbar_Plot[Dplot_specs["title"]] = plt
		return self.updatePlotCounter("bar",Dplot_specs["title"]) # return a cookie referebce for the bar plot


	def box_plot(self,Dplot_specs):

		data = Dplot_specs["data"]

		if Dplot_specs["x_axis"] == None:
			Y = data[Dplot_specs["y_axis"]]
			self.DBox_Plot[Dplot_specs["title"]] = sns.boxplot(y=Y)
		else:
			X , Y = data[Dplot_specs["x_axis"]] , data[Dplot_specs["y_axis"]]
			self.DBox_Plot[Dplot_specs["title"]] = sns.boxplot(x= X , y=Y)

		return self.updatePlotCounter("box",Dplot_specs["title"]) # return a cookie reference to box plot


	def dist_plot(self,Dplot_specs):
		
		self.Ddist_plot[Dplot_specs["title"]] =  sns.distplot(Dplot_specs["DataCols"])
		return self.updatePlotCounter("dist",Dplot_specs["title"]) # return cookie reference to dist plot


	def heat_map(self, Dplot_specs):
		data = Dplot_specs["data"]
		heat =  np.array(data)
		heat[np.tril_indices_from(heat)]= False
		fig, ax= plt.subplots()
		fig.set_size_inches(Dplot_specs["figsize"])
		sns.set(font_scale=1.0)
		self.Dheatmap_plot =  sns.heatmap(data, mask=heat, vmax=1.0, vmin =0.0, square=True, annot=True, cmap ='Reds')
		return self.updatePlotCounter("heatMap",Dplot_specs["title"]) # return a cookie reference to heatMapplot



	def updatePlotCounter(self,plotType,plotTitle):
		self.plot_counter =  self.plot_counter + 1 
		self.plot_cookie_ref[self.plot_counter] =  [plotType,plotTitle]
		return self.plot_counter


	def view_plot(self,RefCookie):
	
		#print (self.plot_cookie_ref[(RefCookie)])

		if RefCookie > self.plot_counter or RefCookie <= 0 :
			return 0

		plotSpec = self.plot_cookie_ref[RefCookie]
		
		
		plotType = plotSpec[0]
		plotTitle = plotSpec[1]
		
		
		if plotType ==  "line":
			plot = self.Dline_plots[plotTitle]

		elif plotType ==  "hist":

			plot =  self.Dhist_Plot[plotTitle]

		elif plotType == "bar":
			
			plot = self.Dbar_Plot[plotTitle]
		
		elif plotType == "box":

			plot = self.DBox_Plot[plotTitle]
		
		elif plotType ==  "dist":

			plot = self.Ddist_plot[plotTitle]

		elif plotType == "heatMap":

			plot = self.Dheatmap_plot[plotTitle]
			
		plt.show()
		return plot