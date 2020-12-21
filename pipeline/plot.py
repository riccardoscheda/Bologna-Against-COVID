import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, CDSView, Select, IndexFilter
from bokeh.io import save
from bokeh.layouts import column
from datetime import timedelta
from utils import create_dataset


def covid_plot(cases_file, preds_file):
	"""
	Makes an interactive plot with bokeh. It will generate an html file in which we can select the country, region, and model for the predictions
	of daily new cases.

	Parameters:
	------------

	cases_file : pandas DataFrame with historical cases
	preds_file : pandas DataFrame with the column "PredictedDailyNewCases"
	"""
	df = pd.read_csv(cases_file,
					parse_dates=['Date'],
					encoding="ISO-8859-1",
					dtype={"RegionName": str,"RegionCode": str},
			 		error_bad_lines=False)

	# reading the file with predictions of daily new cases
	pred_df =  pd.read_csv(preds_file,
							parse_dates=['Date'],
							encoding="ISO-8859-1",
							dtype={"RegionName": str,"RegionCode": str},
							error_bad_lines=False)


	# filling missing values of Region column to make easier the selection
	default = "--"
	df["RegionName"] = df["RegionName"].fillna(default)
	#df["DailyChangedConfirmedCases"] = df.groupby(["CountryName","RegionName"]).ConfirmedCases.diff().fillna(0)
	pred_df["RegionName"] = pred_df["RegionName"].fillna(default)

	df = create_dataset(df)
	# Bokeh needs as input the type of ColumnDataSource
	source = ColumnDataSource(df)
	source2 = ColumnDataSource(pred_df)
	# listing all the countries and all the regions and models to be put in the selections of the plot
	countries = sorted(list(set(source2.data['CountryName'])))
	regions = sorted(list(set(source2.data['RegionName'])))
	models = sorted(list(set(source2.data['Model'])))
	# default initial country plot when we open the html file
	ita_indeces = list(df[(df["CountryName"]==countries[0]) & (df["RegionName"]== regions[0])].index)
	ita_indeces2 = list(pred_df[(pred_df["CountryName"]==countries[0]) & (pred_df["RegionName"]== regions[0]) & (pred_df["Model"] == models[0])].index)

	# filter is used when we click a country in the selection, and it filters the columns with the name of the country,
	# the name of the region, etc
	filter = IndexFilter(ita_indeces)
	filter2 = IndexFilter(ita_indeces2)
	# view is an argument for the plot, and it allows to update the plot with the filters
	view = CDSView(source=source, filters=[filter])
	view2=CDSView(source=source2,filters=[filter2])

    # These are the three selections for the plot, country, region and model
	country_select = Select(title='Country Selection', value=countries[0], options=countries)
	region_select = Select(title='Region Selection', value= regions[0], options=regions)
	model_select = Select(title='Model', value= models[0], options=models)

	# creates the figure
	plot = figure(x_axis_type="datetime",title="Daily Cases",plot_width=1500,toolbar_location="above")
	# creates the histogram for historical real data of daily cases
	plot.vbar('Date', top='NewCases', width=timedelta(days=1),source=source,view=view,fill_color="#b3de69",color="green")

	#plot.vbar('Date', top='MA', width=timedelta(days=1),source=source2,view=view2,fill_color="red",color="red",alpha=0.5)

	# creates the histogram of the moving average of real data
	plot.vbar('Date', top='MA', width=timedelta(days=1),source=source,view=view,fill_color="blue",color="blue",alpha=0.5)

	# creates the histogram of the predictions
	plot.vbar('Date', top='PredictedDailyNewCases', width=timedelta(days=1),source=source2,view=view2,fill_color="orange",color="orange",alpha=0.8)

	# to create an interactive html, bokeh need a javascript code in order to update the plots based on the filters
	callback = CustomJS(args=dict(source=source,source2=source2,country_select=country_select,region_select=region_select,model_select=model_select,filter=filter,filter2=filter2), code='''
	     const indices = []
		 const indices2 = []

	      for (var i = 0; i < source.get_length(); i++) {
	        if (source.data['CountryName'][i] == country_select.value && source.data['RegionName'][i] == region_select.value) {
	          indices.push(i)
	        }
	      }
	      filter.indices = indices
		  source.change.emit()


		   for (var i = 0; i < source2.get_length(); i++) {
	         if (source2.data['CountryName'][i] == country_select.value && source2.data['RegionName'][i] == region_select.value && source2.data["Model"][i] == model_select.value) {
	           indices2.push(i)
	         }
	       }
	       filter2.indices = indices2
	 	  source2.change.emit();


	                    '''
	                )

	# updateing the plot based on the selection of country region and model
	country_select.js_on_change('value', callback)
	region_select.js_on_change('value',callback)
	model_select.js_on_change('value',callback)
	# saves the interactive html
	save(column(model_select,column(country_select,column(region_select, plot))),"plot.html")
