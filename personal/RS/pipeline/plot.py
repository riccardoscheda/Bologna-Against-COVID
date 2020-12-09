import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, CustomJSFilter, CDSView, Select, IndexFilter
from bokeh.io import show, output_notebook, save
from bokeh.layouts import column
import numpy as np
from datetime import timedelta
def covid_plot(cases_file, preds_file):
	df = pd.read_csv(cases_file,
			  parse_dates=['Date'],
			 encoding="ISO-8859-1",
			 dtype={"RegionName": str,
			        "RegionCode": str},
			 error_bad_lines=False)
	pred_df =  pd.read_csv(preds_file,
			  parse_dates=['Date'],
			 encoding="ISO-8859-1",
			 dtype={"RegionName": str,
			        "RegionCode": str},
			 error_bad_lines=False)


	default = "--"
	df["RegionName"] = df["RegionName"].fillna(default)
	df["DailyChangedConfirmedCases"] = df.groupby(["CountryName","RegionName"]).ConfirmedCases.diff().fillna(0)
	pred_df["RegionName"] = pred_df["RegionName"].fillna(default)

	source = ColumnDataSource(df)
	source2 = ColumnDataSource(pred_df)
	countries = sorted(list(set(source2.data['CountryName'])))
	regions = sorted(list(set(source2.data['RegionName'])))
	ita_indeces = list(df[(df["CountryName"]=="Italy") & (df["RegionName"]== regions[0])].index)
	ita_indeces2 = list(pred_df[(pred_df["CountryName"]=="Italy") & (pred_df["RegionName"]== regions[0])].index)

	filter = IndexFilter(ita_indeces)
	filter2 = IndexFilter(ita_indeces2)
	view = CDSView(source=source, filters=[filter])
	view2=CDSView(source=source2,filters=[filter2])

	country_select = Select(title='Country Selection', value="Italy", options=countries)
	region_select = Select(title='Region Selection', value= regions[0], options=regions)
	plot = figure(x_axis_type="datetime",title="Daily Cases",plot_width=1500,toolbar_location="above")

	plot.vbar('Date', top='DailyChangedConfirmedCases', width=timedelta(days=1),source=source,view=view,fill_color="#b3de69",color="green")
	plot.vbar('Date', top='NewCases', width=timedelta(days=1),source=source2,view=view2,fill_color="orange",color="orange",alpha=0.9)
	callback = CustomJS(args=dict(source=source,source2=source2,country_select=country_select,region_select=region_select,filter=filter,filter2=filter2), code='''
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
	         if (source2.data['CountryName'][i] == country_select.value && source2.data['RegionName'][i] == region_select.value) {
	           indices2.push(i)
	         }
	       }
	       filter2.indices = indices2
	 	  source2.change.emit();

	                    '''
	                )

	country_select.js_on_change('value', callback)
	region_select.js_on_change('value',callback)
	save(column(country_select,column(region_select, plot)),filename="plot.html")
