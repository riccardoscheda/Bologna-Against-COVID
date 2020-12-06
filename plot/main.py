import datetime
from os.path import dirname, join

import pandas as pd
from scipy.signal import savgol_filter

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, DataRange1d, Select,HoverTool, HBar
from bokeh.palettes import GnBu3, OrRd3
from bokeh.palettes import Blues4
from bokeh.plotting import figure
import numpy as np
from datetime import date
from bokeh.palettes import mpl, d3,brewer
from bokeh.io import output_file, show
from bokeh.models import (BasicTicker, ColorBar, ColumnDataSource,
                          LinearColorMapper, PrintfTickFormatter)
from bokeh.plotting import figure
from bokeh.sampledata.unemployment1948 import data
from bokeh.transform import transform

interventions = ['C1_School closing',
               'C2_Workplace closing',
               'C3_Cancel public events',
               'C4_Restrictions on gatherings',
               'C5_Close public transport',
               'C6_Stay at home requirements',
               'C7_Restrictions on internal movement',
               'C8_International travel controls',
               'H1_Public information campaigns',
               'H2_Testing policy',
               'H3_Contact tracing',
               'H6_Facial Coverings']

DATA_URL = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv'
#df = pd.read_csv(DATA_URL,
#		  parse_dates=['Date'],
#		 encoding="ISO-8859-1",
#		 dtype={"RegionName": str,
#		        "RegionCode": str},
#		 error_bad_lines=False)
#df.to_csv("plot/data/data.csv")


def get_dataset(src, name):
	df = pd.read_csv("plot/data/data.csv",
			  parse_dates=['Date'],
			 encoding="ISO-8859-1",
			 dtype={"RegionName": str,
			        "RegionCode": str},
			 error_bad_lines=False)
	df["DailyChangeConfirmedCases"] = df.groupby(["CountryName"]).ConfirmedCases.diff().fillna(0)

	pred = pd.read_csv("plot/data/predictions.csv", parse_dates=['Date'],
			 encoding="ISO-8859-1",
			 dtype={"RegionName": str,
			        "RegionCode": str},
			 error_bad_lines=False)
	df = pd.concat([pred, df])
	df["RegionName"] = df["RegionName"].fillna("--")
	df = df[(df["CountryName"]==select.value) & (df["RegionName"] == region.value)]
	df["Date2"] = df["Date"].astype(str)
	df[interventions].fillna(0)
	return ColumnDataSource(data=df)

def make_plot(source,df, title, title2):
    plot = figure(x_axis_type="datetime", plot_width=1500, tools="", toolbar_location="above")
    plot.title.text = title

    #plot.line("Date","ConfirmedCases",source=source)
    plot.line("Date","PredictedDailyNewCases",source=source,line_width=3,color="orange")
    plot.line("Date","DailyChangeConfirmedCases",source=source,line_width=3)
    #plot.line(x=[current_time,current_time], y=[-10,50000], color="#FB8072", line_width=4, line_alpha =0.6, line_dash="dashed")

    ################### INTERVENTIONS #########################

    colors = list(brewer['RdYlGn'][4])
    colors[1] = '#FDE724'
    colors[2] = '#FBA40A'
    mapper = LinearColorMapper(palette=colors, low=df["C1_School closing"].min(), high=df["C2_Workplace closing"].max())

    graph = figure(plot_width=1500, plot_height=600, title="Interventions",
           x_range=dates, y_range=interventions,
           toolbar_location=None, tools="", x_axis_location=None)

    for i,j in enumerate(interventions):
      graph.rect(x="Date2", y=i, width=1.01, height=0.8, source=source,
       line_color = transform(j, mapper),fill_color=transform(j, mapper))

    color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                     ticker=BasicTicker(desired_num_ticks=len(colors)))

    graph.add_layout(color_bar, 'right')

    graph.axis.axis_line_color = None
    graph.axis.major_tick_line_color = None
    graph.axis.major_label_text_font_size = "10px"
    graph.axis.major_label_standoff = 0
    #graph.xaxis.major_label_orientation = 1.2
    graph.outline_line_color = None
    graph.xgrid.grid_line_color = None
    #graph.line(x=[df["Date2"].iloc[-2],df["Date2"].iloc[-2]], y=["C1_School closing", "H6_Facial Coverings"], color="#FB8072", line_width=4, line_alpha =0.9, line_dash="dashed")
    graph.title.text = title2

    return plot,graph

def update_plot(attrname, old, new):

    plot.title.text = "Daily Cases for " + select.value
    graph.title.text = "Interventions for " + select.value

    src = get_dataset(df[(df["CountryName"]==select.value) & (df["RegionName"] == region.value)]['ConfirmedCases'], df[(df["CountryName"]==select.value) & (df["RegionName"] == region.value)]['ConfirmedCases'])
    source.data.update(src.data)


city = 'Italy'
region = "--"

df = pd.read_csv("plot/data/data.csv",
	          parse_dates=['Date'],
			 encoding="ISO-8859-1",
			 dtype={"RegionName": str,
			        "RegionCode": str},
			 error_bad_lines=False)

dates = list(df["Date"].astype(str).unique())
options = list(np.unique(df["CountryName"]))
select = Select(value="Italy", title='Country', options=options)
df["RegionName"] = df["RegionName"].fillna("--")
regions = df["RegionName"].unique()
df["Date2"] = df["Date"].astype(str)
from datetime import datetime
now = datetime.now()
#current_time = now.strftime("%y:%m:%d")
current_time = df["Date"].iloc[-1]
region = Select(value=region,title="Region",options=list(regions))
source = get_dataset(df[(df["CountryName"]== select.value) & (df["RegionName"]==region.value)],select.value)
region.on_change('value',update_plot)
select.on_change('value', update_plot)
plot, graph = make_plot(source,df, "Daily Cases for " + select.value, "Interventions for " + select.value)
plot.add_tools(HoverTool(tooltips=[("Confirmed Cases:", "@ConfirmedCases"),
				("Predicted Cases:", "@PredictedDailyNewCases"),
				("Date","@Date{%F}")],
				formatters={'@Date': 'datetime'}))

graph.add_tools(HoverTool(tooltips=[("Date:","@Date2")]))

controls = column(select,region)
graphs = column(plot,graph)
curdoc().add_root(row(controls,graphs))
curdoc().title = "Covid"
