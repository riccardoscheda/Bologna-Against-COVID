import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, CustomJSFilter, CDSView, Select, IndexFilter
from bokeh.io import show, output_notebook, save
from bokeh.layouts import column
output_notebook()

df = pd.read_csv("data/data.csv",
		  parse_dates=['Date'],
		 encoding="ISO-8859-1",
		 dtype={"RegionName": str,
		        "RegionCode": str},
		 error_bad_lines=False)
pred_df =  pd.read_csv("data/predictions.csv",
		  parse_dates=['Date'],
		 encoding="ISO-8859-1",
		 dtype={"RegionName": str,
		        "RegionCode": str},
		 error_bad_lines=False)

df["DailyChangedConfirmedCases"] = df.groupby(["CountryName"]).ConfirmedCases.diff().fillna(0)
source = ColumnDataSource(df)
source2 = ColumnDataSource(pred_df)
countries = sorted(list(set(source2.data['CountryName'])))
ita_indeces = list(df[df["CountryName"]=="Italy"].index)
ita_indeces2 = list(pred_df[pred_df["CountryName"]=="Italy"].index)
filter = IndexFilter(ita_indeces)
filter2 = IndexFilter(ita_indeces2)
view = CDSView(source=source, filters=[filter])
view2=CDSView(source=source2,filters=[filter2])
plot = figure(x_axis_type="datetime",plot_width=1500, tools="", toolbar_location="above")

plot.vbar('Date', top='DailyChangedConfirmedCases', source=source, fill_color="#b3de69",view=view)
plot.vbar('Date', top='PredictedDailyNewCases', source=source2,view=view2,color="orange",alpha=0.5)
select = Select(title='Country Selection', value="Italy", options=countries)

callback = CustomJS(args=dict(source=source,source2=source2,select=select,filter=filter,filter2=filter2), code='''
     const indices = []
	 const indices2 = []
      for (var i = 0; i < source.get_length(); i++) {
        if (source.data['CountryName'][i] == select.value) {
          indices.push(i)
        }
      }
      filter.indices = indices;
	  source.change.emit();

	   for (var i = 0; i < source2.get_length(); i++) {
         if (source2.data['CountryName'][i] == select.value) {
           indices2.push(i)
         }
       }
       filter2.indices = indices2;
 	  source2.change.emit()

                    '''
                )

select.js_on_change('value', callback)

save(column(select, plot))
