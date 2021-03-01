import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mlt

df = pd.read_pickle('data/test_data.pkl')

def preproccesing_data(df):
    df.dropna(inplace=True)
    df['Timestamps'] = pd.to_datetime(df['Timestamps'], utc=True, errors ='coerce')
    df.drop(857, inplace=True)
    df.set_index('Timestamps', inplace=True)
    df['day'] = df.index.day
    return df

df = preproccesing_data(df)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

line_plot_general = px.line(df, x=df.index.values, y=df['Values'], title='Line Plot de los 3 dias', color=df['day'],  labels = {'x':'Hours', 'y':'Values'})

#std = df['Values'].std()
#mean = df['Values'].mean()
#variance = df['Values'].var()

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Div([

        html.H1(children='Time Series Cliengo'),
        
        
        dcc.Markdown('Selecciona el dia'),
        dcc.Dropdown(
            id="dropdown",
            options=[{"label": day, "value": day} 
                    for day in df.index.day.unique()],
            value=25
        ),
        
        dcc.Graph(id="line-plot"),
        html.P("Hours:"),
        dcc.RangeSlider(
            id='range-slider',
            min=df.index.hour.min(), max=df.index.hour.max(), step=1,
            marks={str(hour): str(hour) for hour in df.index.hour.unique()},
            value=[4, 8]
        ),
        
    ],
    style={'width': '49%', 'display': 'inline-block'}),
    
    
    html.Div([
        
        html.H1(children='-------------------------------------------'),

        dcc.Markdown('Rango completo - 3 dias'),
        dcc.Dropdown(
            id="dropdown-graph",
            options=[{"label": graph, "value": graph} 
                    for graph in ('Line','Box')],
            value='Line'
        ),

        dcc.Graph(id="line-plot-2"),
        
        html.Div(id='std'),
        html.Div(id='mean'),
        html.Div(id='variance')

    ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}
    ), 

    html.Div([
        
        html.Hr(),
        dcc.Graph(id="moving-average-graph"),
        
        
        

    ], style={'width': '49%', 'float': 'left','display': 'inline-block'}),

    html.Div([
        
        html.Hr(),
        dcc.Graph(id="moving-std-graph"),
        
        

    ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})

])

@app.callback(
    Output("line-plot", "figure"), 
    Output('std', 'children'),
    Output('mean', 'children'),
    Output('variance', 'children'),
    Output('moving-average-graph', 'figure'),
    Output('moving-std-graph', 'figure'),
    Input("range-slider", "value"),
    Input("dropdown", "value"))
def update_line_chart(slider_range, dropdown_day):
    # global std, mean, variance
    low, high = slider_range
    x = df[(df.index.hour >= low) & (df.index.hour < high) & (df.index.day == dropdown_day)].index 
    y = df[(df.index.hour >= low) & (df.index.hour < high) & (df.index.day == dropdown_day)]['Values']

    std = y.std()
    mean = y.mean()
    variance = y.var()

    std = '   Standard deviation: ' + str(std)
    mean = '   Mean: ' + str(mean)
    variance = '   Variance: ' + str(variance)

    fig = px.line(x=x, y=y, title='Variaciones durante un rango de tiempo determinado', labels = {'x':'Hours', 'y':'Values'})
    
    ######## Fig 2 -- Moving Average

    fig2 = px.line(x=x, y=[y, y.rolling(window=10).mean(), y.rolling(window=30).mean()], labels = {'x':'Hours', 'y':'Values'}, title='Moving Average')

    def function_traces(trace):
        if trace.name == "wide_variable_0":
            trace.update(name="Normal")
        elif trace.name == 'wide_variable_1':
            trace.update(name='Moving Average 10')
        elif trace.name == 'wide_variable_2':
            trace.update(name='Moving Average 30')

    fig2.for_each_trace(function_traces)

    ######## Fig 3 -- Moving STD

    fig3 = px.line(x=x, y=[y, y.rolling(window=10).std(), y.rolling(window=30).std()], labels = {'x':'Hours', 'y':'Values'}, title='Moving STD')

    def function_traces(trace):
        if trace.name == "wide_variable_0":
            trace.update(name="Normal")
        elif trace.name == 'wide_variable_1':
            trace.update(name='Moving Std 10')
        elif trace.name == 'wide_variable_2':
            trace.update(name='Moving Std 30')

    fig3.for_each_trace(function_traces)
    
    return fig , std , mean , variance , fig2 , fig3

@app.callback(
    Output("line-plot-2", "figure"), 
    Input("dropdown-graph", "value")
   )
def update_graph(value):
    if value == 'Line':
        fig = px.line(df, x=df.index.values, y=df['Values'], color=df['day'],  labels = {'x':'Hours', 'y':'Values'})
        return fig
    else:
        fig = px.box(df, y='Values')
        return fig
    

if __name__ == '__main__':
    app.run_server(debug=True)