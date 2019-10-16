import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.externals import joblib
import plotly.graph_objs as go

import feature_engineering as fe
import neural_network as nn
import random_forest as rf
from keras.models import load_model

FILEPATH_X = "multilabel/X.csv"
FILEPATH_Y = "multilabel/Y_Type.csv"
TEST_SIZE = 0.05

app = dash.Dash()

#training_data = joblib.load("./training_data.pkl")
#training_labels = joblib.load("./training_labels.pkl")

app.layout = html.Div(children=[
    html.H1(children='NFL Play Predictor', style={'textAlign': 'center'}),

    html.Div(children=[
        html.Label('Enter years of experience: '),
        dcc.Input(id='quarter', placeholder='Quarter', type='text'),
        dcc.Input(id='minute', placeholder='Minute', type='text'),
        dcc.Input(id='second', placeholder='Second', type='text'),
        dcc.Input(id='down', placeholder='Down', type='text'),
        dcc.Input(id='togo', placeholder='ToGo', type='text'),
        dcc.Input(id='yardline', placeholder='YardLine', type='text'),
        dcc.Input(id='formation', placeholder='Under Center, Shotgun, or No Huddle', type='text'),
        dcc.Input(id='yardlinedirection', placeholder='OWN or OPP', type='text'),
        dcc.Input(id='offenseteam', placeholder='Off', type='text'),
        dcc.Input(id='defenseteam', placeholder='Def', type='text'),
        html.Div(id='result')
    ], style={'textAlign': 'center'}),

    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                go.Scatter(
                    x=[0, 1, 2, 3],
                    y=[5, 5, 5, 2],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                )
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'Years of Experience'},
                yaxis={'title': 'Salary'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                hovermode='closest'
            )
        }
    )
])


@app.callback(
    Output(component_id='result', component_property='children'),
    [Input(component_id='quarter', component_property='value')],
    [Input(component_id='minute', component_property='value')],
    [Input(component_id='second', component_property='value')],
    [Input(component_id='down', component_property='value')],
    [Input(component_id='togo', component_property='value')],
    [Input(component_id='yardline', component_property='value')],
    [Input(component_id='formation', component_property='value')],
    [Input(component_id='yardlinedirection', component_property='value')],
    [Input(component_id='offenseteam', component_property='value')],
    [Input(component_id='defenseteam', component_property='value')],
    )
def update_years_of_experience_input(quarter,
									 minute,
									 second,
									 down,
									 togo,
									 yardline,
									 formation,
									 yardlinedirection,
									 offenseteam,
									 defenseteam):
    
	



    if years_of_experience is not None and years_of_experience is not '':
        try:
            salary = nn.predict(model, float(years_of_experience))
            return salary
        except ValueError:
            return 'Unable to run'


if __name__ == '__main__':
    model = load_model('play_predictor.h5')
    app.run_server(debug=True)








