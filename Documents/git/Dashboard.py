
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import pandas as pd
import datetime
import json# librería para conectarnos a base de datos de oracle.
import cx_Oracle as ORA
# librería para trabajar con DataFrames (tablas)
import pandas as pd
# librería de álgebra: vectores bla bla
import numpy as np
# librerías para trabajar con fechas
import time
import datetime
import calendar
#librerias analisis de clusters
from scipy import stats, integrate
from sklearn.cluster import KMeans
#librerias para visualizacion de datos
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import plotly.plotly as py
# librería para importar modelo
import pickle
# librería para conectarnos a base de datos de oracle.
import json
import cx_Oracle as ORA

pd.set_option('display.max_colwidth', -1)
pd.set_option("display.colheader_justify","left")


# PARA MOSTRAR GRÁFICOS

df4=pd.read_csv('C:/Users/Alvaro Santoma/Documents/ASOS aws/Sprint 6/conjunto_reducido_unmillon.csv', 
               error_bad_lines=False, warn_bad_lines =False)
df4 = df4.fillna(0)
df4.drop(df4.columns[[0]], axis=1, inplace=True)

consumo_anual_elec =df4.iloc[:,1:13].sum(axis=1)
consumo_anual_gas =df4.iloc[:,25:37].sum(axis=1)
elidx0 = [i for i in range(len(df4)) if consumo_anual_elec[i] > 100000000000]
gasidx0 = [i for i in range(len(df4)) if consumo_anual_gas[i] > 100000000000]
indices_para_filtrar = list(set(elidx0 + gasidx0))
indices_para_filtrar.sort() # hay 2340 para filtrar
df4 =df4.drop(df4.index[indices_para_filtrar])
df4.reset_index(drop=True, inplace=True)
df4_total = df4.shape[0]

# variables para las figuras al principio
df_tabla1 = df4.loc[df4['TIPO_CLIENTE']==1]
df_media_electrico_tipo_1 = df_tabla1.iloc[:,:12].mean()
df_media_gas_tipo_1 = df_tabla1.iloc[:,24:36].mean()

#CARGA DE Dataset para el modelo
data_modelo=pd.read_csv('C:/Users/Alvaro Santoma/Documents/ASOS aws/Sprint 6/bills_consum.csv', 
               sep=';')

# Carga dataset para el cluster
df_cluster = pd.read_csv('C:/Users/Alvaro Santoma/Documents/ASOS aws/Sprint 6/Notebooks/connsumo_normalizado_cluster.csv', error_bad_lines=False)
df_cluster.drop(df_cluster.columns[[0]], axis=1, inplace=True)
elec_df = df_cluster.iloc[:,1:13]
gas_df = df_cluster.iloc[:,13:25]
total_df = df_cluster.shape[0]


"""
    Importo el modeldo
"""
infile = open('modelo_cesa_no_cesa','rb')
modelo = pickle.load(infile)
infile.close()


#FUNCTIONS 

def consumo_electrico(df, doble=None, fact=None, norm = None, parcial_df =None, total_df = None):
    if doble != None:
        trace1 = go.Scatter(x=list(range(1,25)), y=df, marker ={'color':'red', 'size':10}, mode='markers+lines', name='media eléctrico')
        data = go.Data([trace1])
        if fact != None:
            layout = go.Layout(title='Media facturas eléctrico', xaxis=dict(range=[1,25],title= 'Mes'), yaxis= dict(title='Facturas €'))
        else:
            layout = go.Layout(title='Media consumo eléctrico', xaxis=dict(range=[1,25],title= 'Mes'), yaxis= dict(title='Consumo kWhe'))
        figure = go.Figure(data=data,layout=layout) 
        return figure

    trace1 = go.Scatter(x=list(range(1,13)), y=df, marker ={'color':'red', 'size':10}, mode='markers+lines', name='media eléctrico')
    data = go.Data([trace1])
    if norm != None:
        layout = go.Layout(title='Media del consumo eléctrico normalizado en 2016\n en un cluster que representa el {}%'.format(round((100*parcial_df)/total_df,1)), xaxis=dict(range=[1,13],title= 'Mes'), yaxis= dict(title='Consumo kWhe', range=[0, 0.3]))
    else:
        layout = go.Layout(title='Media consumo eléctrico', xaxis=dict(range=[1,13],title= 'Mes'), yaxis= dict(title='Consumo kWhe'))
    figure = go.Figure(data=data,layout=layout)  

    return figure


def consumo_gas1(df, norm = None, parcial_df =None, total_df = None):
    trace1 = go.Scatter(x=list(range(1,13)), y=df, marker ={'color':'yellow', 'size':10}, mode='markers+lines', name='media gas')
    data = go.Data([trace1])
    if norm != None:
        layout = go.Layout(title='Media del consumo de gas normalizado en 2016\n en un cluster que representa el {}%'.format(round((100*parcial_df)/total_df,1)), xaxis=dict(range=[1,13],title= 'Mes'), yaxis= dict(title='Consumo kWhe', range=[0, 0.3]))
    else:
        layout = go.Layout(title='Media gas', xaxis=dict(range=[1,13],title= 'Mes'), yaxis= dict(title='Consumo kWhe'))
    figure = go.Figure(data=data,layout=layout)  

    return figure

# Se empieza ael Dash

app = dash.Dash()
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True
#CSS y estilos del Dashboard
app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

app.layout = html.Div([
    dcc.Tabs(
        tabs=[
            {'label': 'GLOBAL', 'value': 1},
            {'label': 'TIPO DE CLIENTE', 'value': 2},
            {'label': 'PARTICULAR', 'value': 3},
            {'label': 'ANÁLISIS DE CLÚSTERS', 'value': 4},
            ],
        value=2,
        id='tab',
    ),
    html.Div(id='container')
], style={
        'width': '80%',
        'fontFamily': 'Sans-Serif',
        'margin-left': 'auto',
        'margin-right': 'auto'
    })

#Se definen los diferentes callbacks
@app.callback(Output('container', 'children'), [Input('tab', 'value')])
def display_content(selected_tab):
    if selected_tab == 2:
        return html.Div(children=[
        html.H1(children='Visualización Según Tipo de Clientes'),

            html.Div([
            dcc.RadioItems(
                id='electrico',
                options=[
                    {'label': 'Activo Eléctrico', 'value': 1},
                    {'label': 'No Activo Eléctrico', 'value': 2},
                    {'label': 'Cesado Eléctrico', 'value': 3}
                ],
                value=1, style ={'fontSize': 20},  className="four columns"
            ),
            dcc.RadioItems(
                id='gas',
                options=[
                    {'label': 'Activo Gas', 'value': 0},
                    {'label': 'No Activo Gas', 'value': 3},
                    {'label': 'Cesado Gas', 'value': 6}
                ],
                value=0, style ={'fontSize': 20},  className="four columns"
            ),

            dcc.Dropdown(
                id = 'año',
                options=[
                    {'label': '2015', 'value': '2015'},
                    {'label': '2016', 'value': '2016'}
                ],
                value='2015', className= "four columns"
            ), 
        ], className="row"),

        html.Div([
            html.Div(id='tipo_clientes')
        ], className="row" ),  
        ])

    if selected_tab == 1:
       return html.Div(children=[
        html.H1(children='Visualización Global'),

            html.Div([
            dcc.Dropdown(
                id = 'año_global',
                options=[
                    {'label': '2015', 'value': '2015'},
                    {'label': '2016', 'value': '2016'}
                ],
                value='2015', className= "three columns"
            ), 
        ], className="row"),

        html.Div([
            html.Div(id='global')
        ], className="row" ),      
        ])

    if selected_tab == 3:
        return html.Div(children=[
        html.H1(children='Particular'),
        dcc.Input(id='IDSS', value='700000360224', type='number'),
        html.Div(id='modelo'),
        ])

    if selected_tab == 4:
        return html.Div(children=[
        html.H1(children='Análisis de Clústers'),
        dcc.RadioItems(
            id = 'cluster_tipo',
            options=[
            {'label': 'Eléctrico', 'value': 'E'},
            {'label': 'Gas', 'value': 'G'}
        ],
        value='E'
        ),
        html.Div(id='cluster_view'),
        ])

@app.callback(
     dash.dependencies.Output('cluster_view', 'children'),
     [dash.dependencies.Input('cluster_tipo', 'value')])

def update_analisis_cluster(tipo):

    if tipo == 'E':
        numero_de_clusters = 4
        model = KMeans(n_clusters=numero_de_clusters, random_state=40310)
        model.fit(elec_df)
        labels=model.predict(elec_df)
        
        #saco los indices de cada grupo
        eidx0 = [i for i in range(len(labels)) if labels[i] == 0]
        eidx1 = [i for i in range(len(labels)) if labels[i] == 1]
        eidx2 = [i for i in range(len(labels)) if labels[i] == 2] 
        eidx3 = [i for i in range(len(labels)) if labels[i] == 3]
        leng = elec_df.shape[0]
        media_elec0 = elec_df.iloc[eidx0,:]
        media_elec0 = media_elec0.iloc[:,:12].mean()
        media_elec1 = elec_df.iloc[eidx1,:]
        media_elec1 = media_elec1.iloc[:,:12].mean()
        media_elec2 = elec_df.iloc[eidx2,:]
        media_elec2 = media_elec2.iloc[:,:12].mean()        
        media_elec3 = elec_df.iloc[eidx3,:]
        media_elec3 = media_elec3.iloc[:,:12].mean()
        return html.Div([
            html.Div([
            html.H2('Número de clientes: {}'.format(total_df)),
            ], className= "row"),
        html.Div([
            html.Div([
                dcc.Graph(id='consumo_electrico0', figure=consumo_electrico(media_elec0, norm = 1, parcial_df=len(eidx0), total_df=total_df))
            ], className="six columns"),
            html.Div([
                dcc.Graph(id='consumo_electrico1', figure=consumo_electrico(media_elec1, norm = 1, parcial_df=len(eidx1), total_df=total_df))
            ], className="six columns"),
        ], className="row"),
        html.Div([
            html.Div([
                dcc.Graph(id='consumo_electrico2', figure=consumo_electrico(media_elec2, norm = 1, parcial_df=len(eidx2), total_df=total_df))
            ], className="six columns"),

            html.Div([
                dcc.Graph(id='consumo_electrico3', figure=consumo_electrico(media_elec3, norm = 1, parcial_df=len(eidx3), total_df=total_df))
            ], className="six columns"),
        ], className="row"),
        ])


    #gas 
    if tipo == 'G':
        numero_de_clusters = 4
        model = KMeans(n_clusters=numero_de_clusters, random_state=1)
        model.fit(gas_df)
        labels=model.predict(gas_df)
        
        #saco los indices de cada grupo
        gidx0 = [i for i in range(len(labels)) if labels[i] == 0]
        gidx1 = [i for i in range(len(labels)) if labels[i] == 1]
        gidx2 = [i for i in range(len(labels)) if labels[i] == 2] 
        gidx3 = [i for i in range(len(labels)) if labels[i] == 3]
        media_gas0 = gas_df.iloc[gidx0,:]
        media_gas0 = media_gas0.iloc[:,:12].mean()
        media_gas1 = gas_df.iloc[gidx1,:]
        media_gas1 = media_gas1.iloc[:,:12].mean()
        media_gas2 = gas_df.iloc[gidx2,:]
        media_gas2 = media_gas2.iloc[:,:12].mean()        
        media_gas3 = gas_df.iloc[gidx3,:]
        media_gas3 = media_gas3.iloc[:,:12].mean()        
  
        return html.Div([
                  html.Div([
                  html.H2('Número de clientes: {}'.format(total_df)),
            ], className= "row"),
        html.Div([
            html.Div([
                dcc.Graph(id='consumo_g0', figure=consumo_gas1(media_gas0, norm = 1, parcial_df=len(gidx0), total_df=total_df))
            ], className="six columns"),
            html.Div([
                dcc.Graph(id='consumo_g1', figure=consumo_gas1(media_gas1, norm = 1, parcial_df=len(gidx1), total_df=total_df))
            ], className="six columns"),
        ], className="row"),
        html.Div([
            html.Div([
                dcc.Graph(id='consumo_g2', figure=consumo_gas1(media_gas2, norm = 1, parcial_df=len(gidx2), total_df=total_df))
            ], className="six columns"),

            html.Div([
                dcc.Graph(id='consumo_g3', figure=consumo_gas1(media_gas3, norm = 1, parcial_df=len(gidx3), total_df=total_df))
            ], className="six columns"),
        ], className="row"),
        ])

@app.callback(
     dash.dependencies.Output('tipo_clientes', 'children'),
     [dash.dependencies.Input('electrico', 'value'),
      dash.dependencies.Input('gas', 'value'),
      dash.dependencies.Input('año', 'value')])

def update_number_clients(electrico, gas, año):
    module = electrico + gas
    df_tabla = df4.loc[df4['TIPO_CLIENTE']==module]
    numero_clientes = df_tabla.shape[0]

    kpi_anual_2016_e = '-'
    kpi_anual_2016_g = '-'
    if año =='2015':
        media_elec = df_tabla.iloc[:,:12].mean()
        mensual_kpi = round(100*(media_elec[11] - media_elec[10])/media_elec[10], 2)
        media_gas = df_tabla.iloc[:,24:36].mean()
        mensual_kpi_g = round(100*(media_gas[11] - media_gas[10])/media_gas[10], 2)
    else:
        media_elec = df_tabla.iloc[:,12:24].mean()
        mensual_kpi = round(100*(media_elec[11] - media_elec[10])/ media_elec[10],2)
        media_gas = df_tabla.iloc[:,36:48].mean()
        mensual_kpi_g = round(100*(media_gas[11] - media_gas[10])/ media_gas[10], 2)
        kpi_anual_2016_e = round(100*(media_elec.sum() - df_tabla.iloc[:,:12].mean().sum()) / df_tabla.iloc[:,:12].mean().sum(), 2)
        kpi_anual_2016_g = round(100*(media_gas.sum() - df_tabla.iloc[:,24:36].mean().sum()) /  df_tabla.iloc[:,24:36].mean().sum(), 2)

    consumo_elec_medio = round(media_elec.mean(), 2)
    consumo_gas_medio = round(media_gas.mean(), 2)
    if gas == 3:
        consumo_gas_medio = 0.0
        mensual_kpi = '-'
        kpi_anual_2016_g = '-'
        media_gas = pd.Series([0]*12) 


    return html.Div([
        html.Div([
             html.Label('Hay {} clientes seleccionados'.format(numero_clientes), style={'fontSize': 20}, className="three columns")
        ], className = 'row'),
        html.Div([
                html.Div([
                html.H1('CONSUMO: ELÉCTRICO'),
            ], className= "six columns"),
                  html.Div([
                  html.H1('GAS'),
            ], className= "six columns"),
                  html.Div([          
                  html.Div([
                  html.H2('Mensual medio: {} kWhe'.format(consumo_elec_medio)),
            ], className= "six columns"),
             html.Div([
                  html.H2('Mensual medio: {} m3'.format(consumo_gas_medio)),
            ], className= "six columns"),
            ], className= "row"),   
            html.Div([
            html.Div([
                  html.H2('Respecto al año anterior: {}%'.format(kpi_anual_2016_e)),
            ], className= "six columns"),
            html.Div([
                  html.H2('Respecto al año anterior: {}%'.format(kpi_anual_2016_g)),
            ], className= "six columns"),    
            ], className= "row"),
                ]),
    
            html.Div([
            html.Div([
                dcc.Graph(id='consumo_electrico', figure=consumo_electrico(media_elec))
            ], className="six columns"),

            html.Div([
                dcc.Graph(id='consumo_gas', figure=consumo_gas1(media_gas))
            ], className="six columns"),
        ], className="row"),
        ])


@app.callback(dash.dependencies.Output('modelo', 'children'),
     [dash.dependencies.Input('IDSS', 'value')])

def update_output_div(value):
    x = data_modelo.loc[data_modelo['ID_SECTOR_SUPPLY']==int(value)].iloc[:,3:51]
    if x.shape == (1,48):
        y = round(100*(modelo.predict_proba(x)[0][1]),2)
        datos_elec = x[x.columns[::-1]].iloc[:,:24].mean()
        consumo_elec_medio = round(x[x.columns[::-1]].iloc[:,12:24].mean().mean(), 2)
        kpi_anual_2016_e = round(100*(x[x.columns[::-1]].iloc[:,12:24].mean().mean() - x[x.columns[::-1]].iloc[:,:12].mean().mean()) /  x[x.columns[::-1]].iloc[:,:12].mean().mean(), 2)
        media_fact = round(x[x.columns[::-1]].iloc[:,24:48].mean().mean(), 2)
        datos_fact = x[x.columns[::-1]].iloc[:,24:48].mean()
        kpi_anual_2016_f = round(100*(x[x.columns[::-1]].iloc[:,36:48].mean().mean() - x[x.columns[::-1]].iloc[:,24:36].mean().mean()) /  x[x.columns[::-1]].iloc[:,24:36].mean().mean(), 2)


        return html.Div([
        html.Div([    
        html.H3('La probabilidad de cese del cliente es del: {}%'.format(y))
        ], className = 'row'),
        html.Div([
                    html.Div([
                    html.H1('CONSUMO'),
            ], className= "six columns"),
                  html.Div([
                  html.H1('FACTURAS'),
            ], className= "six columns"),
                  html.Div([          
                  html.Div([
                  html.H2('Mensual medio: {} kWhe'.format(consumo_elec_medio)),
            ], className= "six columns"),
             html.Div([
                  html.H2('Mensual medio: {} Euros'.format(media_fact)),
            ], className= "six columns"),
            ], className= "row"),   
            html.Div([
            html.Div([
                  html.H2('Respecto al año anterior: {}%'.format(kpi_anual_2016_e)),
            ], className= "six columns"),
            html.Div([
                  html.H2('Respecto al año anterior: {}%'.format(kpi_anual_2016_f)),
            ], className= "six columns"),    
            ], className= "row")
            ]),

        html.Div([
            html.Div([
                dcc.Graph(id='consumo_electrico', figure=consumo_electrico(datos_elec, 'True'))
            ], className="six columns"),

            html.Div([
                dcc.Graph(id='factura_electrica', figure=consumo_electrico(datos_fact, 'True', 'True'))
            ], className="six columns"),
        ], className="row")
        ])

    else:
        return html.Div([
        html.Label('El ID_SECTOR_SUPPLY no se encuentra en la base de datos "{}"'.format(value))
        ]) 




@app.callback(
     dash.dependencies.Output('kpis', 'children'),
     [dash.dependencies.Input('electrico', 'value'),
      dash.dependencies.Input('gas', 'value'),
      dash.dependencies.Input('año', 'value'),
      dash.dependencies.Input('tab', 'value')]) 



@app.callback(
     dash.dependencies.Output('global', 'children'),
     [dash.dependencies.Input('año_global', 'value')])

def update_output_div_global(año):
    

    df_gas = df4.loc[(df4['TIPO_CLIENTE']==1) | (df4['TIPO_CLIENTE']==2) | (df4['TIPO_CLIENTE']==3)]
    df_gas_2015 = round(df_gas.iloc[:,24:36].mean().mean(), 2)
    df_gas_2016 = round(df_gas.iloc[:, 36:48].mean().mean(), 2)
    kpi_anual_2016_g = round(100*(df_gas_2016 - df_gas_2015)/ df_gas_2015, 2)

    df_elec = df4.loc[(df4['TIPO_CLIENTE']==1) | (df4['TIPO_CLIENTE']==4) | (df4['TIPO_CLIENTE']==7)]
    df_elec_2015 = round(df_elec.iloc[:,:12].mean().mean(), 2)
    df_elec_2016 = round(df_elec.iloc[:,12:24].mean().mean(), 2)
    kpi_anual_2016_e =  round(100*(df_elec_2016 - df_elec_2015)/ df_elec_2015, 2)

    df_mean = df_elec.iloc[:,12:24].mean()
    df_mean_gas = df_gas.iloc[:,36:48].mean()


    consumo_elec_medio = eval('df_elec_' + str(año))
    consumo_gas_medio = eval('df_gas_' + str(año))
    if año == '2015':       
        kpi_anual_2016_e = '-' 
        kpi_anual_2016_g = '-' 
        df_mean = df_elec.iloc[:,:12].mean()
        df_mean_gas = df_gas.iloc[:,24:36].mean()

    return html.Div([
                    html.Div([
                    html.H1('CONSUMO ELÉCTRICO'),
            ], className= "six columns"),
                  html.Div([
                  html.H1('GAS'),
            ], className= "six columns"),
                  html.Div([          
                  html.Div([
                  html.H2('Mensual medio: {} kWhe'.format(consumo_elec_medio)),
            ], className= "six columns"),
             html.Div([
                  html.H2('Mensual medio: {} m3'.format(consumo_gas_medio)),
            ], className= "six columns"),
            ], className= "row"),   
            html.Div([
            html.Div([
                  html.H2('Respecto al año anterior: {}%'.format(kpi_anual_2016_e)),
            ], className= "six columns"),
            html.Div([
                  html.H2('Respecto al año anterior: {}%'.format(kpi_anual_2016_g)),
            ], className= "six columns"),    
            ], className= "row"),   
       
        html.Div([
            html.Div([
                dcc.Graph(id='consumo_electrico_global', figure=consumo_electrico(df_mean))
            ], className="six columns"),

            html.Div([
                dcc.Graph(id='consumo_gas_global', figure=consumo_gas1(df_mean_gas))
            ], className="six columns"),
        ], className="row")
            ])


@app.callback(
     dash.dependencies.Output('consumo_electrico', 'figure'),
     [dash.dependencies.Input('electrico', 'value'),
      dash.dependencies.Input('gas', 'value'),
      dash.dependencies.Input('año', 'value')]) 

def electrico_consumo(electrico, gas, año=None):
    if electrico != 2:
        tipo = gas + electrico
        df = df4.loc[df4['TIPO_CLIENTE']==tipo]
        if año == '2015' or año == None:
            df_mean = df.iloc[:,:12].mean()
        else:
            df_mean = df.iloc[:,12:24].mean()
        return consumo_electrico(df_mean)
    else:
        a = pd.Series([0]*12) 
        return consumo_electrico(a)
    

@app.callback(dash.dependencies.Output('consumo_gas', 'figure'),
              [dash.dependencies.Input('gas', 'value'),                  
               dash.dependencies.Input('electrico', 'value'),
               dash.dependencies.Input('año','value')])  

def gas_consumo(gas, electrico, año):
    if gas != 3:
        tipo = gas + electrico
        df = df4.loc[df4['TIPO_CLIENTE']==tipo]
        if año == '2015':
            df_mean = df.iloc[:,24:36].mean()
        else:
            df_mean = df.iloc[:,36:48].mean()
        return consumo_gas1(df_mean)
    else:
        a = pd.Series([0]*12) 
        return consumo_gas1(a)


if __name__ == '__main__':
    app.run_server(debug=True)

