

# coding: utf-8

# In[1]:


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
# librería para generar reports de AMM #merci arnau
# librerías para trabajar con fechas
import time
import datetime

#librerias analisis de clusters
import seaborn as sns
from scipy import stats, integrate


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

from sklearn.cluster import KMeans


# In[2]:


pd.set_option('display.max_colwidth', -1)
pd.set_option("display.colheader_justify","left")


# In[3]:


# fila = 3
# try:
#     conf = pd.read_csv("./config_file.csv", sep = ';');
# except:
#     sys.exit("No se ha encontrado el archivo config_file_rep.csv: Comprobar que se encuentra en el directorio principal. \n");

# try:
#     esquema = conf.schema[fila];
# except: 
#     sys.exit("No se ha leido correctamente el esquema del archivo config_file_rep.csv: Comprobar formato. \n");

# try:
#     password = conf.passw[fila];
# except: 
#     sys.exit("No se ha leido correctamente el password del archivo config_file_rep.csv: Comprobar formato. \n");

# try:
#     servicio = conf.service[fila];
# except: 
#     sys.exit("No se ha leido correctamente el servicio del archivo config_file_rep.csv: Comprobar formato. \n");

# try:
#     update_time = conf.time[fila]*60;
# except: 
#     sys.exit("No se ha leido correctamente el tiempo del archivo config_file_rep.csv: Comprobar formato. \n");

# credentials = esquema + '/' + password + '@' + servicio
# print(credentials)


# # In[9]:


# conn = ORA.connect(credentials)  #Conexió a la DB 
# # puntos de suministro comunes
# query1 = """ select * from asos_aux_tipo_cliente
# where rownum <10000"""


# READ DATA
#desde oracle directa
# df_ora = pd.read_sql(query1, con = conn)
# df_ora.head()
# df_tabla = df_ora.loc[df_ora['TIPO_CLIENTE'] == '2']
# df_dni = df_ora.loc[df_ora['DOC_NUMBER_G'] == '46342855V']
# df_total = df_ora.shape[0]

# para analisis avanzado. De momento no se usa
# df=pd.read_csv('C:/Users/Alvaro Santoma/conjunto_solo2016_normalizado_.csv', error_bad_lines=False)
# df = df.fillna(0)
# df.drop(df.columns[[0]], axis=1, inplace=True)

# df2 = pd.read_csv('C:/Users/Alvaro Santoma/Downloads/datos_conjuntos_sin_nulos.csv', error_bad_lines=False)
# df2 = df2.fillna(0)
# df2.drop(df2.columns[[0]], axis=1, inplace=True)

# Cálculo de consumos totales
# consumo_anual_elec =df2.iloc[:,1:13].sum(axis=1)
# consumo_anual_gas =df2.iloc[:,25:37].sum(axis=1)

# # Filtraje de los puntos anomalos
# elidx0 = [i for i in range(len(df2)) if consumo_anual_elec[i] > 100000000000]
# gasidx0 = [i for i in range(len(df2)) if consumo_anual_gas[i] > 100000000000]
# indices_para_filtrar = list(set(elidx0 + gasidx0))
# indices_para_filtrar.sort() # hay 2340 para filtrar
# df2 =df2.drop(df2.index[indices_para_filtrar])
# df2.reset_index(drop=True, inplace=True)


# df = df.drop(df.index[indices_para_filtrar])
# df.reset_index(drop=True, inplace=True)

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



"""
    Create a Dash about archive 'log' for view the warnings
"""
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd
import datetime
import json# librería para conectarnos a base de datos de oracle.
import cx_Oracle as ORA
# librería para trabajar con DataFrames (tablas)
import pandas as pd
# librería de álgebra: vectores bla bla
import numpy as np
# librería para generar reports de AMM #merci arnau
# librerías para trabajar con fechas
import time
import datetime
import plotly.plotly as py
# librería para importar modelo
import pickle

infile = open('modelo_cesa_no_cesa','rb')
modelo = pickle.load(infile)
infile.close()


#FUNCTIONS
_COLORS = {
    "red":          {50:"#FFEBEE", 100:"#FFCDD2", 200:"#EF9A9A", 300:"#E57373", 400:"#EF5350", 500:"#F44336", 600:"#E53935", 700:"#D32F2F", 800:"#C62828", 900:"#B71C1C"}, # pylint: disable=line-too-long
    "pink":         {50:"#FCE4EC", 100:"#F8BBD0", 200:"#F48FB1", 300:"#F06292", 400:"#EC407A", 500:"#E91E63", 600:"#D81B60", 700:"#C2185B", 800:"#AD1457", 900:"#880E4F"}, # pylint: disable=line-too-long
    "purple":       {50:"#F3E5F5", 100:"#E1BEE7", 200:"#CE93D8", 300:"#BA68C8", 400:"#AB47BC", 500:"#9C27B0", 600:"#8E24AA", 700:"#7B1FA2", 800:"#6A1B9A", 900:"#4A148C"}, # pylint: disable=line-too-long
    "deep purple":  {50:"#EDE7F6", 100:"#D1C4E9", 200:"#B39DDB", 300:"#9575CD", 400:"#7E57C2", 500:"#673AB7", 600:"#5E35B1", 700:"#512DA8", 800:"#4527A0", 900:"#311B92"}, # pylint: disable=line-too-long
    "indigo":       {50:"#E8EAF6", 100:"#C5CAE9", 200:"#9FA8DA", 300:"#7986CB", 400:"#5C6BC0", 500:"#3F51B5", 600:"#3949AB", 700:"#303F9F", 800:"#283593", 900:"#1A237E"}, # pylint: disable=line-too-long
    "blue":         {50:"#E3F2FD", 100:"#BBDEFB", 200:"#90CAF9", 300:"#64B5F6", 400:"#42A5F5", 500:"#2196F3", 600:"#1E88E5", 700:"#1976D2", 800:"#1565C0", 900:"#0D47A1"}, # pylint: disable=line-too-long
    "light blue":   {50:"#E1F5FE", 100:"#B3E5FC", 200:"#81D4FA", 300:"#4FC3F7", 400:"#29B6F6", 500:"#03A9F4", 600:"#039BE5", 700:"#0288D1", 800:"#0277BD", 900:"#01579B"}, # pylint: disable=line-too-long
    "cyan":         {50:"#E0F7FA", 100:"#B2EBF2", 200:"#80DEEA", 300:"#4DD0E1", 400:"#26C6DA", 500:"#00BCD4", 600:"#00ACC1", 700:"#0097A7", 800:"#00838F", 900:"#006064"}, # pylint: disable=line-too-long
    "teal":         {50:"#E0F2F1", 100:"#B2DFDB", 200:"#80CBC4", 300:"#4DB6AC", 400:"#26A69A", 500:"#009688", 600:"#00897B", 700:"#00796B", 800:"#00695C", 900:"#004D40"}, # pylint: disable=line-too-long
    "green":        {50:"#E8F5E9", 100:"#C8E6C9", 200:"#A5D6A7", 300:"#81C784", 400:"#66BB6A", 500:"#4CAF50", 600:"#43A047", 700:"#388E3C", 800:"#2E7D32", 900:"#1B5E20"}, # pylint: disable=line-too-long
    "light green":  {50:"#F1F8E9", 100:"#DCEDC8", 200:"#C5E1A5", 300:"#AED581", 400:"#9CCC65", 500:"#8BC34A", 600:"#7CB342", 700:"#689F38", 800:"#558B2F", 900:"#33691E"}, # pylint: disable=line-too-long
    "lime":         {50:"#F9FBE7", 100:"#F0F4C3", 200:"#E6EE9C", 300:"#DCE775", 400:"#D4E157", 500:"#CDDC39", 600:"#C0CA33", 700:"#AFB42B", 800:"#9E9D24", 900:"#827717"}, # pylint: disable=line-too-long
    "yellow":       {50:"#FFFDE7", 100:"#FFF9C4", 200:"#FFF59D", 300:"#FFF176", 400:"#FFEE58", 500:"#FFEB3B", 600:"#FDD835", 700:"#FBC02D", 800:"#F9A825", 900:"#F57F17"}, # pylint: disable=line-too-long
    "amber":        {50:"#FFF8E1", 100:"#FFECB3", 200:"#FFE082", 300:"#FFD54F", 400:"#FFCA28", 500:"#FFC107", 600:"#FFB300", 700:"#FFA000", 800:"#FF8F00", 900:"#FF6F00"}, # pylint: disable=line-too-long
    "orange":       {50:"#FFF3E0", 100:"#FFE0B2", 200:"#FFCC80", 300:"#FFB74D", 400:"#FFA726", 500:"#FF9800", 600:"#FB8C00", 700:"#F57C00", 800:"#EF6C00", 900:"#E65100"}, # pylint: disable=line-too-long
    "deep orange":  {50:"#FBE9E7", 100:"#FFCCBC", 200:"#FFAB91", 300:"#FF8A65", 400:"#FF7043", 500:"#FF5722", 600:"#F4511E", 700:"#E64A19", 800:"#D84315", 900:"#BF360C"}, # pylint: disable=line-too-long
    "brown":        {50:"#EFEBE9", 100:"#D7CCC8", 200:"#BCAAA4", 300:"#A1887F", 400:"#8D6E63", 500:"#795548", 600:"#6D4C41", 700:"#5D4037", 800:"#4E342E", 900:"#3E2723"}, # pylint: disable=line-too-long
    "grey":         {50:"#FAFAFA", 100:"#F5F5F5", 200:"#EEEEEE", 300:"#E0E0E0", 400:"#BDBDBD", 500:"#9E9E9E", 600:"#757575", 700:"#616161", 800:"#424242", 900:"#212121"}, # pylint: disable=line-too-long
    "blue grey":    {50:"#ECEFF1", 100:"#CFD8DC", 200:"#B0BEC5", 300:"#90A4AE", 400:"#78909C", 500:"#607D8B", 600:"#546E7A", 700:"#455A64", 800:"#37474F", 900:"#263238"}, # pylint: disable=line-too-long
    "black":        {50:"#000000", 100:"#000000", 200:"#000000", 300:"#000000", 400:"#000000", 500:"#000000", 600:"#000000", 700:"#000000", 800:"#000000", 900:"#000000"}, # pylint: disable=line-too-long
    "white":        {50:"#FFFFFF", 100:"#FFFFFF", 200:"#FFFFFF", 300:"#FFFFFF", 400:"#FFFFFF", 500:"#FFFFFF", 600:"#FFFFFF", 700:"#FFFFFF", 800:"#FFFFFF", 900:"#FFFFFF"}  # pylint: disable=line-too-long
}

def get_colors(clist):
    """
        Gives a list of colors using a list of index of the colors needed

        Args:
            clist:  list of indexs of the colors to be used

        Returns:
            list of colors in hex format
    """

    if isinstance(clist, list):
        output = [_COLORS[color.lower()][index] for color, index in clist]
    else:
        return _COLORS[clist[0]][clist[1]]

    #If there is only 1 color it shouldn't return a list
    return output if len(clist) > 1 else output[0]

def get_options(iterable):
    """
        Populates a dash dropdawn from an iterable
    """
    lista = []
    for x in iterable:
        if x == '1':
            lista.append({"label":"1. Activo gas y electrico", "value": x} )
        if x == '2':
            lista.append({"label":"2.Activo gas", "value": x} )
        if x == '3':
            lista.append({"label":"3.Activo electrico", "value": x} )
        if x == '4':
            lista.append({"label":"4.Cesado Gas", "value": x} )
        if x == '5':
            lista.append({"label":"5.Cesado Electrico", "value": x} )        
    return lista

    # return [{"label": x, "value": x} for x in iterable]

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )   


def table_details(df_input, module, n_rows=50):
   """
       Updates the details table
   """

   df = df_input.reset_index().copy()
   filas_totales = df_input.shape[0]

   header = {"values": [], "fill": {"color": []}, "font": {"color": []}}

   for col in df.columns:
       header["values"] += ["<b>{}</b>".format(col)]
       header["font"]["color"] += ["white"]

   data = go.Table(header=header,
                   cells={"values": df.head(n_rows).transpose().values})

   title = ' {} CLIENTES ENCONTRADOS DEL TIPO {}'.format(filas_totales,module)

   layout = go.Layout(title=title, height=800)
   return {"data": [data], "layout": layout}  

def consumo_electrico(df, doble=None, fact=None):
    # elec_df = df.iloc[:,1:13]
    # elec_mean = elec_df.mean()
    # elec_std = elec_df.std()
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
    layout = go.Layout(title='Media consumo eléctrico', xaxis=dict(range=[1,13],title= 'Mes'), yaxis= dict(title='Consumo kWhe'))
    figure = go.Figure(data=data,layout=layout)  

    return figure


def consumo_gas1(df, lista_indices = None):

    trace1 = go.Scatter(x=list(range(1,13)), y=df, marker ={'color':'yellow', 'size':10}, mode='markers+lines', name='media eléctrico')
    data = go.Data([trace1])
    layout = go.Layout(title='Media consumo gas', xaxis=dict(range=[1,13],title= 'Mes'), yaxis= dict(title='Consumo [m3]'))
    figure=go.Figure(data=data,layout=layout)  

    return figure

def calculo_clusters(df):
    ks = range(1, 8)
    inertias = []

    for k in ks:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k)
        
        # Fit model to samples
        model.fit(df)
        
        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    trace1 = go.Scatter(x=ks, y=inertias, marker ={'color':'yellow', 'size':10}, mode='markers+lines', name='media eléctrico')
    data = go.Data([trace1])
    layout = go.Layout(title='Grafico para escoger clusters')
    figure=go.Figure(data=data,layout=layout)  

    return figure



app = dash.Dash()
app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True

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
                value=1, style ={'fontSize': 20},  className="three columns"
            ),
            dcc.RadioItems(
                id='gas',
                options=[
                    {'label': 'Activo Gas', 'value': 0},
                    {'label': 'No Activo Gas', 'value': 3},
                    {'label': 'Cesado Gas', 'value': 6}
                ],
                value=0, style ={'fontSize': 20},  className="three columns"
            ),

            dcc.Dropdown(
                id = 'año',
                options=[
                    {'label': '2015', 'value': '2015'},
                    {'label': '2016', 'value': '2016'}
                ],
                value='2015', className= "three columns"
            ),
            dcc.Input(id='filas', type='number', value='50', className= "three columns"), 
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
       
        # html.Div([
        #     html.Div([
        #         dcc.Graph(id='consumo_electrico_global', figure=consumo_electrico(df_media_electrico_tipo_1))
        #     ], className="six columns"),

        #     html.Div([
        #         dcc.Graph(id='consumo_gas_global', figure=consumo_gas1(df_media_gas_tipo_1))
        #     ], className="six columns"),
        # ], className="row"),   
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
            options=[
            {'label': 'Eléctrico', 'value': 'E'},
            {'label': 'Gas', 'value': 'G'}
        ],
        value='E'
        ),
        dcc.Dropdown(
            options=[
                {'label': '1', 'value': '1'},
                {'label': '2', 'value': '2'},
                {'label': '3', 'value': '3'},
                {'label': '4', 'value': '4'},
                {'label': '5', 'value': '5'},
                {'label': '6', 'value': '6'},
                {'label': '7', 'value': '7'}
            ],
            value='4'
        ),
        html.Div(id='cluster'),
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
            # dcc.Graph(
            # id='taula',
            # figure = table_details(df4.iloc[:,-3:],'1')       
            # ), 
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
                dcc.Graph(id='consumo_gas_global', figure=consumo_gas1(df_media_gas_tipo_1))
            ], className="six columns"),
        ], className="row")
            ])


# primera tabla con clientes según tipo
# @app.callback(dash.dependencies.Output('taula', 'figure'),
#               [dash.dependencies.Input('electrico', 'value'),                  
#                dash.dependencies.Input('gas', 'value')])
#                # dash.dependencies.Input('filas', 'value')])       

# def taula(electrico, gas):
#     module = electrico + gas
#     df_tabla = df4.loc[df4['TIPO_CLIENTE']==module]
#     return table_details(df_tabla.iloc[:,-3:],module)

#dni borrado para prueba
# @app.callback(
#      dash.dependencies.Output('taula_dni', 'figure'),
#      [dash.dependencies.Input('dni', 'value')])

# def taula1(module):
#     #module.sort()
#     df_tabla = df4
#     if module:
#         df_tabla = df4.loc[df4['DOC_NUMBER_G']==module]
#     return table_details(df_tabla.iloc[:,-3:],module) 


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
 

# @app.callback(dash.dependencies.Output('clusters_elec', 'figure'),
#               [dash.dependencies.Input('electrico', 'value'),
#                dash.dependencies.Input('año','value')])

# def clusters_electrico(electrico, año):
#     tipo = electrico
#     df = df4.loc[df4['TIPO_CLIENTE']==tipo]
#     if año == '2015' or año == None:
#         df_clusters = df.iloc[df.loc[df['TIPO_CLIENTE']==1].index,0:12]
#     else:
#         df_clusters = df4.iloc[df4.loc[df4['TIPO_CLIENTE']==1].index,12:24]
#     return calculo_clusters(df_clusters)



if __name__ == '__main__':
    app.run_server(debug=True)


#