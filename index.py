# Importamos las bibliotecas necesarias 
import os
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from crypt import methods           # Para subir archivos con métodos POST
from flask import Flask, render_template,request,redirect,url_for,make_response # Para utilizar flask
from werkzeug.utils import secure_filename # Para el manejo de nombre de archivos
from apyori import apriori
# ---------------------------- { AQUÍ COMIENZA LA PARTE DE FLASK } ----------------------------
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['csv'])
ListaListas = {}

# --- { REGLAS DE ASOCIACIÓN } ---
@app.route('/reglas_asociacion',methods=['GET', 'POST'])
def reglas_asociacion():
    if request.method == 'POST':
        #Importamos los datos
        ra_file = request.files["ra_csvfile"]
        # Separamos el nombre del archivo
        filename = secure_filename(ra_file.filename)
        file = filename.split('.')
        if file[1] in ALLOWED_EXTENSIONS:
            # Leemos los datos
            df = pd.read_csv(ra_file)
            # Procesamos los datos
            #Se incluyen todas las transacciones en una sola lista
            Transacciones = df.values.reshape(-1).tolist() #-1 significa 'dimensión no conocida'
            Lista = pd.DataFrame(Transacciones)
            Lista['Frecuencia'] = 1
            #Se agrupa los elementos
            Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
            Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
            Lista = Lista.rename(columns={0 : 'Item'})
            #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
            #level=0 especifica desde el primer índice
            ListaListas = df.stack().groupby(level=0).apply(list).tolist()
            #return redirect(url_for('apriori',ListaListas = ListaListas, tables=[Lista.to_html(classes='data')], titles=Lista.columns.values,filename = filename))
            return render_template('reglas_asociacion_parametros.html',  tables=[Lista.to_html(classes='data')], titles=Lista.columns.values,filename = filename)
    else:
        
        return render_template('reglas_asociacion.html')
    
@app.route('/reglas_asociacion/resultados',methods=['GET','POST'])
def apriori():
    if request.method == 'POST':
        soporte = request.form['soporte']
        confianza = request.form['confianza']
        elevacion = request.form['elevacion']
        
        
        return render_template('reglas_asociacion_resultados.html',soporte = soporte, confianza = confianza, elevacion = elevacion)
    else:
        return render_template('reglas_asociacion_parametros.html')



# Ruta por defecto de la página
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '_main_':
    
    app.run(host='0.0.0.0', port=80)
    



