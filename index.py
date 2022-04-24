# Importamos las bibliotecas necesarias 
import os
import base64
from io import BytesIO
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from crypt import methods           # Para subir archivos con métodos POST
from flask import Flask, redirect, render_template,request,Response, url_for# Para utilizar flask
from werkzeug.utils import secure_filename # Para el manejo de nombre de archivos
from apyori import apriori as ap
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_svg import FigureCanvasSVG
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance
from flask_wtf import FlaskForm
from wtforms import Form, IntegerField,SelectField,SubmitField, FileField

# ---------------------------- { AQUÍ COMIENZA LA PARTE DE FLASK } ----------------------------
app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = "/static/csv"


# --- { Forms } ---
class raForm(FlaskForm):
    ra_file = FileField()


ALLOWED_EXTENSIONS = set(['csv'])

# --- { REGLAS DE ASOCIACIÓN } ---


@app.route('/reglas_asociacion/parametros',methods=['GET', 'POST'])
def ra_upload():
    if request.method == 'POST':
        #Importamos los datos
        ra_file = request.files["ra_csvfile"]

        # Obtenemos el nombre del archivo
        filename = secure_filename(ra_file.filename)

        # Separamos el nombre del archivo
        file = filename.split('.')
        
        if file[1] in ALLOWED_EXTENSIONS:
            #ra_file.save(os.path.join(app.static_folder,filename))
             # Leemos los datos
            df = pd.read_csv(ra_file,header=None)

            #Se incluyen todas las transacciones en una sola lista
            Transacciones = df.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida'
            
            #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
            Lista = pd.DataFrame(Transacciones)
            Lista['Frecuencia'] = 1

            #Se agrupa los elementos
            Lista = Lista.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
            Lista['Porcentaje'] = (Lista['Frecuencia'] / Lista['Frecuencia'].sum()) #Porcentaje
            Lista = Lista.rename(columns={0 : 'Item'})
            

            #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
            #level=0 especifica desde el primer índice
            TransaccionesLista = df.stack().groupby(level=0).apply(list).tolist()

            soporte = request.form['soporte']
            confianza = request.form['confianza']
            elevacion = request.form['elevacion']

            Reglas = ap(TransaccionesLista,
                    min_support = float(soporte),
                    min_confidence = float(confianza),
                    min_lift = float(elevacion))

            Resultados = list(Reglas)
            return render_template('reglas_asociacion_resultados.html',Resultados = Resultados,soporte = soporte, confianza = confianza, elevacion = elevacion, size = len(Resultados))
            
           
     
    else:
        
        return render_template('reglas_asociacion_parametros.html')


@app.route('/reglas_asociacion',)
def ra():
    return render_template('reglas_asociacion.html')



# Ruta por defecto de la página
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '_main_':
    
    app.run(host='0.0.0.0', port=80)
    



