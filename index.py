# Importamos las bibliotecas necesarias 

import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from crypt import methods           # Para subir archivos con métodos POST
from flask import Flask, render_template,request # Para utilizar flask
from werkzeug.utils import secure_filename # Para el manejo de nombre de archivos

# ---------------------------- { AQUÍ COMIENZA LA PARTE DE FLASK } ----------------------------
app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['csv'])
app.config['UPLOAD_FOLDER'] = 'static/uploads'
 
# --- { REGLAS DE ASOCIACIÓN } ---
@app.route('/reglas_asociacion.html',methods = ["POST"])
def apriori():
    # Importamos los datos
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
        ListaM = pd.DataFrame(Transacciones)
        ListaM['Frecuencia'] = 1
        #Se agrupa los elementos
        ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
        ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
        ListaM = ListaM.rename(columns={0 : 'Item'})
        #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
        #level=0 especifica desde el primer índice
        ListaListas = df.stack().groupby(level=0).apply(list).tolist()
        return render_template('reglas_asociacion.html',  tables=[ListaM.to_html(classes='data')], titles=ListaM.columns.values,filename = filename, barchart = plt)



# Ruta por defecto de la página
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para ver las reglas de asociación
@app.route('/reglas_asociacion.html')
def reglas_asociacion():
    return render_template('reglas_asociacion.html')

if __name__ == '_main_':
    app.run(debug=True)



