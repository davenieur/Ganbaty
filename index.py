import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from crypt import methods           # Para subir archivos con métodos POST
from flask import Flask, render_template,request # Para utilizar flask
from werkzeug.utils import secure_filename # Para el manejo de nombre de archivos

# ---------------------------- { AQUÍ COMIENZA LA PARTE DE FLASK } ----------------------------
app = Flask(__name__)




# Subir archivos en la sección de reglas de asociación
app.config['UPLOAD_FOLDER'] = 'static/upload/reglas_asociacion'
ALLOWED_EXTENSIONS = set(['csv'])
@app.route('/reglas_asociacion.html/tabla',methods = ["POST"])
def apriori():
    ra_file = request.files["ra_csvfile"]
    df = pd.read_csv(ra_file)
    return render_template('simple.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)



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



