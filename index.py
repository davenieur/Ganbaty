# Importamos las bibliotecas necesarias 
import os
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from crypt import methods           # Para subir archivos con métodos POST
from flask import Flask, render_template,request,redirect,url_for,send_from_directory # Para utilizar flask
from werkzeug.utils import secure_filename # Para el manejo de nombre de archivos
from apyori import apriori as ap
# ---------------------------- { AQUÍ COMIENZA LA PARTE DE FLASK } ----------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "./uploads"


ALLOWED_EXTENSIONS = set(['csv'])

# --- { REGLAS DE ASOCIACIÓN } ---

@app.route('/reglas_asociacion/error',methods=['GET','POST'])
def error_parametros():
    return "Error en los parametros"

@app.route('/reglas_asociacion/resultados',methods=['GET','POST'])
def apriori():
    if request.method == 'POST':
        #filename = request.form['filename']
        soporte = request.form['soporte']
        confianza = request.form['confianza']
        elevacion = request.form['elevacion']

        

        
        
        return render_template('reglas_asociacion_resultados.html',soporte = soporte, confianza = confianza, elevacion = elevacion)
    else:
        filename = request.args.get('filename')
        
            
            
       
        
        return render_template('reglas_asociacion_parametros.html',filename = filename,soporte = 0, confianza = 0, elevacion = 0)



@app.route('/reglas_asociacion',methods=['GET', 'POST'])
def reglas_asociacion():
    if request.method == 'POST':
        #Importamos los datos
        ra_file = request.files["ra_csvfile"]

        # Obtenemos el nombre del archivo
        filename = secure_filename(ra_file.filename)

        # Separamos el nombre del archivo
        file = filename.split('.')
        
        if file[1] in ALLOWED_EXTENSIONS:
            #ra_file.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))

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
            
            for item in Resultados:
                #El primer índice de la lista
                Emparejar = item[0]
                items = [x for x in Emparejar]
                print("Regla: " + str(item[0]))

                #El segundo índice de la lista
                print("Soporte: " + str(item[1]))

                #El tercer índice de la lista
                print("Confianza: " + str(item[2][0][2]))
                print("Lift: " + str(item[2][0][3])) 
                

            return render_template('reglas_asociacion_resultados.html',soporte = soporte, confianza = confianza, elevacion = elevacion, size = len(Resultados))
            #return redirect(url_for('apriori', soporte = soporte, filename = filename))

        else:
            return render_template('reglas_asociacion_error_archivo.html',filename = filename)   
            
    else:
        
        return render_template('reglas_asociacion.html')
    


# Ruta por defecto de la página
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '_main_':
    
    app.run(host='0.0.0.0', port=80)
    



