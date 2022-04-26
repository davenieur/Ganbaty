# Importamos las bibliotecas necesarias 
import os
import base64
from io import BytesIO
import pandas as pd                 # Para la manipulación y análisis de los datos
import numpy as np                  # Para crear vectores y matrices n dimensionales
import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
from crypt import methods           # Para subir archivos con métodos POST
from flask import Flask, render_template,request # Para utilizar las herramientas de Flask
from werkzeug.utils import secure_filename # Para el manejo de nombre de archivos
from apyori import apriori as ap    # Para el algoritmo apriori
from scipy.spatial.distance import cdist    # Para el cálculo de distancias
from scipy.spatial import distance  # Para el cálculo de distancias
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Para la estandarización de datos
import seaborn as sns             # Para la visualización de datos basado en matplotlib
import scipy.cluster.hierarchy as shc #Se importan las bibliotecas de clustering jerárquico para crear el árbol
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans #Se importan las bibliotecas para el clustering particional
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator # Para el método de la rodilla

# ---------------------------- { AQUÍ COMIENZA LA PARTE DE FLASK } ----------------------------
app = Flask(__name__)
#app.config['UPLOAD_FOLDER'] = "/static/csv"

ALLOWED_EXTENSIONS = set(['csv'])

# --- { CLUSTERING } ---
@app.route('/clustering/parametros',methods=['GET', 'POST'])

def c_upload():
    if request.method == 'POST':
        #Importamos los datos
        c_file = request.files["c_csvfile"]

        # Obtenemos el nombre del archivo
        filename = secure_filename(c_file.filename)

        # Separamos el nombre del archivo
        file = filename.split('.')
        
        if file[1] in ALLOWED_EXTENSIONS:
            df = pd.read_csv(c_file)

            # Borramos las filas con valores nulos
            df = df.dropna()

            # Clustering jerárquico

            MatrizVariables = df.iloc[:, [3, 5, 6, 7, 10, 11]].values  #iloc para seleccionar filas y columnas
                    


            estandarizar = StandardScaler()                         # Se instancia el objeto StandardScaler o MinMaxScaler 
            MEstandarizada = estandarizar.fit_transform(MatrizVariables) # Se calculan la media y desviación y se escalan los datos
            
            # Obtenemos el árbol jerárquico
            Arbol = shc.dendrogram(shc.linkage(MEstandarizada, method='complete', metric='euclidean'))

            #Se crean las etiquetas de los elementos en los clusters
            MJerarquico = AgglomerativeClustering(n_clusters= int(request.form['numclusters']), linkage='complete', affinity='euclidean')
            MJerarquico.fit_predict(MEstandarizada)

            df['clusterH'] = MJerarquico.labels_
        

            CentroidesH = df.groupby(['clusterH'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
            

            # Clustering particional
            #Definición de k clusters para K-means
            #Se utiliza random_state para inicializar el generador interno de números aleatorios
            SSE = []
            for i in range(2, 12):
                km = KMeans(n_clusters=i, random_state=0)
                km.fit(MEstandarizada)
                SSE.append(km.inertia_)

            kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
            
            
            #Se crean las etiquetas de los elementos en los clusters
            MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
            MParticional.predict(MEstandarizada)
            
            df['clusterP'] = MParticional.labels_

            CentroidesP = df.groupby(['clusterP'])['Texture', 'Area', 'Smoothness', 'Compactness', 'Symmetry', 'FractalDimension'].mean()
            return render_template('clustering_resultados.html',
                    tablesJ=[ CentroidesH.to_html(classes='data')], titlesJ=CentroidesH.columns.values,
                    tablesP=[ CentroidesP.to_html(classes='data')], titlesP=CentroidesP.columns.values)
            
    else:   
        
        return render_template('clustering_parametros.html')
    

# --- { METRICAS DE DISTANCIA } ---

@app.route('/metricas_distancia/parametros',methods=['GET', 'POST'])
def md_upload():
    if request.method == 'POST':
        #Importamos los datos
        md_file = request.files["md_csvfile"]

        # Obtenemos el nombre del archivo
        filename = secure_filename(md_file.filename)

        # Separamos el nombre del archivo
        file = filename.split('.')
        
        if file[1] in ALLOWED_EXTENSIONS:
            df = pd.read_csv(md_file)

            # Borramos las filas con valores nulos
            df = df.dropna()


            estandarizar = StandardScaler()                         # Se instancia el objeto StandardScaler o MinMaxScaler 
            MEstandarizada = estandarizar.fit_transform(df)         # Se calculan la media y desviación y se escalan los datos
            
            DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
            MEuclidiana = pd.DataFrame(DstEuclidiana)
            
            DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
            MChebyshev = pd.DataFrame(DstChebyshev)
            
            DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
            MManhattan = pd.DataFrame(DstManhattan)

            DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
            MMinkowski = pd.DataFrame(DstMinkowski)
            
            
            return render_template('metricas_distancia_resultados.html',
                tablesE=[ MEuclidiana.to_html(classes='data')], titlesE=MEuclidiana.columns.values,
                tablesC=[ MChebyshev.to_html(classes='data')], titlesC=MChebyshev.columns.values,
                tablesMan=[ MManhattan.to_html(classes='data')], titlesMan=MManhattan.columns.values,
                tablesMin=[ MMinkowski.to_html(classes='data')], titlesMin=MMinkowski.columns.values)
            
    else:   
        
        return render_template('metricas_distancia_parametros.html')



















    

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
                    min_support = float(soporte)/100,
                    min_confidence = float(confianza)/100,
                    min_lift = float(elevacion))

            Resultados = list(Reglas)
            
            return render_template('reglas_asociacion_resultados.html',Resultados = Resultados,soporte = soporte, confianza = confianza, elevacion = elevacion, size = len(Resultados))
    else:
        
        return render_template('reglas_asociacion_parametros.html')




# --- { Redireccionando } ---
@app.route('/clustering')
def clustering():
    return render_template('clustering.html')

@app.route('/reglas_asociacion')
def ra():
    return render_template('reglas_asociacion.html')

@app.route('/metricas_distancia')
def md():
    return render_template('metricas_distancia.html')

@app.route('/')
def home():
    return render_template('index.html')

# --- { Main } ---
if __name__ == '_main_':
    
    app.run(host='0.0.0.0', port=80)
    



