                        # --- { IMPORTAMOS LAS BIBLIOTECAS NECESARIAS } ----

# --- Para el manejo de imagénes y archivos de la página 
import os
import base64
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64
from werkzeug.utils import secure_filename

# --- Para la manipulación, análisis de datos y generación de gráficas
import pandas as pd               
import numpy as np                 
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D

# --- Para el redireccionamiento de flask
from flask import Flask, redirect, render_template,request, url_for 

# --- Para las reglas de asociacion
from apyori import apriori as ap    

# --- Para el cálculo de distancias
from scipy.spatial.distance import cdist    
from scipy.spatial import distance  

# --- Para estandarizar los datos
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

# --- Para visualizar los datos basados en matplotlib
import seaborn as sns             # Para la visualización de datos basado en matplotlib

# --- Para clustering jerárquico y particional
import scipy.cluster.hierarchy as shc 
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans 
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator 

# --- Para regresión logística
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import utils

# --- Para validación de la clasificación
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# --- Para árboles de decisión
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import model_selection

# ---------------------------- { AQUÍ COMIENZA LA PARTE DE FLASK } ----------------------------
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['csv'])

                                # ---{ ÁRBOLES DE DECISIÓN } ---
@app.route('/arboles_decision/resultados',methods=['GET', 'POST'])
def arboles_decision():
    if request.method == 'POST':
        #Importamos los datos
        ad_file = request.files["ad_csvfile"]

        # Obtenemos el nombre del archivo
        filename = secure_filename(ad_file.filename)

        # Separamos el nombre del archivo
        file = filename.split('.')
        
        if file[1] in ALLOWED_EXTENSIONS:
            df = pd.read_csv(ad_file)

            # Obtenemos las variables predictorias
            ints = [int(request.form['v_clase'])-1]

            for element in request.form.getlist('colum[]'):
                ints.append(int(element)-1)


            if int(request.form['tipoArbol']) == 1:
                
                # Variable clase
                Y = np.array(df[[df.columns[int(request.form['v_clase'])-1]]])

                # Eliminamos las columnas elegidas
                df = df.drop(df.columns[ints], axis='columns') 

                # Variables predictoras
                X = np.array(df[df.columns.values.tolist()])

                # Separamos los datos
                X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, 
                                                                    test_size = 0.2, 
                                                                    random_state = 1234, 
                                                                    shuffle = True)

                # Entrenamos el modelo
                PronosticoAD = DecisionTreeRegressor(random_state=0)
                PronosticoAD.fit(X_train, Y_train)

                #Se genera el pronóstico
                Y_Pronostico = PronosticoAD.predict(X_test)

                score = r2_score(Y_test, Y_Pronostico)
            
                return render_template('arboles_decision_resultados.html', score = score, 
                    predictoras = df.columns.values.tolist(), tam = len(df.columns.values.tolist())-1)
            
    else:   
        
        return render_template('arboles_decision_resultados.html')



@app.route('/arboles_decision/parametros',methods=['GET', 'POST'])
def ad_upload():
    if request.method == 'POST':
        #Importamos los datos
        ad_file = request.files["ad_csvfile"]

        # Obtenemos el nombre del archivo
        filename = secure_filename(ad_file.filename)

        # Separamos el nombre del archivo
        file = filename.split('.')
        
        if file[1] in ALLOWED_EXTENSIONS:
            df = pd.read_csv(ad_file)

            # Preparamos la imagen para mostrarla en la página web
            img = BytesIO()

            Corrdf = df.corr(method='pearson')
            plt.figure(figsize=(14,14))
            MatrizInf = np.triu(Corrdf)
            sns.heatmap(Corrdf, cmap='RdBu_r', annot=True, mask=MatrizInf)


            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')


            # Obtenemos el nombre de las columnas del dataframe para la selección manual
            columns_names = df.columns.values

            return render_template('arboles_decision_parametros.html',columns_names_list = list(columns_names), 
                plot_url=plot_url, filename = filename)
        
    else:   
        
        return render_template('arboles_decision_parametros.html')





# --- { REGRESIÓN LOGÍSTICA } ---     
@app.route('/regresion_logistica/pronostico',methods=['GET', 'POST'])
def regresion_logistica_pronostico():
    if request.method == 'POST':
        ints = []
        for element in request.form.getlist('colum[]'):


            var = request.form['variable1']
        # Obtenemos las variables predictorias
        #ints = [int(request.form['v_clase'])-1]
        #for element in request.form.getlist('colum[]'):
        #    ints.append(int(element)-1)

           
        return render_template('regresion_logistica_pronostico.html',var = var)





@app.route('/regresion_logistica/resultados',methods=['GET', 'POST'])
def regresion_logistica():
    if request.method == 'POST':
        #Importamos los datos
        rl_file = request.files["rl_csvfile"]

        # Obtenemos el nombre del archivo
        filename = secure_filename(rl_file.filename)

        # Separamos el nombre del archivo
        file = filename.split('.')
        
        if file[1] in ALLOWED_EXTENSIONS:
            df = pd.read_csv(rl_file)

            # Borramos las filas con valores nulos
            df = df.dropna()

            # Obtenemos la variable clase
            valores = df[df.columns[int(request.form['v_clase'])-1]].unique()

            # Convertimos valores cualitativos a cuantitativos de la variable clase
            df = df.replace({valores[0]: 0, valores[1]: 1})
            
            # Variable clase
            Y = np.array(df[[df.columns[int(request.form['v_clase'])-1]]])

            # Obtenemos las variables predictorias
            ints = [int(request.form['v_clase'])-1]

            for element in request.form.getlist('colum[]'):
                ints.append(int(element)-1)

            # Eliminamos las columnas elegidas
            df = df.drop(df.columns[ints], axis='columns') 

            # Variables predictoras
            X = np.array(df[df.columns.values.tolist()])
 
            # Regresión logística
            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, 
                                                                                test_size = 0.2, 
                                                                                random_state = 1234,
                                                                                shuffle = True)

            #Se entrena el modelo a partir de los datos de entrada
            Clasificacion = linear_model.LogisticRegression()
            Clasificacion.fit(X_train, Y_train)

            #Predicciones probabilísticas de los datos de prueba
            Probabilidad = Clasificacion.predict_proba(X_validation)
            pd.DataFrame(Probabilidad)
            
            #Predicciones con clasificación final 
            Predicciones = Clasificacion.predict(X_validation)
            pd.DataFrame(Predicciones)

            #Se calcula la exactitud promedio de la validación
            score = Clasificacion.score(X_validation, Y_validation)

            #Matriz de clasificación
            Y_Clasificacion = Clasificacion.predict(X_validation)
            Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), 
                                            Y_Clasificacion, 
                                            rownames=['Real'], 
                                            colnames=['Clasificación']) 
            
            #Reporte de la clasificación
            reporte = classification_report(Y_validation, Y_Clasificacion,output_dict='true')
            
            return render_template('regresion_logistica_resultados.html', score = score, 
                tables=[ Matriz_Clasificacion.to_html(classes='data')], titles=Matriz_Clasificacion.columns.values,
                valores = valores, reporte = reporte, predictoras = df.columns.values.tolist(), tam = len(df.columns.values.tolist())-1)
            
    else:   
        
        return render_template('regresion_logistica_resultados.html')


@app.route('/regresion_logistica/parametros',methods=['GET', 'POST'])
def rl_upload():
    if request.method == 'POST':
        #Importamos los datos
        rl_file = request.files["rl_csvfile"]

        # Obtenemos el nombre del archivo
        filename = secure_filename(rl_file.filename)

        # Separamos el nombre del archivo
        file = filename.split('.')
        
        if file[1] in ALLOWED_EXTENSIONS:
            df = pd.read_csv(rl_file)

            # Preparamos la imagen para mostrarla en la página web
            img = BytesIO()

            Corrdf = df.corr(method='pearson')
            plt.figure(figsize=(14,14))
            MatrizInf = np.triu(Corrdf)
            sns.heatmap(Corrdf, cmap='RdBu_r', annot=True, mask=MatrizInf)


            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')


            # Obtenemos el nombre de las columnas del dataframe para la selección manual
            columns_names = df.columns.values

            return render_template('regresion_logistica_parametros.html',columns_names_list = list(columns_names), 
                plot_url=plot_url, filename = filename)
        
    else:   
        
        return render_template('regresion_logistica_parametros.html')




# --- { CLUSTERING } ---  
@app.route('/clustering/resultados',methods=['GET', 'POST'])
def clustering():
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

            ints = []

            for element in request.form.getlist('colum[]'):
                ints.append(int(element)-1)

            # Eliminamos las columnas elegidas
            MatrizVariables = df.drop(df.columns[ints], axis='columns') 

            df = df.drop(df.columns[ints], axis='columns') 

            estandarizar = StandardScaler()                         # Se instancia el objeto StandardScaler o MinMaxScaler 
            MEstandarizada = estandarizar.fit_transform(MatrizVariables) # Se calculan la media y desviación y se escalan los datos

            if int(request.form['clustering']) == 1: # Clustering particional
                
                #Se utiliza random_state para inicializar el generador interno de números aleatorios
                SSE = []
                for i in range(2, 12):
                    km = KMeans(n_clusters=i, random_state=0)
                    km.fit(MEstandarizada)
                    SSE.append(km.inertia_)

                # Aplicando el método de la rodilla
                kl = KneeLocator(range(2, 12), SSE, curve="convex", direction="decreasing")
                
                
                #Se crean las etiquetas de los elementos en los clusters
                MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
                MParticional.predict(MEstandarizada)
                
                df['clusterP'] = MParticional.labels_

                #Obtenemos centroides
                CentroidesP = df.groupby(['clusterP']).mean()
                
                #Cantidad de elementos en los clusters
                clusters = df.groupby(['clusterP'])['clusterP'].count()
                
                
                # Preparamos las tablas y titulos para mostrarlos en pantalla
                tables = [CentroidesP.to_html(classes='data')]

                titles = CentroidesP.columns.values
                
                clustering = "particional"

                # Preparamos la imagen para mostrarla en la página web
                img = BytesIO()

                plt.figure(figsize=(10, 7))
                sns.scatterplot(MEstandarizada[:,0], MEstandarizada[:,1], data=df, hue='clusterP', s=50, palette="deep")
                plt.title('Clusters')
                plt.grid()


                plt.savefig(img, format='png')
                plt.close()
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                        
            elif int(request.form['clustering']) == 2: # Clustering jerárquico
                


                #Se crean las etiquetas de los elementos en los clusters
                MJerarquico = AgglomerativeClustering(n_clusters= int(request.form['numclusters']), linkage='complete', affinity='euclidean')
                MJerarquico.fit_predict(MEstandarizada)

                df['clusterH'] = MJerarquico.labels_
            
                #Obtenemos centroides
                CentroidesH = df.groupby(['clusterH']).mean()

                #Cantidad de elementos en los clusters
                clusters = df.groupby(['clusterH'])['clusterH'].count()

                tables = [ CentroidesH.to_html(classes='data')]

                titles = CentroidesH.columns.values

                clustering = "jerárquico"

                # Preparamos la imagen para mostrarla en la página web
                img = BytesIO()

                plt.figure(figsize=(10, 7))
                sns.scatterplot(MEstandarizada[:,0], MEstandarizada[:,1], data=df, hue='clusterH', s=50, palette="deep")
                plt.title('Clusters')
                plt.grid()


                plt.savefig(img, format='png')
                plt.close()
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            

            return render_template('clustering_resultados.html', clusters = clusters.to_list(),
                tables = tables, titles = titles, clustering = clustering,plot_url=plot_url)        
            
    else:   
        
        return render_template('clustering_resultados.html')



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

            # Preparamos la imagen para mostrarla en la página web
            img = BytesIO()

            Corrdf = df.corr(method='pearson')
            plt.figure(figsize=(14,14))
            MatrizInf = np.triu(Corrdf)
            sns.heatmap(Corrdf, cmap='RdBu_r', annot=True, mask=MatrizInf)


            plt.savefig(img, format='png')
            plt.close()
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')


            # Obtenemos el nombre de las columnas del dataframe para la selección manual
            columns_names = df.columns.values

            return render_template('clustering_parametros.html',columns_names_list = list(columns_names), 
                plot_url=plot_url, filename = filename)
            
    else:   
        
        return render_template('clustering_parametros.html')
    

# --- { METRICAS DE DISTANCIA } ---
@app.route('/metricas_distancia/resultados',methods=['GET', 'POST'])
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
            
            opcion = int(request.form['distancia'])
            
            if opcion == 1:
                DstEuclidiana = cdist(MEstandarizada, MEstandarizada, metric='euclidean')
                MEuclidiana = pd.DataFrame(DstEuclidiana)
                return render_template('metricas_distancia_resultados.html',  metrica = "Euclidiana",
                    tables=[ MEuclidiana.to_html(classes='data')], titles=MEuclidiana.columns.values)
            elif opcion == 2:
                DstChebyshev = cdist(MEstandarizada, MEstandarizada, metric='chebyshev')
                MChebyshev = pd.DataFrame(DstChebyshev)
                return render_template('metricas_distancia_resultados.html',  metrica = "Chebyshev",
                    tables=[ MChebyshev.to_html(classes='data')], titles=MChebyshev.columns.values)
            elif opcion == 3:
                DstManhattan = cdist(MEstandarizada, MEstandarizada, metric='cityblock')
                MManhattan = pd.DataFrame(DstManhattan)
                return render_template('metricas_distancia_resultados.html',  metrica = "Manhattan",
                    tables=[ MManhattan.to_html(classes='data')], titles=MManhattan.columns.values)
            elif opcion ==4:
                DstMinkowski = cdist(MEstandarizada, MEstandarizada, metric='minkowski', p=1.5)
                MMinkowski = pd.DataFrame(DstMinkowski)
                return render_template('metricas_distancia_resultados.html', metrica = "Minkowski",
                    tables=[ MMinkowski.to_html(classes='data')], titles=MMinkowski.columns.values)           
            return render_template('metricas_distancia_resultados.html')           
    
    else:      
        return render_template('metricas_distancia.html')

# --- { REGLAS DE ASOCIACIÓN } ---
@app.route('/reglas_asociacion/resultados',methods=['GET', 'POST'])
def ra_upload():
    if request.method == 'POST':
        #Importamos los datos
        ra_file = request.files["ra_csvfile"]

        # Obtenemos el nombre del archivo
        filename = secure_filename(ra_file.filename)

        # Separamos el nombre del archivo
        file = filename.split('.')
        
        if file[1] in ALLOWED_EXTENSIONS:

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
            
            return render_template('reglas_asociacion_resultados.html',filename = filename, Resultados = Resultados,soporte = soporte, confianza = confianza, elevacion = elevacion, size = len(Resultados))
    else:
        
        return render_template('reglas_asociacion.html')




# --- { Redireccionando } ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/reglas_asociacion')
def ra():
    return render_template('reglas_asociacion.html')

@app.route('/metricas_distancia')
def md():
    return render_template('metricas_distancia.html')

@app.route('/clustering')
def cl():
    return render_template('clustering.html')

@app.route('/regresion_logistica')
def rl():
    return render_template('regresion_logistica.html')

@app.route('/arboles_decision')
def ad():
    return render_template('arboles_decision.html')



# --- { Main } ---
if __name__ == '_main_':
    
    app.run(host='0.0.0.0', port=80)
    



