import csv
import pandas as pd
import math
from decimal import getcontext, Decimal
from math import sqrt
import matplotlib.pyplot as plt
import time

path="datasets/ml-20m/"


'''
#Analisis 
movies = pd.read_csv(path+'movies.csv', sep=',')
print("***********MOVIES.CSV***************")
print("************************************")
print("\nPrimeros registros movies.csv\n")
print(movies.head())
print("\nDescribe movies.csv\n")
print(movies.describe())
print("\nShape movies.csv\n")
print(movies.shape)
print("V\nerificamos nulos movies.csv\n")
print(movies.isnull().any())


print("************************************")
print("***********RATING.CSV***************")
print("************************************")
print("Primeros registros ratings.csv")
ratings = pd.read_csv(path+'ratings.csv', sep=',').drop('timestamp', axis=1)
print(ratings.head())
print("\nDescribe ratings.csv\n")
print(ratings.describe())
print("\nCorrelacion ratings.csv\n")
print(ratings.corr())
print("\nShape ratings.csv\n")
print(ratings.shape)
print("\nVerificamos nulos ratings.csv\n")
print(ratings.isnull().any())


'''
# Diccionarios globales para convertir IDs
username_to_id = {}
userid_to_name = {}
productid_to_name = {}

def convertProductIDToName(id, productid_to_name):

    if id in productid_to_name:
        return productid_to_name[id]
    else:
        return id
    
    
#METRICAS

def manhattan(rating1, rating2):
    distance = 0
    for key in rating1:
        if key in rating2:
            distance += abs(rating1[key] - rating2[key])
    return distance

def euclidiana(rating1, rating2):
    distance = 0
    for key in rating1:
        if key in rating2:
            distance += math.pow(rating1[key] - rating2[key], 2)
    return sqrt(distance)

def pearson(rating1, rating2):
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x2 = 0
    sum_y2 = 0
    n = 0
    for key in rating1:
        if key in rating2:
            n += 1
            x = rating1[key]
            y = rating2[key]
            sum_xy += x * y
            sum_x += x
            sum_y += y
            sum_x2 += pow(x, 2)
            sum_y2 += pow(y, 2)
    if n == 0:
        return 0

    denominator = (sqrt(sum_x2 - pow(sum_x, 2) / n) * sqrt(sum_y2 - pow(sum_y, 2) / n))
    if denominator == 0:
        return 0

    producto = sum_xy - (sum_x * sum_y) / n
    res = producto / denominator

    if res >= 1.0 or producto == denominator:
        return 1.0
    else:
        return res


def coseno(rating1, rating2):
    p_punto = 0
    logx = 0
    logy = 0
    for key in rating1:
        if key in rating2:
            p_punto += (rating1[key] * rating2[key])
            logx += pow(rating1[key], 2)
            logy += pow(rating2[key], 2)

    denominator = sqrt(logx) * sqrt(logy)

    if denominator == 0:
        return 0

    getcontext().prec = 5
    res = p_punto / denominator
    if res >= 1.0 or p_punto == denominator:
        return 1.0
    else:
        return res


#calculo de los vecinos mas cercanos
def computeNearestNeighbor(data, username, funcion):
    distances = []
    for instance in data:
        if instance != username:
            distance = funcion(data[username], data[instance])
            distances.append((instance, distance))

    # Ordena los vecinos por la distancia calculada
    distances.sort(key=lambda artistTuple: artistTuple[1], reverse=True)
    return distances

#uso de KNN para las recomendaciones
def KNN(data, user, k, funcion, n):
    recommendations = {}

    # obtenemos los vecinos más cercanos
    start_time_distancia= time.time()
    nearest = computeNearestNeighbor(data, user, funcion)
    tiempo_calcular_distancia = round((time.time() - start_time_distancia),6)
    print("--- Calcular distancia: %s segundos ---" % tiempo_calcular_distancia)
    
    # obtenemos las valoraciones del usuario
    start_time_knn= time.time()
    user_ratings = data[user]

    # Suma las distancias totales de los k vecinos más cercanos
    total_distance = 0.0
    for i in range(k):
        total_distance += nearest[i][1]

    # Calculamos las recomendaciones ponderadas
    for i in range(k):
        weight = nearest[i][1] / total_distance
        name = nearest[i][0]
        neighbor_ratings = data[name]

        for artist in neighbor_ratings:
            if artist not in user_ratings:
                if artist not in recommendations:
                    recommendations[artist] = (neighbor_ratings[artist] * weight)
                else:
                    recommendations[artist] += (neighbor_ratings[artist] * weight)
    
    tiempo_knn = round((time.time() - start_time_knn),6)
    print("--- Calculo KNN: %s segundos ---" % tiempo_knn)
    

    # convertimos las recomendaciones a una lista de tuplas y las ordena
    recommendations = list(recommendations.items())
    recommendations = [(convertProductIDToName(k, productid_to_name), v) for (k, v) in recommendations]

    recommendations.sort(key=lambda artistTuple: artistTuple[1], reverse=True)

    return recommendations[:n], tiempo_calcular_distancia, tiempo_knn

#mostrar las valoraciones de un usuario, n = num de valoracion, data= diccionario dato y valoracion
def user_ratings(data, id, n):

    print("Ratings for " + userid_to_name[id])
    ratings = data[id]
    print(len(ratings))
    ratings = list(ratings.items())

    # Convierte los IDs de productos a nombres
    ratings = [(convertProductIDToName(k, productid_to_name), v) for (k, v) in ratings]

    # Ordena las valoraciones de mayor a menor
    ratings.sort(key=lambda artistTuple: artistTuple[1], reverse=True)
    ratings = ratings[:n]  # Toma las primeras n valoraciones

    # Imprime las valoraciones
    for rating in ratings:
        print("%s\t%i" % (rating[0], rating[1]))
        

# Clase para manejar la base de datos de MovieLens
def leerDatos():
    PATH_DATA = path+"ratings.csv"
    new_data_dict = {}
    with open(PATH_DATA, 'r') as data_file:
        data = csv.DictReader(data_file, delimiter=",")
        for row in data:
            item = new_data_dict.get(row["userId"], dict())
            item[row["movieId"]] = float(row["rating"])
            new_data_dict[row["userId"]] = item
    return new_data_dict

#buscador por codigo
def buscar_movie(ID):
    n = []
    nombre_archivo = path+"movies.csv"
    dataframe = pd.read_csv(nombre_archivo, delimiter=",")

    movieId = dataframe['movieId'].tolist()
    title = dataframe['title'].tolist()

    for u, m in zip(movieId, title):
        if u == ID:
            n = m
            break
    return n


def replace_keys(tuple_list):
    new_tuple_list = []

    # reemplazamos los codigos por el nombre de la pelicula
    for tuple_item in tuple_list:
        old_key, value = tuple_item
        new_key = buscar_movie(int(old_key))
        new_tuple_list.append((new_key, value))

    return new_tuple_list

###################EJECUCION########################
# Parametros 

metrica1 = 'pearson'
metrica2 = 'manhattan'
metrica3 = 'euclidiana'
metrica4 = 'coseno'
k = 5
umbral = 5
user = '13'

#Lectura de datos

print("\n\n**************TIEMPO EN EJECUCIONES**************")
start_time_lectura = time.time()
dataLocal=leerDatos()
tiempo_lectura_datos = round((time.time() - start_time_lectura),6)
print("--- Lectura de datos: %s segundos ---" % tiempo_lectura_datos)

#Ejecución KNN
start_time_knn = time.time()
resultados_ml = KNN(dataLocal, user, k, pearson, umbral)
tiempo_calculo_distancia = resultados_ml[1]
tiempo_knn = resultados_ml[2]
tiempo_total = round(tiempo_lectura_datos+tiempo_calculo_distancia+tiempo_knn,6)
print("--- Tiempo total: %s segundos ---" % tiempo_total)
print("**************RESULTADO**************")
print("**Para el usuario: "+user+",con un umbral de: "+str(umbral)+", metríca: "+metrica2+" la recomendación es: \n")
print(replace_keys(resultados_ml[0]))
print("**************Legible**************")
# Imprimir la lista formateada
formatted_movies = [f"{title}: {rating:.2f}" for title, rating in replace_keys(resultados_ml[0])]
for d_movie in formatted_movies:
    print(d_movie)