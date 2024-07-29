import pandas as pd
import random
import numpy as np
import tkinter as tk
from tkinter import simpledialog

# Cargar el dataset
file_path = 'C:/Users/angel/Documents/IA/AG dataset/movies/movie_dataset.csv'
df = pd.read_csv(file_path)

# Seleccionar columnas relevantes
columnas_relevantes = ['title', 'genres', 'cast', 'director', 'release_date', 'vote_average', 'runtime']
df = df[columnas_relevantes]

def preprocesar_datos(df):
    df['genres'] = df['genres'].fillna('')  # Reemplaza NaN con una cadena vacía
    df['genres'] = df['genres'].astype(str)  # Asegúrate de que todos los valores sean cadenas
    df['cast'] = df['cast'].fillna('')  # Reemplaza NaN con una cadena vacía
    df['cast'] = df['cast'].astype(str)  # Asegúrate de que todos los valores sean cadenas
    df['director'] = df['director'].fillna('')  # Reemplaza NaN con una cadena vacía
    df['director'] = df['director'].astype(str)  # Asegúrate de que todos los valores sean cadenas
    return df

df = preprocesar_datos(df)

# Formatear las entradas para que cada palabra comience con mayúscula
def formatear_entrada(lista):
    return [palabra.strip().title() for palabra in lista]

# Función para calcular la aptitud de una película basada en las preferencias del usuario
def calcular_aptitud(pelicula, preferencias):
    aptitud = 0
    if any(genre.strip() in pelicula['genres'].split('|') for genre in preferencias['genres']):
        aptitud += 1
    if any(actor.strip() in pelicula['cast'].split('|') for actor in preferencias['actors']):
        aptitud += 1
    if pelicula['director'].strip() in preferencias['directors']:
        aptitud += 1
    aptitud += pelicula['vote_average'] / 10
    aptitud -= abs(preferencias['year'] - int(pelicula['release_date'][:4])) / 100
    aptitud -= abs(preferencias['duration'] - pelicula['runtime']) / 100
    return aptitud

# Generar una población inicial
def generar_poblacion(df, tamano_poblacion, num_peliculas):
    poblacion = []
    for _ in range(tamano_poblacion):
        indices = random.sample(range(len(df)), num_peliculas)
        muestra = df.iloc[indices].to_dict('records')
        poblacion.append(muestra)
    return poblacion

# Evaluar la aptitud de la población
def evaluar_poblacion(poblacion, preferencias):
    return [sum(calcular_aptitud(pelicula, preferencias) for pelicula in lista) for lista in poblacion]

# Selección de padres basada en la probabilidad de cruce
def seleccion(poblacion, aptitudes, crossover_prob):
    padres = []
    for i in range(len(poblacion)):
        for j in range(i + 1, len(poblacion)):
            if np.random.random() <= crossover_prob:
                padres.append((poblacion[i], poblacion[j]))
    return padres

# Cruza de padres
def cruzamiento(padres):
    nueva_poblacion = []
    for padre1, padre2 in padres:
        punto_cruce = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto_cruce] + padre2[punto_cruce:]
        hijo2 = padre2[:punto_cruce] + padre1[punto_cruce:]
        nueva_poblacion.extend([hijo1, hijo2])
    return nueva_poblacion

# Mutación
def mutacion(poblacion, df, mutacion_prob):
    for lista in poblacion:
        if random.random() < mutacion_prob:
            indice = random.randint(0, len(lista) - 1)
            nuevo_indice = random.randint(0, len(df) - 1)
            lista[indice] = df.iloc[nuevo_indice].to_dict()
    return poblacion

# Ordenar la población por aptitud
def ordenar_poblacion(poblacion, aptitudes):
    poblacion_ordenada = sorted(zip(aptitudes, poblacion), key=lambda x: x[0], reverse=True)
    return [x[1] for x in poblacion_ordenada]

# Poda para mantener las mejores soluciones
def poda(poblacion, tamano_poblacion, pelicula_no_deseada):
    # Filtrar listas que no contengan la película no deseada
    poblacion_filtrada = [lista for lista in poblacion if not any(pelicula_no_deseada in pelicula['title'] for pelicula in lista)]
    return poblacion_filtrada[:tamano_poblacion]

# Función principal del AG
def algoritmo_genetico(df, preferencias, generaciones, crossover_prob, mutacion_prob, tamano_poblacion, num_peliculas, pelicula_no_deseada):
    print(preferencias)
    poblacion = generar_poblacion(df, tamano_poblacion, num_peliculas)
    for _ in range(generaciones):
        aptitudes = evaluar_poblacion(poblacion, preferencias)
        poblacion = ordenar_poblacion(poblacion, aptitudes)
        padres = seleccion(poblacion, aptitudes, crossover_prob)
        nueva_poblacion = cruzamiento(padres)
        nueva_poblacion = mutacion(nueva_poblacion, df, mutacion_prob)
        aptitudes_nueva = evaluar_poblacion(nueva_poblacion, preferencias)
        nueva_poblacion = ordenar_poblacion(nueva_poblacion, aptitudes_nueva)
        poblacion = poda(poblacion + nueva_poblacion, tamano_poblacion, pelicula_no_deseada)
    return poblacion

# Interfaz gráfica
def obtener_preferencias():
    root = tk.Tk()
    root.withdraw()
    
    genres = formatear_entrada(simpledialog.askstring("Input", "Enter your favorite genres (comma-separated):").split(','))
    actors = formatear_entrada(simpledialog.askstring("Input", "Enter your favorite actors (comma-separated):").split(','))
    directors = formatear_entrada(simpledialog.askstring("Input", "Enter your favorite directors (comma-separated):").split(','))

    year = simpledialog.askinteger("Input", "Enter your preferred year:")
    rating = simpledialog.askfloat("Input", "Enter your minimum rating (0-10):")
    duration = simpledialog.askinteger("Input", "Enter your preferred duration (in minutes):")

    pelicula_favorita = simpledialog.askstring("Input", "Enter your favorite movie:")
    pelicula_no_deseada = simpledialog.askstring("Input", "Enter a movie you don't want to see:")

    generaciones = simpledialog.askinteger("Input", "Enter the number of generations:")
    crossover_prob = simpledialog.askfloat("Input", "Enter the crossover probability (0-1):")
    mutacion_prob = simpledialog.askfloat("Input", "Enter the mutation probability (0-1):")
    tamano_poblacion = simpledialog.askinteger("Input", "Enter the initial population size:")
    num_peliculas = simpledialog.askinteger("Input", "Enter the number of movies per list (individual):")

    root.destroy()

    # Obtener detalles de la película favorita
    if pelicula_favorita:
        pelicula_favorita = pelicula_favorita.title()
        pelicula = df[df['title'].str.contains(pelicula_favorita, case=False, na=False)]
        if not pelicula.empty:
            pelicula = pelicula.iloc[0]
            genres.extend(pelicula['genres'].split('|'))
            actors.extend(pelicula['cast'].split('|'))
            directors.append(pelicula['director'])

    preferencias = {
        'genres': genres,
        'actors': actors,
        'directors': directors,
        'year': year,
        'rating': rating,
        'duration': duration
    }

    return preferencias, generaciones, crossover_prob, mutacion_prob, tamano_poblacion, num_peliculas, pelicula_no_deseada

# Obtener preferencias y parámetros del usuario
preferencias, generaciones, crossover_prob, mutacion_prob, tamano_poblacion, num_peliculas, pelicula_no_deseada = obtener_preferencias()

# Ejecutar el algoritmo genético
poblacion_final = algoritmo_genetico(df, preferencias, generaciones, crossover_prob, mutacion_prob, tamano_poblacion, num_peliculas, pelicula_no_deseada)

# Imprimir los resultados
for i, lista in enumerate(poblacion_final):
    print(f"Lista {i+1}:")
    for pelicula in lista:
        print(pelicula)
    print()
