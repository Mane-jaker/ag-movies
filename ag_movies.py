import pandas as pd
import numpy as np
import random
import tkinter as tk
from tkinter import Toplevel, ttk, messagebox, scrolledtext
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Cargar el dataset
file_path = 'movie_dataset.csv'
df = pd.read_csv(file_path)

# Seleccionar columnas relevantes
columnas_relevantes = ['title', 'genres', 'cast', 'director', 'release_date', 'vote_average', 'runtime']
df = df[columnas_relevantes]

def preprocesar_datos(df):
    df['genres'] = df['genres'].fillna('').astype(str)  # Reemplaza NaN con una cadena vacía
    df['cast'] = df['cast'].fillna('').astype(str)  # Reemplaza NaN con una cadena vacía
    df['director'] = df['director'].fillna('').astype(str)  # Reemplaza NaN con una cadena vacía
    return df

df = preprocesar_datos(df)

def formatear_entrada(lista):
    return [palabra.strip().title() for palabra in lista]

def evaluar_aptitud_poblacion(poblacion, preferencias):
    evaluacion_aptitud = []
    for lista in poblacion:
        aptitud_lista = 0
        for pelicula in lista:
            aptitud = 0
            if any(genre.strip() in pelicula['genres'].split('|') for genre in preferencias['genres']):
                aptitud += 2
            if any(actor.strip() in pelicula['cast'].split('|') for actor in preferencias['actors']):
                aptitud += 1
            if pelicula['director'].strip() in preferencias['directors']:
                aptitud += 1
            aptitud -= abs(preferencias['duration'] - pelicula['runtime']) / 100
            aptitud_lista += aptitud
        evaluacion_aptitud.append({"individual": lista, "fitness": aptitud_lista})
    return evaluacion_aptitud

def generar_poblacion(df, tamano_poblacion, num_peliculas):
    poblacion = []
    for _ in range(tamano_poblacion):
        indices = random.sample(range(len(df)), num_peliculas)
        muestra = df.iloc[indices].to_dict('records')
        poblacion.append(muestra)
    return poblacion

def seleccion(poblacion, crossover_prob):
    padres = []
    for i in range(len(poblacion)):
        for j in range(i + 1, len(poblacion)):
            if random.random() <= crossover_prob:
                padres.append((poblacion[i], poblacion[j]))
    return padres

def cruzamiento(padres):
    nueva_poblacion = []
    for padre1, padre2 in padres:
        if len(padre1) == 0 or len(padre2) == 0:
            continue
        punto_cruce = random.randint(1, min(len(padre1), len(padre2)) - 1)
        hijo1 = padre1[:punto_cruce] + padre2[punto_cruce:]
        hijo2 = padre2[:punto_cruce] + padre1[punto_cruce:]
        nueva_poblacion.extend([hijo1, hijo2])
    return nueva_poblacion

def mutacion(poblacion, df, mutacion_prob):
    for lista in poblacion:
        if random.random() < mutacion_prob:
            indice = random.randint(0, len(lista) - 1)
            nuevo_indice = random.randint(0, len(df) - 1)
            lista[indice] = df.iloc[nuevo_indice].to_dict()
    return poblacion

def ordenar_poblacion_por_aptitud(evaluacion_aptitud):
    return sorted(evaluacion_aptitud, key=lambda x: x['fitness'], reverse=True)

def poda(evaluacion_aptitud, tamano_poblacion_maxima, pelicula_no_deseada):
    poblacion_unica = []
    peliculas_vistas = set()

    def tiene_peliculas_duplicadas(lista):
        titulos = set()
        for pelicula in lista:
            if pelicula['title'] in titulos:
                return True
            titulos.add(pelicula['title'])
        return False

    for entrada in evaluacion_aptitud:
        lista_tuple = tuple(tuple(pelicula.items()) for pelicula in entrada['individual'])
        if lista_tuple not in peliculas_vistas:
            # Verificar si la lista contiene películas duplicadas
            if not tiene_peliculas_duplicadas(entrada['individual']):
                peliculas_vistas.add(lista_tuple)
                poblacion_unica.append(entrada)

    poblacion_filtrada = [entrada for entrada in poblacion_unica if not any(pelicula_no_deseada in pelicula['title'] for pelicula in entrada['individual'])]

    poblacion_ordenada = ordenar_poblacion_por_aptitud(poblacion_filtrada)

    return [entrada['individual'] for entrada in poblacion_ordenada[:tamano_poblacion_maxima]]


def algoritmo_genetico(df, preferencias, generaciones, crossover_prob, mutacion_prob, tamano_poblacion, num_peliculas, tamano_poblacion_maxima, pelicula_no_deseada):
    historial_aptitud = {'mejor': [], 'promedio': [], 'peor': []}
    
    poblacion = generar_poblacion(df, tamano_poblacion, num_peliculas)
    
    for _ in range(generaciones):
        padres = seleccion(poblacion, crossover_prob)
        hijos = cruzamiento(padres)
        nuevos_hijos = mutacion(hijos, df, mutacion_prob)
        nueva_poblacion = list(poblacion) + list(nuevos_hijos)
        fitness_evaluation = evaluar_aptitud_poblacion(nueva_poblacion, preferencias)
        poblacion = poda(fitness_evaluation, tamano_poblacion_maxima, pelicula_no_deseada) 
        last_fitness = evaluar_aptitud_poblacion(poblacion, preferencias)
        fitness_values = [individual['fitness'] for individual in last_fitness]
        historial_aptitud['mejor'].append(max(fitness_values))
        historial_aptitud['promedio'].append(np.mean(fitness_values))
        historial_aptitud['peor'].append(min(fitness_values))   
    return poblacion, last_fitness, historial_aptitud

def mostrar_grafica(aptitudes):
    ventana_grafica = Toplevel(root)
    ventana_grafica.title("Evolución de la Aptitud")
    ventana_grafica.geometry("800x600")

    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    
    generaciones = list(range(len(aptitudes['mejor'])))
    
    ax.plot(generaciones, aptitudes['mejor'], label='Mejor Aptitud', color='green')
    ax.plot(generaciones, aptitudes['promedio'], label='Aptitud Promedio', color='blue')
    ax.plot(generaciones, aptitudes['peor'], label='Peor Aptitud', color='red')
    
    ax.set_xlabel('Generación')
    ax.set_ylabel('Aptitud')
    ax.set_title('Evolución de la Aptitud a lo largo de las Generaciones')
    ax.legend()
    
    canvas = FigureCanvasTkAgg(fig, master=ventana_grafica)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Función para ejecutar el algoritmo a partir de las preferencias ingresadas
def ejecutar_algoritmo():
    try:
        genres = formatear_entrada(genre_entry.get().split(','))
        actors = actor_entry.get().split(',')
        directors = director_entry.get().split(',')
        duration = int(duration_entry.get())

        peliculas_favoritas = favorite_movie_entry.get().split(',')
        pelicula_no_deseada = unwanted_movie_entry.get()

        generaciones = int(generations_entry.get())
        crossover_prob = float(crossover_prob_entry.get())
        mutacion_prob = float(mutation_prob_entry.get())
        tamano_poblacion_inicial = int(initial_population_size_entry.get())
        tamano_poblacion_maxima = int(max_population_size_entry.get())
        num_peliculas = int(movies_per_list_entry.get())

        for pelicula_favorita in peliculas_favoritas:
            pelicula_favorita = pelicula_favorita.strip().title()
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
            'duration': duration
        }
        
        poblacion, resultados, historial_aptitud = algoritmo_genetico(
            df,
            preferencias,
            generaciones,
            crossover_prob,
            mutacion_prob,
            tamano_poblacion_inicial,
            num_peliculas,
            tamano_poblacion_maxima,
            pelicula_no_deseada
        )
        
        # Mostrar resultados
        resultado_text.delete(1.0, tk.END)
        if poblacion:
            mejor_lista = poblacion[0]  # Asumiendo que la primera lista es la de mayor aptitud
            resultado_text.insert(tk.END, f"Mejor Lista de Películas:\n\n")
            for pelicula in mejor_lista:
                resultado_text.insert(tk.END, f"Título: {pelicula['title']}\n")
                resultado_text.insert(tk.END, f"Elenco: {pelicula['cast']}\n")
                resultado_text.insert(tk.END, f"Director: {pelicula['director']}\n")
                resultado_text.insert(tk.END, "-"*40 + "\n")
        else:
            resultado_text.insert(tk.END, "No se encontraron listas de películas.")

        mostrar_grafica(historial_aptitud)

    except ValueError as e:
        messagebox.showerror("Error de Entrada", f"Verifica que todos los campos estén correctamente llenos.\nError: {e}")

# Configuración de la ventana principal
root = tk.Tk()
root.title("Algoritmo Genético para Recomendaciones de Películas")
root.geometry("900x600")

# Estilo de ttk
style = ttk.Style()
style.configure('TButton', background='lightblue')
style.configure('TLabel', font=('Arial', 10))
style.configure('TEntry', font=('Arial', 10))

# Frames
frame_inputs = ttk.Frame(root, padding="10")
frame_inputs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

frame_results = ttk.Frame(root, padding="10")
frame_results.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Entradas de datos con valores predeterminados
ttk.Label(frame_inputs, text="Géneros (separados por coma):").grid(row=0, column=0, sticky=tk.W)
genre_entry = ttk.Entry(frame_inputs, width=50)
genre_entry.grid(row=0, column=1)
genre_entry.insert(0, 'Family')

ttk.Label(frame_inputs, text="Actores (separados por coma):").grid(row=1, column=0, sticky=tk.W)
actor_entry = ttk.Entry(frame_inputs, width=50)
actor_entry.grid(row=1, column=1)
actor_entry.insert(0, 'Johnny Depp')

ttk.Label(frame_inputs, text="Directores (separados por coma):").grid(row=2, column=0, sticky=tk.W)
director_entry = ttk.Entry(frame_inputs, width=50)
director_entry.grid(row=2, column=1)
director_entry.insert(0, 'Gore Verbinski')

ttk.Label(frame_inputs, text="Duración deseada (minutos):").grid(row=3, column=0, sticky=tk.W)
duration_entry = ttk.Entry(frame_inputs, width=50)
duration_entry.grid(row=3, column=1)
duration_entry.insert(0, '110')

ttk.Label(frame_inputs, text="Películas Favoritas (separadas por coma):").grid(row=4, column=0, sticky=tk.W)
favorite_movie_entry = ttk.Entry(frame_inputs, width=50)
favorite_movie_entry.grid(row=4, column=1)
favorite_movie_entry.insert(0, 'Megamind, Pirates of the Caribbean')

ttk.Label(frame_inputs, text="Película No Deseada (opcional):").grid(row=5, column=0, sticky=tk.W)
unwanted_movie_entry = ttk.Entry(frame_inputs, width=50)
unwanted_movie_entry.grid(row=5, column=1)
unwanted_movie_entry.insert(0, 'The Good Dinosaur')

ttk.Label(frame_inputs, text="Número de Generaciones:").grid(row=6, column=0, sticky=tk.W)
generations_entry = ttk.Entry(frame_inputs, width=50)
generations_entry.grid(row=6, column=1)
generations_entry.insert(0, '100')

ttk.Label(frame_inputs, text="Probabilidad de Cruce:").grid(row=7, column=0, sticky=tk.W)
crossover_prob_entry = ttk.Entry(frame_inputs, width=50)
crossover_prob_entry.grid(row=7, column=1)
crossover_prob_entry.insert(0, '0.8')

ttk.Label(frame_inputs, text="Probabilidad de Mutación:").grid(row=8, column=0, sticky=tk.W)
mutation_prob_entry = ttk.Entry(frame_inputs, width=50)
mutation_prob_entry.grid(row=8, column=1)
mutation_prob_entry.insert(0, '0.3')

ttk.Label(frame_inputs, text="Tamaño de la Población Inicial:").grid(row=9, column=0, sticky=tk.W)
initial_population_size_entry = ttk.Entry(frame_inputs, width=50)
initial_population_size_entry.grid(row=9, column=1)
initial_population_size_entry.insert(0, '10')

ttk.Label(frame_inputs, text="Tamaño Máximo de la Población:").grid(row=10, column=0, sticky=tk.W)
max_population_size_entry = ttk.Entry(frame_inputs, width=50)
max_population_size_entry.grid(row=10, column=1)
max_population_size_entry.insert(0, '20')

ttk.Label(frame_inputs, text="Número de Películas por Lista:").grid(row=11, column=0, sticky=tk.W)
movies_per_list_entry = ttk.Entry(frame_inputs, width=50)
movies_per_list_entry.grid(row=11, column=1)
movies_per_list_entry.insert(0, '5')

# Botón para ejecutar el algoritmo
ejecutar_button = ttk.Button(frame_inputs, text="Ejecutar Algoritmo", command=ejecutar_algoritmo)
ejecutar_button.grid(row=12, column=0, columnspan=2, pady=10)

# Área de resultados
resultado_text = scrolledtext.ScrolledText(frame_results, width=50, height=30)
resultado_text.pack(fill=tk.BOTH, expand=True)

# Ejecutar la interfaz
root.mainloop()



entradas = {
    'Family'
    'Johnny Depp'
    'Gore Verbinski'
    '110'
    'Megamind'
    'The Good Dinosaur'
    '100'
    '0.8'
    '0.3'
    '10'
    '20'
    '5'
}
