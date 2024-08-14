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

def poda(evaluacion_aptitud, tamano_poblacion_maxima, peliculas_no_deseadas, peliculas_favoritas):
    poblacion_unica = []
    peliculas_vistas = set()

    def tiene_peliculas_duplicadas(lista):
        titulos = set()
        for pelicula in lista:
            if pelicula['title'] in titulos:
                return True
            titulos.add(pelicula['title'])
        return False

    def contiene_peliculas_favoritas(lista):
        titulos_favoritos = set(pelicula['title'] for pelicula in peliculas_favoritas)
        for pelicula in lista:
            if pelicula['title'] in titulos_favoritos:
                return True
        return False

    for entrada in evaluacion_aptitud:
        lista_tuple = tuple(tuple(pelicula.items()) for pelicula in entrada['individual'])
        if lista_tuple not in peliculas_vistas:
            # Verificar si la lista contiene películas duplicadas
            if not tiene_peliculas_duplicadas(entrada['individual']):
                peliculas_vistas.add(lista_tuple)
                if not contiene_peliculas_favoritas(entrada['individual']):
                    poblacion_unica.append(entrada)

    def contiene_peliculas_no_deseadas(lista):
        for pelicula in lista:
            if any(pelicula_no_deseada in pelicula['title'] for pelicula_no_deseada in peliculas_no_deseadas):
                return True
        return False

    poblacion_filtrada = [entrada for entrada in poblacion_unica if not contiene_peliculas_no_deseadas(entrada['individual'])]

    poblacion_ordenada = ordenar_poblacion_por_aptitud(poblacion_filtrada)

    return [entrada['individual'] for entrada in poblacion_ordenada[:tamano_poblacion_maxima]]


def algoritmo_genetico(df, preferencias, generaciones, crossover_prob, mutacion_prob, tamano_poblacion, num_peliculas, tamano_poblacion_maxima, peliculas_no_deseadas, peliculas_favoritas):
    historial_aptitud = {'mejor': [], 'promedio': [], 'peor': []}
    
    poblacion = generar_poblacion(df, tamano_poblacion, num_peliculas)
    
    for _ in range(generaciones):
        padres = seleccion(poblacion, crossover_prob)
        hijos = cruzamiento(padres)
        nuevos_hijos = mutacion(hijos, df, mutacion_prob)
        nueva_poblacion = list(poblacion) + list(nuevos_hijos)
        fitness_evaluation = evaluar_aptitud_poblacion(nueva_poblacion, preferencias)
        poblacion = poda(fitness_evaluation, tamano_poblacion_maxima, peliculas_no_deseadas, peliculas_favoritas)
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

def ejecutar_algoritmo():
    try:
        # Obtener preferencias del formulario
        genres = formatear_entrada(genre_entry.get().split(','))
        actors = actor_entry.get().split(',')
        directors = director_entry.get().split(',')
        duration = int(duration_entry.get())

        # Obtener y procesar películas favoritas
        peliculas_favoritas = []
        for pelicula_favorita in favorite_movie_entry.get().split(','):
            pelicula_favorita = pelicula_favorita.strip().title()
            pelicula = df[df['title'].str.contains(pelicula_favorita, case=False, na=False)]
            if not pelicula.empty:
                pelicula = pelicula.iloc[0]
                peliculas_favoritas.append(pelicula)
                # Agregar género, actores y director de la película favorita a las preferencias
                genres.extend(pelicula['genres'].split('|'))
                actors.extend(pelicula['cast'].split('|'))
                directors.append(pelicula['director'])

        # Obtener película no deseada
        peliculas_no_deseadas = unwanted_movie_entry.get().split(',')
 
        # Obtener parámetros del formulario
        generaciones = int(generations_entry.get())
        crossover_prob = float(crossover_prob_entry.get())
        mutacion_prob = float(mutacion_prob_entry.get())
        tamano_poblacion_inicial = int(initial_population_size_entry.get())
        tamano_poblacion_maxima = int(max_population_size_entry.get())
        num_peliculas = int(movies_per_list_entry.get())

        # Crear diccionario de preferencias
        preferencias = {
            'genres': genres,
            'actors': actors,
            'directors': directors,
            'duration': duration
        }
        
        # Ejecutar el algoritmo genético
        poblacion, resultados, historial_aptitud = algoritmo_genetico(
            df,
            preferencias,
            generaciones,
            crossover_prob,
            mutacion_prob,
            tamano_poblacion_inicial,
            num_peliculas,
            tamano_poblacion_maxima,
            peliculas_no_deseadas,
            peliculas_favoritas
        )
        
        # Mostrar resultados
        lista_resultados.delete(1.0, tk.END)
        if poblacion:
            mejor_lista = poblacion[0]  # Asumiendo que la primera lista es la de mayor aptitud
            lista_resultados.insert(tk.END, f"Mejor Lista de Películas:\n\n")
            for pelicula in mejor_lista:
                lista_resultados.insert(tk.END, f"Título: {pelicula['title']}\n")
                lista_resultados.insert(tk.END, f"Generos: {pelicula['genres']}\n")
                lista_resultados.insert(tk.END, f"Elenco: {pelicula['cast']}\n")
                lista_resultados.insert(tk.END, f"Director: {pelicula['director']}\n")
                lista_resultados.insert(tk.END, "-"*40 + "\n")
        else:
            lista_resultados.insert(tk.END, "No se encontraron listas de películas.")

        mostrar_grafica(historial_aptitud)

    except ValueError as e:
        messagebox.showerror("Error de Entrada", f"Verifica que todos los campos estén correctamente llenos.\nError: {e}")

        
# Clase AutocompleteEntry
class AutocompleteEntry(ttk.Entry):
    def __init__(self, lista, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lista = sorted(lista)
        self.var = self["textvariable"]
        if self.var == '':
            self.var = self["textvariable"] = tk.StringVar()
        self.var.trace_add("write", self.changed)
        self.bind("<Return>", self.selection)
        self.bind("<Up>", self.move_up)
        self.bind("<Down>", self.move_down)
        self.lb_up = False

    def changed(self, name, index, mode):
        if self.var.get() == '':
            if self.lb_up:
                self.lb.destroy()
                self.lb_up = False
        else:
            words = self.comparison()
            if words:
                if not self.lb_up:
                    self.lb = tk.Listbox(width=self["width"])
                    self.lb.bind("<Double-Button-1>", self.selection)
                    self.lb.bind("<Return>", self.selection)
                    self.lb.bind("<Escape>", self.escape)
                    self.lb.place(x=self.winfo_x(), y=self.winfo_y() + self.winfo_height())
                    self.lb_up = True
                self.lb.delete(0, tk.END)
                for w in words:
                    self.lb.insert(tk.END, w)
            else:
                if self.lb_up:
                    self.lb.destroy()
                    self.lb_up = False

    def selection(self, event):
        if self.lb_up:
            self.var.set(self.lb.get(tk.ACTIVE))
            self.lb.destroy()
            self.lb_up = False
            self.icursor(tk.END)

    def escape(self, event):
        if self.lb_up:
            self.lb.destroy()
            self.lb_up = False

    def move_up(self, event):
        if self.lb_up:
            if self.lb.curselection() == ():
                index = '0'
            else:
                index = self.lb.curselection()[0]
            if index != '0':
                self.lb.selection_clear(first=index)
                index = str(int(index) - 1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def move_down(self, event):
        if self.lb_up:
            if self.lb.curselection() == ():
                index = '-1'
            else:
                index = self.lb.curselection()[0]
            if index != tk.END:
                self.lb.selection_clear(first=index)
                index = str(int(index) + 1)
                self.lb.selection_set(first=index)
                self.lb.activate(index)

    def comparison(self):
        pattern = self.var.get()
        return [w for w in self.lista if w.lower().startswith(pattern.lower())]

# Configuración de la ventana principal
root = tk.Tk()
root.title("Algoritmo Genético para Recomendaciones de Películas")
root.geometry("1200x650")

# Estilo de ttk
style = ttk.Style()
style.configure('TButton', background='lightblue')
style.configure('TLabel', font=('Arial', 10), anchor='w')
style.configure('TEntry', font=('Arial', 10))

# Frames
frame_inputs = ttk.Frame(root, padding="10")
frame_inputs.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

frame_results = ttk.Frame(root, padding="10")
frame_results.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Labels y Entry Widgets en frame_inputs con valores predeterminados
def add_input_row(frame, label_text, widget, default_value, row):
    ttk.Label(frame, text=label_text, anchor='w').grid(row=row, column=0, sticky='w', pady=2)
    widget.grid(row=row, column=1, sticky='w', pady=2)
    if default_value:
        widget.insert(0, default_value)

# Crear los widgets de entrada
genre_entry = AutocompleteEntry(df['genres'].str.split('|').explode().unique(), frame_inputs, width=50)
actor_entry = AutocompleteEntry(df['cast'].str.split('|').explode().unique(), frame_inputs, width=50)
director_entry = AutocompleteEntry(df['director'].unique(), frame_inputs, width=50)
duration_entry = ttk.Entry(frame_inputs, width=50)
favorite_movie_entry = AutocompleteEntry(df['title'].unique(), frame_inputs, width=50)
unwanted_movie_entry = AutocompleteEntry(df['title'].unique(), frame_inputs, width=50)
generations_entry = ttk.Entry(frame_inputs, width=50)
crossover_prob_entry = ttk.Entry(frame_inputs, width=50)
mutacion_prob_entry = ttk.Entry(frame_inputs, width=50)
initial_population_size_entry = ttk.Entry(frame_inputs, width=50)
max_population_size_entry = ttk.Entry(frame_inputs, width=50)
movies_per_list_entry = ttk.Entry(frame_inputs, width=50)

# Añadir las filas de entrada al frame
inputs = [
    ("Géneros (separados por comas):", genre_entry, "Action,Comedy"),
    ("Actores (separados por comas):", actor_entry, "Jhonny Depp"),
    ("Directores (separados por comas):", director_entry, "Christopher Nolan"),
    ("Duración preferida (minutos):", duration_entry, "120"),
    ("Películas favoritas (separadas por comas):", favorite_movie_entry, "Inception,Interstellar"),
    ("Películas no deseadas  (separadas por comas):", unwanted_movie_entry, "Titanic"),
    ("Número de generaciones:", generations_entry, "100"),
    ("Probabilidad de cruzamiento (0-1):", crossover_prob_entry, "0.8"),
    ("Probabilidad de mutación (0-1):", mutacion_prob_entry, "0.4"),
    ("Tamaño de la población inicial:", initial_population_size_entry, "10"),
    ("Tamaño máximo de la población:", max_population_size_entry, "20"),
    ("Número de películas por lista:", movies_per_list_entry, "10"),
]

for i, (label, widget, default) in enumerate(inputs):
    add_input_row(frame_inputs, label, widget, default, i)

# Botón para ejecutar el algoritmo
ttk.Button(frame_inputs, text="Ejecutar Algoritmo", command=ejecutar_algoritmo, padding=10, width=40).grid(row=len(inputs), columnspan=2, pady=20)

# Área de texto para mostrar los resultados en frame_results
lista_resultados = scrolledtext.ScrolledText(frame_results, width=80, height=37)
lista_resultados.pack()

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
