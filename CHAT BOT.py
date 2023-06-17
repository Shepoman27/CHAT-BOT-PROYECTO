import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()  # Objeto para realizar el stemming de las palabras

import numpy
import tflearn
import tensorflow
import json
import random
import pickle

# nltk.download('punkt')  # Descargar los recursos necesarios de NLTK (solo se necesita ejecutar una vez)

with open("contenido.js") as archivo:
    datos = json.load(archivo)  # Cargar los datos del archivo JSON

palabras = []
tags = []
auxX = []
auxY = []

# Recorrer los datos cargados del archivo JSON
for contenido in datos["contenido"]:
    for patrones in contenido["patrones"]:
        auxPalabra = nltk.word_tokenize(patrones)  # Tokenizar las frases en palabras
        palabras.extend(auxPalabra)  # Agregar las palabras a la lista
        auxX.append(auxPalabra)  # Agregar las palabras tokenizadas a auxX
        auxY.append(contenido["tag"])  # Agregar el tag correspondiente a auxY

        if contenido["tag"] not in tags:
            tags.append(contenido["tag"])  # Agregar los tags a la lista de tags

palabras = [stemmer.stem(w.lower()) for w in palabras if w != "?"]  # Realizar stemming y pasar las palabras a minúsculas
palabras = sorted(list(set(palabras)))  # Ordenar y eliminar duplicados en la lista de palabras
tags = sorted(tags)  # Ordenar los tags

entrenamiento = []
salida = []

salidaVacia = [0 for _ in range(len(tags))]  # Lista de ceros para la salida
for x, documento in enumerate(auxX):
    cubeta = []
    auxPalabra = [stemmer.stem(w.lower()) for w in documento]
    for w in palabras:
        if w in auxPalabra:
            cubeta.append(1)  # Agregar 1 si la palabra está presente en el documento
        else:
            cubeta.append(0)  # Agregar 0 si la palabra no está presente en el documento
    filaSalida = salidaVacia[:]
    filaSalida[tags.index(auxY[x])] = 1  # Configurar el valor de salida correspondiente al tag
    entrenamiento.append(cubeta)  # Agregar la cubeta al conjunto de entrenamiento
    salida.append(filaSalida)  # Agregar la fila de salida al conjunto de salida

print(entrenamiento)
print(salida)


entrenamiento = numpy.array(entrenamiento)  # Convertir el conjunto de entrenamiento a una matriz numpy
salida = numpy.array(salida)  # Convertir el conjunto de salida a una matriz numpy

tensorflow.compat.v1.reset_default_graph()

red = tflearn.input_data(shape=[None, len(entrenamiento[0])])  # Definir la capa de entrada de la red
red = tflearn.fully_connected(red, 10)  # Agregar una capa oculta con 10 neuronas
red = tflearn.fully_connected(red, 10)  # Agregar otra capa oculta con 10 neuronas
red = tflearn.fully_connected(red, len(salida[0]), activation="softmax")  # Capa de salida con activación softmax
red = tflearn.regression(red)  # Definir la regresión

modelo = tflearn.DNN(red)  # Crear el modelo de red neuronal
modelo.fit(entrenamiento, salida, n_epoch=1000, batch_size=100, show_metric=True)  # Entrenar el modelo
modelo.save("modelo.tflearn")  # Guardar el modelo entrenado

def mainBot():
    while True:
        entrada = input("Tu: ")  # Solicitar la entrada del usuario
        cubeta = [0 for _ in range(len(palabras))]  # Crear una cubeta de ceros para representar la entrada
        entradaProcesada = nltk.word_tokenize(entrada)  # Tokenizar la entrada del usuario
        entradaProcesada = [stemmer.stem(palabra.lower()) for palabra in entradaProcesada]  # Realizar stemming y pasar a minúsculas
        for palabraIndividual in entradaProcesada:
            for i, palabra in enumerate(palabras):
                if palabra == stemmer.stem(palabraIndividual):
                    cubeta[i] = 1  # Establecer 1 si la palabra está presente en la entrada
        resultados = modelo.predict([numpy.array(cubeta)])  # Predecir el resultado basado en la entrada
        resultadosIndices = numpy.argmax(resultados, axis=None)  # Obtener el índice del resultado más alto
        tag = tags[resultadosIndices]  # Obtener el tag correspondiente al resultado

        for tagAux in datos["contenido"]:
            if tagAux["tag"] == tag:
                respuestas = tagAux["respuestas"]  # Obtener las respuestas correspondientes al tag

        print("BOT:", random.choice(respuestas))  # Imprimir una respuesta aleatoria del conjunto de respuestas

mainBot()


