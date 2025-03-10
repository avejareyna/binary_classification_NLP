# Importación de bibliotecas necesarias
from tensorflow.keras.datasets import imdb  # Dataset IMDB de reseñas de películas
from tensorflow.keras import models, layers  # Para construir el modelo de red neuronal
import numpy as np  # Para operaciones numéricas y manejo de matrices
import matplotlib.pyplot as plt  # Para graficar los resultados
from tensorflow.keras.utils import plot_model  # Para visualizar la arquitectura del modelo

def train_and_evaluate_model():
    """
    Función para entrenar un modelo de clasificación binaria en el dataset IMDB.
    Retorna el modelo entrenado y los datos de prueba.
    """
    # 1. Cargar el dataset IMDB
    # Se limita el vocabulario a las 10,000 palabras más frecuentes
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

    # 2. Preprocesamiento de los datos
    # Mapear palabras a índices
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for (key, value) in word_index.items()}

    # Decodificar la primera reseña
    decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(decoded_review)  # Muestra la primera reseña decodificada
    print(train_labels[0])  # Muestra la etiqueta de la primera reseña (1=Positiva, 0=Negativa)

    # 3. Vectorización de secuencias
    # Convertir las reseñas en vectores one-hot
    def vectorize_sequences(sequences, dimension=10000):
        """
        Convierte las secuencias de palabras en vectores one-hot.
        - sequences: Lista de reseñas (cada reseña es una lista de índices de palabras).
        - dimension: Tamaño del vocabulario (10,000 en este caso).
        """
        results = np.zeros((len(sequences), dimension))  # Matriz de ceros
        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1.0  # Marcar las posiciones de las palabras presentes
        return results

    # Vectorizar los datos de entrenamiento y prueba
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    # Convertir las etiquetas a flotantes (requerido para la función de pérdida)
    y_train = np.asarray(train_labels).astype('float32')
    y_test = np.asarray(test_labels).astype('float32')

    # 4. Construir el modelo de red neuronal
    model = models.Sequential([
        # Capa densa con 16 unidades y activación ReLU
        layers.Dense(16, activation='relu', input_shape=(10000,)),
        # Capa densa con 16 unidades y activación ReLU
        layers.Dense(16, activation='relu'),
        # Capa de salida con 1 unidad y activación sigmoide (para clasificación binaria)
        layers.Dense(1, activation='sigmoid')
    ])

    # 5. Visualizar la arquitectura del modelo
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # 6. Separar datos de validación
    # Separar 10,000 muestras del conjunto de entrenamiento para validación
    x_val = x_train[:10000]  # Primeros 10,000 ejemplos para validación
    partial_x_train = x_train[10000:]  # Resto para entrenamiento
    y_val = y_train[:10000]  # Primeras 10,000 etiquetas para validación
    partial_y_train = y_train[10000:]  # Resto de etiquetas para entrenamiento

    # 7. Compilar el modelo
    # Se utiliza el optimizador RMSprop, la función de pérdida binary_crossentropy
    # y la métrica de precisión (accuracy)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 8. Entrenar el modelo
    # Se entrena el modelo con los datos de entrenamiento y se valida con los datos de validación
    history = model.fit(
        partial_x_train, partial_y_train,
        epochs=20,  # Número de épocas
        batch_size=512,  # Tamaño del lote
        validation_data=(x_val, y_val)  # Datos de validación
    )

    # 9. Graficar la pérdida y la precisión durante el entrenamiento
    history_dict = history.history  # Obtener el historial del entrenamiento

    # Graficar la pérdida en entrenamiento y validación
    plt.plot(range(1, len(history_dict['loss']) + 1), history_dict['loss'], 'bo', label='Training loss')
    plt.plot(range(1, len(history_dict['val_loss']) + 1), history_dict['val_loss'], 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Graficar la precisión en entrenamiento y validación
    plt.clf()  # Limpiar la figura anterior
    plt.plot(range(1, len(history_dict['accuracy']) + 1), history_dict['accuracy'], 'bo', label='Training accuracy')
    plt.plot(range(1, len(history_dict['val_accuracy']) + 1), history_dict['val_accuracy'], 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # 10. Evaluar el modelo en el conjunto de prueba
    model.evaluate(x_test, y_test)

    # 11. Segunda versión del modelo
    # Se define un modelo con más unidades en la primera capa (64 en lugar de 16)
    model = models.Sequential([
        layers.Input(shape=(10000,)),  
        layers.Dense(64, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compilar y entrenar el modelo
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=4, batch_size=512)
    results = model.evaluate(x_test, y_test)
    print(results)  # Mostrar los resultados de la evaluación

    # 12. Predicción
    # Realizar predicciones en las primeras dos muestras del conjunto de prueba
    model.predict(x_test[0:2])

    # 13. Tercera versión del modelo
    # Se cambia la función de pérdida a MSE (Mean Squared Error)
    model = models.Sequential([
        layers.Input(shape=(10000,)),  
        layers.Dense(64, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compilar y entrenar el modelo
    model.compile(optimizer='rmsprop',
                  loss='mse',  # Cambio de función de pérdida
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3, batch_size=256)
    results = model.evaluate(x_test, y_test)
