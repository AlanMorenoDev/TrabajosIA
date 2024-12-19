import pygame
import random
import pandas as pd
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import joblib
from sklearn.tree import DecisionTreeClassifier

# Inicializar Pygame
pygame.init()

# Dimensiones de la pantalla
w, h = 800, 500
pantalla = pygame.display.set_mode((w, h))
pygame.display.set_caption("Juego: Disparo de Bala, Salto, Nave y Menú")

# Colores
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)

# Variables globales
jugador = pygame.Rect(50, h - 100, 32, 48)
bala = pygame.Rect(w - 50, h - 90, 16, 12)
nave = pygame.Rect(w - 100, h - 100, 64, 64)
salto = False
salto_altura = 20
gravedad = 1
en_suelo = True
pausa = False
menu_activo = True
modo_auto = False
datos_modelo = []
modelo_arbol = None
modelo_mlp = None

# Variables para la bala
velocidad_bala = -10
bala_disparada = False

# Variables para el fondo
fondo_x1 = 0
fondo_x2 = w

# Cargar imágenes
jugador_frames = [
    pygame.image.load('assets/sprites/mono_frame_1.png'),
    pygame.image.load('assets/sprites/mono_frame_2.png'),
    pygame.image.load('assets/sprites/mono_frame_3.png'),
    pygame.image.load('assets/sprites/mono_frame_4.png')
]
bala_img = pygame.image.load('assets/sprites/purple_ball.png')
fondo_img = pygame.image.load('assets/game/fondo2.png')
nave_img = pygame.image.load('assets/game/ufo.png')
fondo_img = pygame.transform.scale(fondo_img, (w, h))
fuente = pygame.font.SysFont('Arial', 24)

# Variables de animación
current_frame = 0
frame_speed = 10
frame_count = 0

# Función para disparar la bala
def disparar_bala():
    global bala_disparada, velocidad_bala
    if not bala_disparada:
        velocidad_bala = random.randint(-10, -5)
        bala_disparada = True

# Función para reiniciar la posición de la bala
def reset_bala():
    global bala, bala_disparada
    bala.x = w - 50
    bala_disparada = False

# Función para manejar el salto
def manejar_salto():
    global jugador, salto, salto_altura, gravedad, en_suelo
    if salto:
        jugador.y -= salto_altura
        salto_altura -= gravedad
        print(f"Jugador saltando: Y={jugador.y}, salto_altura={salto_altura}")
        
        if jugador.y >= h - 100:
            jugador.y = h - 100
            salto = False
            salto_altura = 20
            en_suelo = True
            print("Jugador aterrizó.")

# Función para guardar datos en memoria
def guardar_datos():
    global jugador, bala, velocidad_bala, salto
    distancia = abs(jugador.x - bala.x)
    if distancia < 300:
        salto_hecho = 1 if salto else 0
        datos_modelo.append((velocidad_bala, distancia, salto_hecho))
        print(f"Datos registrados: Velocidad={velocidad_bala}, Distancia={distancia}, Salto={salto_hecho}")

# Función para guardar los datos en CSV (sobrescribe si el archivo ya existe)
def guardar_datos_csv():
    global datos_modelo
    if len(datos_modelo) == 0:
        print("No hay datos para guardar.")
        return
    df = pd.DataFrame(datos_modelo, columns=['Velocidad', 'Distancia', 'Salto'])
    
    # Sobrescribir el archivo CSV si ya existe
    df.to_csv('datosEntrenamiento.csv', index=False)
    print("Datos guardados en 'datosEntrenamiento.csv'.")

# Función para entrenar y guardar el modelo MLP
def entrenar_guardar_modelo_mlp():
    global datos_modelo
    if len(datos_modelo) == 0:
        print("No hay datos para entrenar el modelo.")
        return

    # Cargar los datos recolectados
    df = pd.DataFrame(datos_modelo, columns=['Velocidad', 'Distancia', 'Salto'])
    X = df[['Velocidad', 'Distancia']].values
    y = df['Salto'].values

    # Crear el modelo de red neuronal multicapa
    model = Sequential([
        Dense(8, input_dim=2, activation='relu'),  # Capa oculta con 8 neuronas y ReLU
        Dense(4, activation='relu'),              # Capa oculta adicional con 4 neuronas y ReLU
        Dense(1, activation='sigmoid')            # Capa de salida con activación sigmoide
    ])

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Entrenar el modelo
    model.fit(X, y, epochs=20, batch_size=32, verbose=1)

    # Evaluar el modelo en los datos
    loss, accuracy = model.evaluate(X, y, verbose=0)
    print(f"\nPrecisión del modelo: {accuracy:.2f}")

    # Guardar el modelo entrenado (sobrescribiendo si ya existe)
    model.save('modelo_mlp.h5')
    print("Modelo MLP guardado como 'modelo_mlp.h5'.")

# Función para entrenar y guardar el modelo Árbol de Decisión
def entrenar_guardar_modelo_arbol():
    global datos_modelo
    if len(datos_modelo) == 0:
        print("No hay datos para entrenar el modelo.")
        return

    # Cargar los datos recolectados
    df = pd.DataFrame(datos_modelo, columns=['Velocidad', 'Distancia', 'Salto'])
    X = df[['Velocidad', 'Distancia']].values
    y = df['Salto'].values

    # Crear el modelo de Árbol de Decisión
    modelo_arbol = DecisionTreeClassifier(random_state=42)
    modelo_arbol.fit(X, y)

    # Guardar el modelo entrenado (sobrescribiendo si ya existe)
    joblib.dump(modelo_arbol, 'modelo_arbol.pkl')
    print("Modelo Árbol de Decisión guardado como 'modelo_arbol.pkl'.")

# Función para cargar el modelo de red neuronal MLP
def cargar_modelo_mlp():
    global modelo_mlp
    try:
        modelo_mlp = load_model('modelo_mlp.h5')
        print("Modelo MLP cargado correctamente.")
    except Exception as e:
        print(f"Error al cargar el modelo MLP: {e}")

# Función para cargar el modelo de Árbol de Decisión
def cargar_modelo_arbol():
    global modelo_arbol
    try:
        modelo_arbol = joblib.load('modelo_arbol.pkl')
        print("Modelo Árbol de Decisión cargado correctamente.")
    except FileNotFoundError:
        print("El modelo 'modelo_arbol.pkl' no existe. Entrena el modelo primero.")

# Función para decisión automática con MLP
def decision_automatica():
    global modelo_mlp, jugador, bala
    if modelo_mlp is None:
        print("Modelo MLP no cargado.")
        return False

    # Obtener los datos de entrada
    velocidad = float(velocidad_bala)  # Convertir a flotante por seguridad
    distancia = float(abs(jugador.x - bala.x))  # Convertir a flotante por seguridad
    entrada = np.array([[velocidad, distancia]])  # Crear entrada como matriz 2D

    try:
        # Realizar la predicción
        prediccion = modelo_mlp.predict(entrada, verbose=0)[0][0]
    except Exception as e:
        print(f"Error durante la predicción con el modelo MLP: {e}")
        return False

    # Decidir si saltar (umbral 0.5)
    return prediccion > 0.5

# Función para decisión automática con Árbol de Decisión
def decision_automatica_arbol():
    global modelo_arbol, jugador, bala
    if modelo_arbol is None:
        print("Modelo Árbol de Decisión no cargado.")
        return False

    # Obtener los datos de entrada
    velocidad = velocidad_bala
    distancia = abs(jugador.x - bala.x)
    entrada = np.array([[velocidad, distancia]])  # Crear entrada para el modelo

    # Hacer la predicción
    prediccion = modelo_arbol.predict(entrada)[0]

    # Decidir si saltar (umbral 0.5)
    return prediccion == 1

# Función para mostrar el menú con una mejor presentación
def mostrar_menu():
    global menu_activo, modo_auto, pausa

    # Limpiar la pantalla
    pantalla.fill(NEGRO)

    # Fondo del menú
    fondo_menu = pygame.Surface((w, h))
    fondo_menu.fill((0, 50, 100))  # Un color de fondo más oscuro
    pantalla.blit(fondo_menu, (0, 0))

    # Título del juego
    titulo = fuente.render("JUEGO: Disparo de Bala y Salto", True, BLANCO)
    pantalla.blit(titulo, (w // 2 - titulo.get_width() // 2, 50))

    # Opciones de menú
    opciones = [
        "Presiona 'A' para Modo Automático",
        "Presiona 'M' para Modo Manual",
        "Presiona 'S' para Guardar Datos",
        "Presiona 'C' para Cargar Modelo MLP",
        "Presiona 'L' para Cargar Modelo Árbol de Decisión",
        "Presiona 'G' para Entrenar y Guardar Modelo MLP",
        "Presiona 'H' para Entrenar y Guardar Modelo Árbol de Decisión",
        "Presiona 'Q' para Salir"
    ]

    # Dibujar opciones
    for i, opcion in enumerate(opciones):
        texto_opcion = fuente.render(opcion, True, BLANCO)
        pantalla.blit(texto_opcion, (w // 2 - texto_opcion.get_width() // 2, 150 + i * 40))

    pygame.display.flip()

    # Reiniciar el menú y ciclo de eventos
    menu_activo = True
    pausa = False  # Asegurarnos de que no está en pausa
    while menu_activo:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                pygame.quit()
                exit()
            if evento.type == pygame.KEYDOWN:
                if evento.key == pygame.K_a:
                    modo_auto = True  # Activar el modo automático
                    menu_activo = False
                    pausa = False  # Salimos del menú
                elif evento.key == pygame.K_m:
                    modo_auto = False  # Activar el modo manual
                    menu_activo = False
                    pausa = False
                elif evento.key == pygame.K_s:
                    guardar_datos_csv()
                elif evento.key == pygame.K_c:
                    cargar_modelo_mlp()
                elif evento.key == pygame.K_l:
                    cargar_modelo_arbol()
                elif evento.key == pygame.K_g:
                    entrenar_guardar_modelo_mlp()
                elif evento.key == pygame.K_h:
                    entrenar_guardar_modelo_arbol()
                elif evento.key == pygame.K_q:
                    pygame.quit()
                    exit()

# Función para detener el juego y mostrar el menú
def detener_y_mostrar_menu():
    global menu_activo, pausa, bala, jugador, nave, bala_disparada, salto, en_suelo
    pausa = True  # Activar pausa
    jugador.x, jugador.y = 50, h - 100
    bala.x = w - 50
    nave.x, nave.y = w - 100, h - 100
    bala_disparada = False
    salto = False
    en_suelo = True
    menu_activo = True  # Asegurarse de que el menú esté activo
    mostrar_menu()

# Función de actualización
def update():
    global fondo_x1, fondo_x2, current_frame, frame_count, bala, velocidad_bala, salto, en_suelo, bala_disparada

    # Movimiento del fondo
    fondo_x1 -= 1
    fondo_x2 -= 1
    if fondo_x1 <= -w: fondo_x1 = w
    if fondo_x2 <= -w: fondo_x2 = w

    pantalla.blit(fondo_img, (fondo_x1, 0))
    pantalla.blit(fondo_img, (fondo_x2, 0))

    # Animación del jugador
    frame_count += 1
    if frame_count >= frame_speed:
        current_frame = (current_frame + 1) % len(jugador_frames)
        frame_count = 0
    pantalla.blit(jugador_frames[current_frame], (jugador.x, jugador.y))

    # Movimiento de la bala
    if bala_disparada:
        bala.x += velocidad_bala
    if bala.x < 0:
        reset_bala()
    pantalla.blit(bala_img, (bala.x, bala.y))

    # Dibujar el UFO
    pantalla.blit(nave_img, (nave.x, nave.y))

    # Colisión
    if jugador.colliderect(bala):
        print("¡Colisión detectada!")
        detener_y_mostrar_menu()

    # Modo automático
    if modo_auto:
        if modelo_mlp:
            if decision_automatica():  # Usar el modelo MLP
                print("Salto activado por el modo automático (MLP).")
                salto = True
                en_suelo = False
        elif modelo_arbol:
            if decision_automatica_arbol():  # Usar el modelo de Árbol de Decisión
                print("Salto activado por el modo automático (Árbol de Decisión).")
                salto = True
                en_suelo = False

# Bucle principal del juego
def main():
    global salto, en_suelo, modo_auto, bala_disparada, pausa

    reloj = pygame.time.Clock()
    mostrar_menu()  # Mostrar el menú al iniciar
    correr = True

    while correr:
        for evento in pygame.event.get():
            if evento.type == pygame.QUIT:
                correr = False
            if evento.type == pygame.KEYDOWN:
                # Regresar al menú
                if evento.key == pygame.K_p:
                    detener_y_mostrar_menu()
                # Salto en modo manual
                if evento.key == pygame.K_SPACE and en_suelo and not pausa and not modo_auto:
                    salto = True
                    en_suelo = False

        if not pausa:  # Solo continuar el juego si no está pausado
            if not bala_disparada:
                disparar_bala()

            if salto:
                manejar_salto()

            if not modo_auto:
                guardar_datos()
            elif modo_auto:
                if modelo_mlp:
                    if decision_automatica():  # Decidir automáticamente con MLP
                        print("Salto activado por el modo automático.")
                        salto = True
                        en_suelo = False
                elif modelo_arbol:
                    if decision_automatica_arbol():  # Decidir automáticamente con Árbol
                        print("Salto activado por el modo automático.")
                        salto = True
                        en_suelo = False

            update()

        pygame.display.flip()
        reloj.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()
