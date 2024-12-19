import pygame
import heapq
import math

# Inicializar pygame
pygame.init()

# Configuración de la ventana y colores
ANCHO = 400
ALTO = 400
TAMAÑO_CELDA = 40
FILAS = ALTO // TAMAÑO_CELDA
COLUMNAS = ANCHO // TAMAÑO_CELDA
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
VERDE = (0, 255, 0)
AZUL = (0, 0, 255)
GRIS = (200, 200, 200)
ROJO = (255, 0, 0)
AMARILLO = (255, 255, 0)
NARANJA = (255, 165, 0)

ventana = pygame.display.set_mode((ANCHO, ALTO))
pygame.display.set_caption("A* - Buscar Camino")

def dibujar_mapa(inicio, objetivo, mapa, camino=[], visitados=set(), disponibles=set()):
    for fila in range(FILAS):
        for columna in range(COLUMNAS):
            if mapa[fila][columna] == 1:
                pygame.draw.rect(ventana, NEGRO, (columna * TAMAÑO_CELDA, fila * TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA))
            else:
                pygame.draw.rect(ventana, BLANCO, (columna * TAMAÑO_CELDA, fila * TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA))
            pygame.draw.rect(ventana, GRIS, (columna * TAMAÑO_CELDA, fila * TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA), 1)

    for (x, y) in visitados:
        if (x, y) not in camino and (x, y) not in disponibles:
            pygame.draw.rect(ventana, AMARILLO, (y * TAMAÑO_CELDA, x * TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA))

    for (x, y) in disponibles:
        if (x, y) not in camino:
            pygame.draw.rect(ventana, NARANJA, (y * TAMAÑO_CELDA, x * TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA))

    for (x, y) in camino:
        pygame.draw.rect(ventana, ROJO, (y * TAMAÑO_CELDA, x * TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA))

    if inicio:
        pygame.draw.rect(ventana, VERDE, (inicio[1] * TAMAÑO_CELDA, inicio[0] * TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA))
    if objetivo:
        pygame.draw.rect(ventana, AZUL, (objetivo[1] * TAMAÑO_CELDA, objetivo[0] * TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA))

class Nodo:
    def __init__(self, posicion, padre=None):
        self.posicion = posicion
        self.padre = padre
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.posicion == other.posicion

    def __lt__(self, other):
        return self.f < other.f

def heuristica(posicion_actual, posicion_objetivo):
    dx = abs(posicion_actual[0] - posicion_objetivo[0])
    dy = abs(posicion_actual[1] - posicion_objetivo[1])
    return (math.sqrt(2) * min(dx, dy)) + abs(dx - dy)

def obtener_vecinos(nodo, mapa):
    movimientos = [
        (0, -1, 1),  # Izquierda
        (0, 1, 1),   # Derecha
        (-1, 0, 1),  # Arriba
        (1, 0, 1),   # Abajo
        (-1, -1, math.sqrt(2)),  # Arriba-Izquierda
        (-1, 1, math.sqrt(2)),   # Arriba-Derecha
        (1, -1, math.sqrt(2)),   # Abajo-Izquierda
        (1, 1, math.sqrt(2))     # Abajo-Derecha
    ]
    vecinos = []

    for movimiento in movimientos:
        nueva_posicion = (nodo.posicion[0] + movimiento[0], nodo.posicion[1] + movimiento[1])
        costo_movimiento = movimiento[2]

        if 0 <= nueva_posicion[0] < len(mapa) and 0 <= nueva_posicion[1] < len(mapa[0]) and mapa[nueva_posicion[0]][nueva_posicion[1]] == 0:
            if abs(movimiento[0]) + abs(movimiento[1]) == 2:  # Movimiento diagonal
                if mapa[nodo.posicion[0] + movimiento[0]][nodo.posicion[1]] == 1 or mapa[nodo.posicion[0]][nodo.posicion[1] + movimiento[1]] == 1:
                    continue
            vecinos.append((nueva_posicion, costo_movimiento))

    return vecinos

def a_estrella(mapa, inicio, objetivo):
    nodos_abiertos = []
    nodos_abiertos_dict = {}
    nodos_cerrados = set()
    visitados = set()
    disponibles = set()
    nodo_inicio = Nodo(inicio)
    nodo_objetivo = Nodo(objetivo)

    heapq.heappush(nodos_abiertos, nodo_inicio)
    nodos_abiertos_dict[nodo_inicio.posicion] = nodo_inicio

    while nodos_abiertos:
        dibujar_mapa(inicio, objetivo, mapa, [], visitados, disponibles)
        pygame.display.update()
        pygame.time.wait(50)

        nodo_actual = heapq.heappop(nodos_abiertos)

        if nodo_actual.posicion in nodos_abiertos_dict:
            nodos_abiertos_dict.pop(nodo_actual.posicion)

        nodos_cerrados.add(nodo_actual.posicion)
        visitados.add(nodo_actual.posicion)

        if nodo_actual == nodo_objetivo:
            camino = []
            while nodo_actual:
                camino.append(nodo_actual.posicion)
                nodo_actual = nodo_actual.padre
            return camino[::-1], visitados, disponibles

        vecinos = obtener_vecinos(nodo_actual, mapa)
        for posicion_vecino, costo_movimiento in vecinos:
            if posicion_vecino in nodos_cerrados:
                continue

            vecino = Nodo(posicion_vecino, nodo_actual)
            vecino.g = nodo_actual.g + costo_movimiento
            vecino.h = heuristica(vecino.posicion, nodo_objetivo.posicion)
            vecino.f = vecino.g + vecino.h

            if posicion_vecino not in nodos_abiertos_dict or nodos_abiertos_dict[posicion_vecino].f > vecino.f:
                heapq.heappush(nodos_abiertos, vecino)
                nodos_abiertos_dict[posicion_vecino] = vecino
                disponibles.add(posicion_vecino)

    return None, visitados, disponibles

def mostrar_camino_gradual(camino):
    """Dibuja el camino gradualmente en la ventana de pygame."""
    for nodo in camino:
        pygame.draw.rect(ventana, ROJO, (nodo[1] * TAMAÑO_CELDA, nodo[0] * TAMAÑO_CELDA, TAMAÑO_CELDA, TAMAÑO_CELDA))
        pygame.display.update()
        pygame.time.wait(50)

inicio = None
objetivo = None
mapa = [[0 for _ in range(COLUMNAS)] for _ in range(FILAS)]
camino = []
visitados = set()
disponibles = set()

ejecutando = True
while ejecutando:
    ventana.fill(BLANCO)
    dibujar_mapa(inicio, objetivo, mapa, camino if camino else [], visitados, disponibles)

    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            ejecutando = False

        if evento.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            fila = y // TAMAÑO_CELDA
            columna = x // TAMAÑO_CELDA

            if inicio is None:
                inicio = (fila, columna)
            elif objetivo is None:
                objetivo = (fila, columna)
            elif mapa[fila][columna] == 0:
                mapa[fila][columna] = 1
            elif mapa[fila][columna] == 1:
                mapa[fila][columna] = 0

        if evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_RETURN and inicio and objetivo:
                camino, visitados, disponibles = a_estrella(mapa, inicio, objetivo)
                if camino:
                    print("Camino encontrado:", camino)
                    mostrar_camino_gradual(camino)
                else:
                    print("No se encontró un camino.")
                    camino = []

    pygame.display.update()

pygame.quit()
