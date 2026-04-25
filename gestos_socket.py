from pathlib import Path
import time
import urllib.request
import socket

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# conf de fuentes y colores
FONT_PRINCIPAL = cv2.FONT_HERSHEY_DUPLEX  
FONT_SIZE_TITULO = 0.9
FONT_SIZE_NORMAL = 0.8
FONT_SIZE_PEQUEÑO = 0.7
FONT_THICKNESS = 2
FONT_COLOR_INFO = (0, 255, 0)   
FONT_COLOR_ALERTA = (255, 0, 0)  
FONT_COLOR_ESTADO = (0, 255, 255) 

# confi de conexión
IP_ROBOT = "192.168.1.26"       
PUERTO_ROBOT = 8000

# paramertros de seguridad
TIEMPO_CONFIRMACION = 1.2  # Segundos que se debe mantener el gesto  para validarlo
TIEMPO_COOLDOWN = 3.0      # Segundos de bloqueo (robot ocupado) tras enviar un comando

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parents[1] / "hand_landmarker.task"

def asegurar_modelo() -> Path:
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        print("Descargando modelo de MediaPipe...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

# confi de acciones y gestos
acciones = ["Abrir Gripper", "Cerrar Gripper", "Detener Robot", "Reanudar Proceso", "Ir a Home"]
diccionario_gestos = {} 

def contar_dedos(hand_landmarks):
    dedos = 0
    tips_ids = [4, 8, 12, 16, 20]
    if hand_landmarks[tips_ids[0]].x < hand_landmarks[tips_ids[0] - 1].x: dedos += 1
    for id in range(1, 5):
        if hand_landmarks[tips_ids[id]].y < hand_landmarks[tips_ids[id] - 2].y: dedos += 1
    return dedos

def configurar_gestos(cap, landmarker):
    print("--- MODO CONFIGURACIÓN DE GESTOS ---")
    for accion in acciones:
        print(f"Muestra el gesto para: '{accion}' y presiona la tecla 'c' para confirmar...")
        while True:
            success, img = cap.read()
            if not success: continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = landmarker.detect(mp_image)
            dedos_actuales = 0
            
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    vision.drawing_utils.draw_landmarks(img, hand_landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS)
                    dedos_actuales = contar_dedos(hand_landmarks)
            
            cv2.putText(img, f"Configurando: {accion}", (10, 50), FONT_PRINCIPAL, FONT_SIZE_TITULO, FONT_COLOR_INFO, FONT_THICKNESS)
            cv2.putText(img, f"Dedos detectados: {dedos_actuales}", (10, 100), FONT_PRINCIPAL, FONT_SIZE_NORMAL, FONT_COLOR_ALERTA, FONT_THICKNESS)
            cv2.imshow("Calibracion de Gestos", img)
            
            if cv2.waitKey(1) & 0xFF == ord('c'):
                if dedos_actuales not in diccionario_gestos.values():
                    diccionario_gestos[accion] = dedos_actuales
                    print(f"> ¡Listo! [{accion}] asignado a gesto de {dedos_actuales} dedos.")
                    time.sleep(0.5) 
                    break
                else:
                    print("> ¡Cuidado! Ese número de dedos ya está en uso.")

def main():
    model_path = asegurar_modelo()
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara por defecto.")
    
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        configurar_gestos(cap, landmarker)

        print("\nConectando a los equipos...")
        try:
            robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            robot_socket.connect((IP_ROBOT, PUERTO_ROBOT))
            print("> Robot IRC5 conectado.")
        except Exception:
            robot_socket = None

        print("\n--- INICIANDO CONTROL SEGURO ---")
        
        # var de control de seguridad
        comando_previo = ""
        gesto_en_progreso = "Ninguna"
        tiempo_inicio_gesto = 0.0
        tiempo_ultimo_comando = 0.0
        
        while True:
            success, img = cap.read()
            if not success: break
            
            tiempo_actual = time.time()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            results = landmarker.detect(mp_image)
            
            lectura_cruda = "Ninguna"
            
            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    vision.drawing_utils.draw_landmarks(img, hand_landmarks, vision.HandLandmarksConnections.HAND_CONNECTIONS)
                    dedos = contar_dedos(hand_landmarks)
                    for accion, gesto_dedos in diccionario_gestos.items():
                        if dedos == gesto_dedos:
                            lectura_cruda = accion
                            break
            
            # interlocks y lógica de seguridad
            estado_sistema = "LISTO"
            color_estado = (0, 255, 0)
            
            # Validar si estamos en periodo de Cooldown 
            if (tiempo_actual - tiempo_ultimo_comando) < TIEMPO_COOLDOWN:
                estado_sistema = "ROBOT OCUPADO (COOLDOWN)"
                color_estado = (0, 0, 255) 
                lectura_cruda = "Ninguna" 
            else:
                # logica de antirrebote
                if lectura_cruda != "Ninguna":
                    if lectura_cruda == gesto_en_progreso:
                       #veificar el tiempo que se ha mantenido el mismo gesto
                        tiempo_mantenido = tiempo_actual - tiempo_inicio_gesto
                        cv2.rectangle(img, (10, 80), (10 + int(min(tiempo_mantenido/TIEMPO_CONFIRMACION, 1) * 200), 100), (0, 255, 255), -1)
                        
                        if tiempo_mantenido >= TIEMPO_CONFIRMACION:
                            # validar zona neutra
                            if lectura_cruda != comando_previo:
                                print(f">>> EJECUTANDO COMANDO SEGURO: {lectura_cruda} <<<")
                                comando_previo = lectura_cruda
                                tiempo_ultimo_comando = tiempo_actual 
                                
                                #  envio de comandos al robot
                                if robot_socket:
                                    try:
                                        if lectura_cruda == "Abrir Gripper": robot_socket.sendall(b"OPN")
                                        elif lectura_cruda == "Cerrar Gripper": robot_socket.sendall(b"CLS")
                                        elif lectura_cruda == "Detener Robot": robot_socket.sendall(b"STP")
                                        elif lectura_cruda == "Reanudar Proceso": robot_socket.sendall(b"RES")
                                        elif lectura_cruda == "Ir a Home": robot_socket.sendall(b"HOM")
                                    except Exception as e:
                                        print(f"Error de red con el robot: {e}")

                    else:
                        # El gesto cambió, asi que, reiniciar el temporizador de confirmación
                        gesto_en_progreso = lectura_cruda
                        tiempo_inicio_gesto = tiempo_actual
                else:
                    # El usuario bajó la mano. Se reinicia el sistema para aceptar nuevos comandos.
                    gesto_en_progreso = "Ninguna"
                    comando_previo = "" 

            # interfaz de usuario
            cv2.putText(img, f"Estado: {estado_sistema}", (10, 40), FONT_PRINCIPAL, FONT_SIZE_NORMAL, color_estado, FONT_THICKNESS)
            if lectura_cruda != "Ninguna" and estado_sistema == "LISTO":
                cv2.putText(img, f"Detectando: {lectura_cruda}...", (10, 70), FONT_PRINCIPAL, FONT_SIZE_PEQUEÑO, FONT_COLOR_ESTADO, FONT_THICKNESS)

            cv2.imshow("HMI Industrial Seguro", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    if robot_socket: robot_socket.close()

if __name__ == "__main__":
    main()