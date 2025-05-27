import cv2
import struct
import numpy as np
import os
import math
import heapq
import time
import traceback
import socket
import pickle
import matplotlib.pyplot as plt
from io import BytesIO

# --- Constantes y Parámetros para la Navegación ---
GRID_CELL_SIZE = 20
ROBOT_RADIUS = 8
ROBOT_WHEELBASE = 20
ROBOT_VELOCITY = 25
ROBOT_MAX_STEER_ANGLE = math.radians(30)
SIM_DT = 0.1
DROPOFF_X_OFFSET = GRID_CELL_SIZE * 2

# --- Definiciones para la Misión ---
class MissionStage:
    IDLE = 0
    GOING_TO_TARGET_OBSTACLE = 1
    GOING_TO_DROPOFF = 2
    MISSION_COMPLETE = 3


# --- Funciones de Utilidad ---
class ObjectDetector:
    
    # Funciones de inicialización y configuración
    def __init__(self):
        self.image = None
        self.original_image = None
        self.processed_image_base = None
        self.active_contours = []
        self.window_name = "Detector y Navegacion"
        self.controls_window_name = "Controles Deteccion"
        self.params = {
            'blur_kernel': 1, 'threshold_value': 68, 'min_area': 100,
            'max_area': 100000, 'aspect_ratio_min': 0.0, 'aspect_ratio_max': 9.6,
            'morph_kernel': 3, 'morph_iterations': 3
        }
        self.cap = None
        self._windows_set_up = False
        
    # Funciones de actualización de parámetros
    def _update_max_area_from_trackbar(self, val):
        self.params['max_area'] = val * 10; self._trigger_reprocess(replan=True)
    def update_blur_kernel(self, val):
        self.params['blur_kernel'] = max(1, val if val % 2 == 1 else val + 1 if val > 0 else 1); self._trigger_reprocess(replan=True)
    def update_threshold(self, val):
        self.params['threshold_value'] = val; self._trigger_reprocess(replan=True)
    def update_min_area(self, val):
        self.params['min_area'] = val; self._trigger_reprocess(replan=True)
    def update_aspect_min(self, val):
        self.params['aspect_ratio_min'] = val / 10.0; self._trigger_reprocess(replan=True)
    def update_aspect_max(self, val):
        self.params['aspect_ratio_max'] = val / 10.0; self._trigger_reprocess(replan=True)
    def update_morph_kernel(self, val):
        self.params['morph_kernel'] = max(1, val); self._trigger_reprocess(replan=True)
    def update_morph_iterations(self, val):
        self.params['morph_iterations'] = val; self._trigger_reprocess(replan=True)

    # Función para re-procesar la imagen y actualizar la visualización
    def _trigger_reprocess(self, replan=False):
        if hasattr(self, 'perform_detection_and_draw_elements') and callable(getattr(self, 'perform_detection_and_draw_elements')):
            self.perform_detection_and_draw_elements(replan=replan)
        elif hasattr(self, 'process_image_core') and callable(getattr(self, 'process_image_core')):
            self.process_image_core()
            if self.processed_image_base is not None:
                 cv2.imshow(self.window_name, self.processed_image_base)

    # Función de configuración de ventanas y trackbars
    def setup_windows_and_trackbars(self):
        if self._windows_set_up:
            return
        cv2.namedWindow(self.window_name)
        cv2.namedWindow(self.controls_window_name)
        cv2.createTrackbar("Blur Kernel", self.controls_window_name, self.params['blur_kernel'], 21, self.update_blur_kernel)
        cv2.setTrackbarPos("Blur Kernel", self.controls_window_name, self.params['blur_kernel'])
        cv2.createTrackbar("Threshold", self.controls_window_name, self.params['threshold_value'], 255, self.update_threshold)
        cv2.createTrackbar("Min Area", self.controls_window_name, self.params['min_area'], 2000, self.update_min_area)
        cv2.createTrackbar("Max Area", self.controls_window_name, self.params['max_area'] // 10, 10000, self._update_max_area_from_trackbar)
        cv2.createTrackbar("Aspect Min x10", self.controls_window_name, int(self.params['aspect_ratio_min'] * 10), 50, self.update_aspect_min)
        cv2.createTrackbar("Aspect Max x10", self.controls_window_name, int(self.params['aspect_ratio_max'] * 10), 100, self.update_aspect_max)
        cv2.createTrackbar("Morph Kernel", self.controls_window_name, self.params['morph_kernel'], 15, self.update_morph_kernel)
        cv2.createTrackbar("Morph Iter", self.controls_window_name, self.params['morph_iterations'], 10, self.update_morph_iterations)

        control_image = np.ones((280, 500, 3), dtype=np.uint8) * 240
        cv2.putText(control_image, "CONTROLES DE DETECCION", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        texts = ["Blur Kernel: Suavizado", "Threshold: Binarizacion", "Min/Max Area: Tamaño objetos",
                 "Aspect Ratio: Proporcion", "Morph: Filtrado morfologico", "--- NAVEGACION ---",
                 "Click Der: Definir META Manual (cancela misión)",
                 "'m': Resetear Misión (Obst. Obj. -> Dropoff)",
                 "'s': Guardar resultado", "'r': Resetear deteccion", "'q': Salir"]
        for i, text in enumerate(texts):
            y_pos = 60 + i * 18
            color = (0,0,0)
            if "NAVEGACION" in text or "Click" in text or "'m'" in text: color = (0,0,255)
            cv2.putText(control_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.imshow(self.controls_window_name, control_image)
        self._windows_set_up = True

    # Función para detectar una cámara válida
    def detectar_camara_valida(self):
        print("🔍 Buscando cámara externa (no la integrada)...")

        for i in [2, 1]:  # Evitar usar cámara 0 (Mac integrada)
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                    print(f"✅ Cámara externa {i} está funcionando.")
                    cap.release()
                    return i
                else:
                    print(f"⚠️ Cámara {i} abierta, pero sin imagen válida.")
                cap.release()
            else:
                print(f"❌ Cámara {i} no se pudo abrir.")
        return -1  # Ninguna cámara válida encontrada

    # Función para iniciar la cámara
    def start_camera(self):
        print(f"{time.time():.4f}: Entrando a start_camera")
        t_method_start = time.time()

        camera_index = 0  # Forzar índice de cámara USB externa; cambia a 1 si corresponde
        print(f"🔧 Forzando uso de cámara índice {camera_index}")
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"{time.time():.4f}: ❌ No se pudo abrir la cámara externa (índice {camera_index}).")
            return False
        print(f"{time.time():.4f}: ✅ Cámara (índice {camera_index}) abierta. Tiempo para abrir: {time.time() - t_method_start:.4f}s")

        desired_width = 1920
        desired_height = 1080
        t_before_set = time.time()
        print(f"{time.time():.4f}: Configurando resolución a {desired_width}x{desired_height}...")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
        t_after_set = time.time()
        print(f"{time.time():.4f}: cap.set (resolución) tomó {t_after_set - t_before_set:.4f}s")

        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Resolución solicitada: {desired_width}x{desired_height}, Resolución obtenida: {int(actual_width)}x{int(actual_height)}")

        t_before_read = time.time()
        print(f"{time.time():.4f}: Leyendo primer fotograma...")
        ret, frame = self.cap.read()
        t_after_read = time.time()
        print(f"{time.time():.4f}: Primera cap.read() tomó {t_after_read - t_before_read:.4f}s")

        if not ret:
            print(f"{time.time():.4f}: Error: No se pudo leer el primer fotograma.")
            if self.cap.isOpened(): self.cap.release()
            return False
        
        self.original_image = frame
        self.image = self.original_image.copy()

        if self._windows_set_up and self.original_image is not None:
            print(f"{time.time():.4f}: Mostrando fotograma preliminar...")
            t_before_quick_show = time.time()
            quick_view_img = self.original_image.copy()
            h_quick, w_quick = quick_view_img.shape[:2]
            cv2.putText(quick_view_img, "Iniciando sistema...",
                        (20, h_quick - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(self.window_name, quick_view_img)
            cv2.waitKey(1) 
            print(f"{time.time():.4f}: Fotograma preliminar mostrado (tomó {time.time() - t_before_quick_show:.4f}s).")
        
        h, w = frame.shape[:2]
        if hasattr(self, 'set_dropoff_point') and callable(getattr(self, 'set_dropoff_point')):
            self.set_dropoff_point((DROPOFF_X_OFFSET, h // 2))

        if hasattr(self, 'robot_x'):
            self.robot_x, self.robot_y = -100.0, -100.0; self.robot_theta = 0.0
            self.start_point = None; self.current_path_origin = None
            self.waypoints = []; self.path_found = False; self.robot_path_history = []
        
        print(f"Cámara lista para procesamiento. Resolución: {self.image.shape[1]}x{self.image.shape[0]}")
        if hasattr(self, 'dropoff_point') and self.dropoff_point:
             print(f"Punto de entrega (drop-off) fijado en: {self.dropoff_point}")
        
        print(f"{time.time():.4f}: Iniciando primer procesamiento completo de imagen...")
        t_before_trigger = time.time()
        self._trigger_reprocess(replan=False)
        t_after_trigger = time.time()
        print(f"{time.time():.4f}: Primera _trigger_reprocess (proc. completo y display) tomó {t_after_trigger - t_before_trigger:.4f}s")
        
        print(f"{time.time():.4f}: Saliendo de start_camera (total en método: {time.time() - t_method_start:.4f}s)")
        return True

    # Función para procesar la imagen y detectar objetos
    def process_image_core(self, image_to_process=None):
        if image_to_process is None:
            if self.image is None:
                self.processed_image_base = None; self.active_contours = []; return
            image_to_process = self.image
        gray = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2GRAY)
        blur_k = self.params['blur_kernel']
        blurred = cv2.GaussianBlur(gray, (blur_k, blur_k), 0) if blur_k > 1 else gray
        _, binary = cv2.threshold(blurred, self.params['threshold_value'], 255, cv2.THRESH_BINARY_INV)
        mk, mi = self.params['morph_kernel'], self.params['morph_iterations']
        if mk > 0 and mi > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mk, mk))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=mi)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=mi)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        base_img_for_drawing = self.original_image if self.original_image is not None else self.image
        if base_img_for_drawing is None:
            h_fallback, w_fallback = (480,640)
            if image_to_process is not None:
                if image_to_process.ndim == 3: h_fallback,w_fallback,_ = image_to_process.shape
                else: h_fallback,w_fallback = image_to_process.shape
            res_img = np.zeros((h_fallback,w_fallback,3), dtype=np.uint8)
        else: res_img = base_img_for_drawing.copy()
        self.active_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.params['min_area'] <= area <= self.params['max_area']): continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0 or w == 0 : continue
            ar = float(w) / h
            if not (self.params['aspect_ratio_min'] <= ar <= self.params['aspect_ratio_max']): continue
            self.active_contours.append(c)
        self.processed_image_base = res_img

    # Función para guardar el resultado de la detección
    def save_result(self, img, fname="result_navigation.jpg"):
        if img is not None: cv2.imwrite(fname, img); print(f"Resultado guardado como {fname}")
        else: print("No hay imagen para guardar.")

    # Función para resetear los parámetros de detección
    def reset_detection_parameters(self):
        self.params.update({'blur_kernel':1,'threshold_value':68,'min_area':100,'max_area':20000,'aspect_ratio_min':0.0,'aspect_ratio_max':9.6,'morph_kernel':3,'morph_iterations':3})
        if self._windows_set_up:
            cv2.setTrackbarPos("Blur Kernel",self.controls_window_name,self.params['blur_kernel']);cv2.setTrackbarPos("Threshold",self.controls_window_name,self.params['threshold_value']);cv2.setTrackbarPos("Min Area",self.controls_window_name,self.params['min_area']);cv2.setTrackbarPos("Max Area",self.controls_window_name,self.params['max_area']//10);cv2.setTrackbarPos("Aspect Min x10",self.controls_window_name,int(self.params['aspect_ratio_min']*10));cv2.setTrackbarPos("Aspect Max x10",self.controls_window_name,int(self.params['aspect_ratio_max']*10));cv2.setTrackbarPos("Morph Kernel",self.controls_window_name,self.params['morph_kernel']);cv2.setTrackbarPos("Morph Iter",self.controls_window_name,self.params['morph_iterations'])
        self._trigger_reprocess(replan=True);print("Parámetros de detección de obstáculos reseteados.")

# --- Clase Principal de Navegación y Detección de Obstáculos ---
class ObstacleAvoidanceNavigation(ObjectDetector):
    
    # Función para enviar datos de control al robot a través de un socket
    def enviar_control_por_socket(self, velocidad, angulo, x, y, theta):
        try:
            import math
            if not hasattr(self, 'sock_ctrl'):
                self.sock_ctrl = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.destino_ctrl = ("172.20.10.10", 8888)  # Puerto del nodo ROS2 receptor

            # Formato de mensaje compatible con el nodo TrayectoriaFollower
            mensaje = f"v={velocidad:.2f}, θ={math.degrees(angulo):.2f}°"
            self.sock_ctrl.sendto(mensaje.encode('utf-8'), self.destino_ctrl)

            print("\n📤 Enviando datos de control al robot (ROS2 receptor):")
            print(f"  🔸 Mensaje: {mensaje}")
            print(f"  🔸 Posición: x={x:.2f}, y={y:.2f}, θ(rad)={theta:.2f}")

        except Exception as e:
            print(f"⚠️ Error al enviar datos de control por socket: {e}")
    
    # Constructor de la clase ObstacleAvoidanceNavigation
    def __init__(self):
        t_init_start = time.time()
        print(f"{t_init_start:.4f}: ObstacleAvoidanceNavigation __init__ - INICIO")
        super().__init__()
        self.robot_x = -100.0; self.robot_y = -100.0; self.robot_theta = 0.0
        self.robot_pose_visually_confirmed = False
        self.pink_marker_center = None; self.yellow_marker_center = None
        self.robot_marker_contours_for_exclusion = []
        self.LOWER_PINK_HSV = np.array([155, 80, 80]); self.UPPER_PINK_HSV = np.array([175, 255, 255])
        self.LOWER_YELLOW_HSV = np.array([18, 80, 80]); self.UPPER_YELLOW_HSV = np.array([35, 255, 255])
        self.MARKER_MIN_AREA = 50; self.robot_path_history = []
        self.start_point = None; self.goal_point = None
        self.grid_map = None; self.dist_transform_map_grid = None
        self.obstacle_proximity_weight = 40.0; self.grid_dilation_kernel_size = 3
        self.waypoints = []; self.current_waypoint_index = 0
        self.planning_in_progress = False; self.path_found = False
        self.K_cte_stanley = 2.0; self.current_path_origin = None
        self.SMOOTHING_MAX_TURN_DEG = 70.0; self.SMOOTHING_CHAMFER_FACTOR = 0.7
        self.mission_active = False
        self.mission_stage = MissionStage.IDLE
        self.object_target_point = None 
        self.target_obstacle_info = None 
        self.dropoff_point = None 
        self.last_known_obstacle_target_for_drawing = None
        self.initial_robot_pos_set_by_vision = False
        self.path_to_final_destination_preview = [] # NUEVA VARIABLE

        print(f"{time.time():.4f}: Antes de setup_windows_and_trackbars en __init__")
        t_before_setup = time.time()
        if not self._windows_set_up:
            self.setup_windows_and_trackbars()
        t_after_setup = time.time()
        print(f"{time.time():.4f}: setup_windows_and_trackbars en __init__ tomó {t_after_setup - t_before_setup:.4f}s")
        print(f"{time.time():.4f}: ObstacleAvoidanceNavigation __init__ - FIN (total: {time.time() - t_init_start:.4f}s)")


    '''
    def encontrar_recuadro_blanco(self, image):
        if image is None:
            return None
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)  # buscar blancos casi puros
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area and area > 5000:  # evitar ruido
                max_area = area
                best_contour = cnt
        return best_contour

    def set_dropoff_point(self, image):
        recuadro = self.encontrar_recuadro_blanco(image)
        if recuadro is not None:
            x, y, w, h = cv2.boundingRect(recuadro)
            self.dropoff_point = (x + w // 2, y + h // 2)  # centro del recuadro blanco
            print(f"Dropoff detectado en centro del recuadro blanco izquierdo: {self.dropoff_point}")
        else:
            print("⚠️ No se detectó recuadro blanco. Usando valor por defecto.")
            h, w = image.shape[:2]
            self.dropoff_point = (DROPOFF_X_OFFSET, h // 2)
    '''
    
    """
    Funcion que publique un socket en tiempo real de la (imagen, matriz o lo mas conveniente) a traves de wifi para poderlo
    recibir desde el nodo llamado trayectoria_publisher desde la rasberry pi en ros. En el formato mas conveniente.
    """

    # Función para establecer el punto de entrega (drop-off)
    def set_dropoff_point(self, point):
        self.dropoff_point = point
        max_h, max_w = 480, 640 
        if self.original_image is not None: max_h, max_w = self.original_image.shape[:2]
        elif self.image is not None: max_h, max_w = self.image.shape[:2]
        safe_x = max(DROPOFF_X_OFFSET, min(point[0], max_w - DROPOFF_X_OFFSET))
        safe_y = max(GRID_CELL_SIZE, min(point[1], max_h - GRID_CELL_SIZE))
        self.dropoff_point = (int(safe_x), int(safe_y))
    
    # Función para encontrar el centro de un marcador de color específico
    def _find_color_marker_center(self, hsv_image, lower_hsv, upper_hsv):
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv); mask = cv2.erode(mask, None, iterations=1); mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None, None
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.MARKER_MIN_AREA]
        if not valid_contours: return None, None
        best_contour = max(valid_contours, key=cv2.contourArea); M = cv2.moments(best_contour)
        if M["m00"] != 0: return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])), best_contour
        return None, None

    # Función para detectar los marcadores del robot y su pose
    def detect_robot_markers_and_pose(self, image_to_process):
        if image_to_process is None: self.robot_pose_visually_confirmed = False; return False
        hsv = cv2.cvtColor(image_to_process, cv2.COLOR_BGR2HSV)
        self.pink_marker_center, pink_contour = self._find_color_marker_center(hsv, self.LOWER_PINK_HSV, self.UPPER_PINK_HSV)
        self.yellow_marker_center, yellow_contour = self._find_color_marker_center(hsv, self.LOWER_YELLOW_HSV, self.UPPER_YELLOW_HSV)
        self.robot_marker_contours_for_exclusion = []
        if pink_contour is not None: self.robot_marker_contours_for_exclusion.append(pink_contour)
        if yellow_contour is not None: self.robot_marker_contours_for_exclusion.append(yellow_contour)
        if self.pink_marker_center and self.yellow_marker_center:
            self.robot_x = (self.pink_marker_center[0] + self.yellow_marker_center[0]) / 2.0
            self.robot_y = (self.pink_marker_center[1] + self.yellow_marker_center[1]) / 2.0
            dx = self.pink_marker_center[0] - self.yellow_marker_center[0]; dy = self.pink_marker_center[1] - self.yellow_marker_center[1]
            self.robot_theta = self.normalize_angle(math.atan2(dy, dx)); self.robot_pose_visually_confirmed = True
            current_robot_pos_for_history = (int(self.robot_x), int(self.robot_y))
            if not self.robot_path_history or self.heuristic(self.robot_path_history[-1],current_robot_pos_for_history) > ROBOT_RADIUS/2:
                if not self.initial_robot_pos_set_by_vision:
                     self.robot_path_history=[current_robot_pos_for_history]; self.initial_robot_pos_set_by_vision=True
                else: self.robot_path_history.append(current_robot_pos_for_history)
            return True
        else: self.robot_pose_visually_confirmed = False; return False
    
    # Función para encontrar el obstáculo con la mayor diferencia de lados        
    def find_obstacle_with_max_side_difference(self):
        if not self.active_contours or not self.robot_pose_visually_confirmed:
            self.last_known_obstacle_target_for_drawing = None
            self.target_obstacle_info = None; return None
        target_obstacle = None; max_side_diff = -1.0 
        for obs_contour in self.active_contours:
            x, y, w, h = cv2.boundingRect(obs_contour)
            if w == 0 or h == 0: continue
            is_border_like = False
            if min(w, h) > 0:
                if max(w, h) / min(w, h) > 4.0: is_border_like = True
            if is_border_like: continue
            is_robot_marker = False
            for marker_contour in self.robot_marker_contours_for_exclusion:
                m_x_r,m_y_r,m_w_r,m_h_r = cv2.boundingRect(marker_contour)
                if not (x+w<m_x_r or x>m_x_r+m_w_r or y+h<m_y_r or y>m_y_r+m_h_r):
                    dist_sq=((x+w/2)-(m_x_r+m_w_r/2))**2+((y+h/2)-(m_y_r+m_h_r/2))**2
                    if dist_sq<(ROBOT_RADIUS*2.5)**2: is_robot_marker=True; break
            if is_robot_marker: continue
            side_difference = abs(w - h)
            M_obs_check = cv2.moments(obs_contour)
            if M_obs_check["m00"] == 0: continue
            if side_difference > max_side_diff:
                max_side_diff = side_difference; M_obs = cv2.moments(obs_contour)
                cX_obs = int(M_obs["m10"]/M_obs["m00"]); cY_obs = int(M_obs["m01"]/M_obs["m00"])
                target_obstacle = {'position':(cX_obs,cY_obs),'contour':obs_contour,'width':w,'height':h,
                                   'side_difference':side_difference,'type':'max_side_diff_obstacle'}
        self.last_known_obstacle_target_for_drawing = target_obstacle
        self.target_obstacle_info = target_obstacle
        return target_obstacle

    # Función de callback del mouse para manejar eventos de clic derecho
    def mouse_callback(self, event, x, y, flags, param):
        if self.image is None: return
        if event == cv2.EVENT_RBUTTONDOWN:
            print(f"Click Der: Meta manual ({x},{y}). Misión cancelada.")
            self.goal_point = (x,y); self.mission_active=False; self.mission_stage=MissionStage.IDLE
            self.object_target_point=None; self.target_obstacle_info=None 
            self.last_known_obstacle_target_for_drawing=None
            self.waypoints=[]; self.path_found=False; self.path_to_final_destination_preview = [] # LIMPIAR PREVIEW
            if self.robot_pose_visually_confirmed:
                self.start_point=(int(self.robot_x),int(self.robot_y)); self.current_path_origin=self.start_point
                self.perform_detection_and_draw_elements(replan=True) 
            else:
                print(f"Meta MANUAL ({x},{y}), esperando detección robot."); self.perform_detection_and_draw_elements(replan=False)

    # Función para construir el mapa de cuadrícula basado en los contornos activos
    def build_grid_map(self):
        if self.original_image is None: self.grid_map=None; self.dist_transform_map_grid=None; return
        h, w = self.original_image.shape[:2]
        self.grid_map = np.zeros((h//GRID_CELL_SIZE,w//GRID_CELL_SIZE),np.uint8)
        for r_idx in range(self.grid_map.shape[0]): 
            for c_idx in range(self.grid_map.shape[1]): 
                cx,cy = c_idx*GRID_CELL_SIZE+GRID_CELL_SIZE//2, r_idx*GRID_CELL_SIZE+GRID_CELL_SIZE//2
                if any(cv2.pointPolygonTest(cnt,(cx,cy),False)>=0 for cnt in self.active_contours):
                    self.grid_map[r_idx,c_idx] = 1
        if self.grid_map is not None and self.grid_dilation_kernel_size>0 and np.any(self.grid_map):
            kernel=np.ones((self.grid_dilation_kernel_size,)*2,np.uint8)
            self.grid_map=cv2.dilate(self.grid_map,kernel,iterations=1)
        if self.grid_map is not None:
            self.dist_transform_map_grid=cv2.distanceTransform(np.where(self.grid_map==1,0,255).astype(np.uint8),cv2.DIST_L2,5)
        else: self.dist_transform_map_grid=None

    # Función para calcular la distancia al obstáculo más cercano desde un punto dado
    def heuristic(self,a,b): return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)
    def normalize_angle(self,angle): 
        while angle > math.pi: angle -= 2*math.pi
        while angle < -math.pi: angle += 2*math.pi
        return angle
    
    # Función para suavizar la trayectoria de los waypoints    
    def smooth_path(self, waypoints, max_turn_deg, chamfer_factor):
        if not waypoints or len(waypoints)<2: return list(waypoints)
        valid_wps=[(float(wp[0]),float(wp[1])) for wp in waypoints if isinstance(wp,(list,tuple)) and len(wp)==2 and all(isinstance(c,(int,float)) for c in wp)]
        if len(valid_wps)<2: return valid_wps
        waypoints, max_turn_rad = valid_wps, math.radians(max_turn_deg)
        chamfer_dist = max(ROBOT_WHEELBASE*0.6,GRID_CELL_SIZE*0.3)*chamfer_factor
        new_path=[waypoints[0]]; i=0
        while i < len(waypoints)-2:
            p1,p2,p3 = new_path[-1],waypoints[i+1],waypoints[i+2]
            if self.heuristic(p1,p2)<1e-3: i+=1; continue
            a12,a23=math.atan2(p2[1]-p1[1],p2[0]-p1[0]),math.atan2(p3[1]-p2[1],p3[0]-p2[0])
            turn=self.normalize_angle(a23-a12)
            if abs(turn)>max_turn_rad:
                d12,d23=self.heuristic(p1,p2),self.heuristic(p2,p3)
                curr_chamfer=min(chamfer_dist,d12*0.45,d23*0.45)
                if curr_chamfer > ROBOT_RADIUS*0.1:
                    ta=(d12-curr_chamfer)/d12 if d12>1e-3 else 0.0; p2a=(p1[0]+ta*(p2[0]-p1[0]),p1[1]+ta*(p2[1]-p1[1]))
                    tb=curr_chamfer/d23 if d23>1e-3 else 0.0; p2b=(p2[0]+tb*(p3[0]-p2[0]),p2[1]+tb*(p3[1]-p2[1]))
                    if self.heuristic(new_path[-1],p2a)>0.1: new_path.append(p2a)
                    if self.heuristic(new_path[-1],p2b)>0.1: new_path.append(p2b)
                elif self.heuristic(new_path[-1],p2)>0.1: new_path.append(p2)
            elif self.heuristic(new_path[-1],p2)>0.1: new_path.append(p2)
            i+=1
        if waypoints and (not new_path or self.heuristic(new_path[-1],waypoints[-1])>0.1): new_path.append(waypoints[-1])
        final_path=[new_path[0]] if new_path else []
        if new_path:
            for k in range(1,len(new_path)):
                if isinstance(new_path[k],(list,tuple)) and len(new_path[k])==2 and all(isinstance(c,(int,float)) for c in new_path[k]):
                    if self.heuristic(final_path[-1],new_path[k])>0.1: final_path.append(new_path[k])
        return final_path

    # Función para encontrar una celda accesible cercana a un punto dado
    def _find_accessible_cell(self, point_px, point_name="point", search_radius_cells=5):
        if self.grid_map is None: print(f"ERROR ({point_name}): grid_map no disponible."); return None
        point_g=(point_px[1]//GRID_CELL_SIZE,point_px[0]//GRID_CELL_SIZE)
        if not(0<=point_g[0]<self.grid_map.shape[0] and 0<=point_g[1]<self.grid_map.shape[1]):
            cl_r,cl_c=max(0,min(point_g[0],self.grid_map.shape[0]-1)),max(0,min(point_g[1],self.grid_map.shape[1]-1))
            og_pg=point_g; point_g=(cl_r,cl_c)
            px_adj=(point_g[1]*GRID_CELL_SIZE+GRID_CELL_SIZE//2,point_g[0]*GRID_CELL_SIZE+GRID_CELL_SIZE//2)
            if og_pg!=point_g: point_px=px_adj
            if not(0<=point_g[0]<self.grid_map.shape[0] and 0<=point_g[1]<self.grid_map.shape[1]):
                print(f"ERROR ({point_name}): {point_g} (para {point_px}) fuera de mapa tras clamp."); return None
        if self.grid_map[point_g[0],point_g[1]]==0: return point_px 
        for r_s in range(1,search_radius_cells+1):
            for dr in range(-r_s,r_s+1):
                for dc in range(-r_s,r_s+1):
                    if abs(dr)<r_s and abs(dc)<r_s: continue 
                    ng=(point_g[0]+dr,point_g[1]+dc)
                    if 0<=ng[0]<self.grid_map.shape[0] and 0<=ng[1]<self.grid_map.shape[1] and self.grid_map[ng[0],ng[1]]==0:
                        return (ng[1]*GRID_CELL_SIZE+GRID_CELL_SIZE//2,ng[0]*GRID_CELL_SIZE+GRID_CELL_SIZE//2)
        print(f"ERROR ({point_name}): No halló celda vecina para {point_px} (radio {search_radius_cells})."); return None

    # Función para planificar la ruta usando A*
    def plan_path_astar(self):
        if not self.start_point or not self.goal_point or self.grid_map is None or not self.robot_pose_visually_confirmed:
            self.waypoints=[]; self.path_found=False; self.planning_in_progress=False; return
        self.planning_in_progress=True
        start_g=(self.start_point[1]//GRID_CELL_SIZE,self.start_point[0]//GRID_CELL_SIZE)
        goal_g=(self.goal_point[1]//GRID_CELL_SIZE,self.goal_point[0]//GRID_CELL_SIZE)
        if not(0<=start_g[0]<self.grid_map.shape[0] and 0<=start_g[1]<self.grid_map.shape[1]): print(f"Error A*: Start_g {start_g} fuera."); self.planning_in_progress=False; self.waypoints=[]; self.path_found=False; return
        if not(0<=goal_g[0]<self.grid_map.shape[0] and 0<=goal_g[1]<self.grid_map.shape[1]): print(f"Error A*: Goal_g {goal_g} fuera."); self.planning_in_progress=False; self.waypoints=[]; self.path_found=False; return
        if self.grid_map[start_g[0],start_g[1]]==1: print(f"Error A*: Start_g {start_g} en obstáculo!"); self.planning_in_progress=False; self.waypoints=[]; self.path_found=False; return
        if self.grid_map[goal_g[0],goal_g[1]]==1: print(f"Error A*: Goal_g {goal_g} en obstáculo!"); self.planning_in_progress=False; self.waypoints=[]; self.path_found=False; return
        
        open_set,came_from,path_reconstructed_grid_cells = [],{},[]
        g_score={(r,c):float('inf') for r in range(self.grid_map.shape[0]) for c in range(self.grid_map.shape[1])}; g_score[start_g]=0
        f_score={(r,c):float('inf') for r in range(self.grid_map.shape[0]) for c in range(self.grid_map.shape[1])}; f_score[start_g]=self.heuristic(start_g,goal_g)
        heapq.heappush(open_set,(f_score[start_g],start_g))
        
        while open_set:
            _,curr_g = heapq.heappop(open_set)
            if curr_g==goal_g:
                path_tmp=[]; curr_path_node=curr_g
                while curr_path_node in came_from: path_tmp.append(curr_path_node); curr_path_node=came_from[curr_path_node]
                path_tmp.append(start_g); path_reconstructed_grid_cells=path_tmp[::-1]; break
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                n_g=(curr_g[0]+dr,curr_g[1]+dc)
                if 0<=n_g[0]<self.grid_map.shape[0] and 0<=n_g[1]<self.grid_map.shape[1] and self.grid_map[n_g[0],n_g[1]]==0:
                    cost=1 if abs(dr)+abs(dc)==1 else math.sqrt(2)
                    d_obs=self.dist_transform_map_grid[n_g[0],n_g[1]] if self.dist_transform_map_grid is not None else GRID_CELL_SIZE
                    penalty=self.obstacle_proximity_weight/(d_obs+1e-6) if d_obs<(GRID_CELL_SIZE*1.5) else 0
                    tent_g=g_score[curr_g]+cost+penalty
                    if tent_g < g_score[n_g]:
                        came_from[n_g]=curr_g; g_score[n_g]=tent_g; f_score[n_g]=tent_g+self.heuristic(n_g,goal_g)
                        if not any(item[1]==n_g for item in open_set): heapq.heappush(open_set,(f_score[n_g],n_g))
        if path_reconstructed_grid_cells:
            raw_wps=[(c*GRID_CELL_SIZE+GRID_CELL_SIZE//2,r*GRID_CELL_SIZE+GRID_CELL_SIZE//2) for r,c in path_reconstructed_grid_cells]
            self.waypoints=self.smooth_path(raw_wps,self.SMOOTHING_MAX_TURN_DEG,self.SMOOTHING_CHAMFER_FACTOR)
            is_short_or_at_goal = not self.waypoints or \
                                  (len(self.waypoints)==1 and self.current_path_origin and self.heuristic(self.current_path_origin,self.waypoints[0])<GRID_CELL_SIZE*0.5) or \
                                  (len(self.waypoints)==0 and self.current_path_origin and self.goal_point and self.heuristic(self.current_path_origin,self.goal_point)<GRID_CELL_SIZE*0.5)
            if is_short_or_at_goal:
                if self.current_path_origin and self.goal_point and self.heuristic(self.current_path_origin,self.goal_point)<GRID_CELL_SIZE*0.8:
                     self.path_found=True
                     if not self.waypoints and self.goal_point: self.waypoints=[self.goal_point] 
                else: self.path_found=False; self.waypoints=[]
            else: self.current_waypoint_index=0; self.path_found=True
        else: self.waypoints=[]; self.path_found=False; print(f"A* no pudo encontrar ruta de {self.start_point} a {self.goal_point}")
        self.planning_in_progress=False

    # Función para generar una ruta para visualización (nuevo método)
    def generate_path_for_visualization(self, start_pixel_coords, goal_pixel_coords): # NUEVO METODO
        if not start_pixel_coords or not goal_pixel_coords or self.grid_map is None: return []
        preview_start_adj = self._find_accessible_cell(start_pixel_coords, "preview_start_vis")
        preview_goal_adj = self._find_accessible_cell(goal_pixel_coords, "preview_goal_vis")
        if not preview_start_adj or not preview_goal_adj: return []
        
        start_g = (preview_start_adj[1]//GRID_CELL_SIZE, preview_start_adj[0]//GRID_CELL_SIZE)
        goal_g = (preview_goal_adj[1]//GRID_CELL_SIZE, preview_goal_adj[0]//GRID_CELL_SIZE)

        if not(0<=start_g[0]<self.grid_map.shape[0] and 0<=start_g[1]<self.grid_map.shape[1]) or \
           not(0<=goal_g[0]<self.grid_map.shape[0] and 0<=goal_g[1]<self.grid_map.shape[1]) or \
           self.grid_map[start_g[0],start_g[1]]==1 or self.grid_map[goal_g[0],goal_g[1]]==1: return []

        open_set,came_from,path_reconstructed_grid_cells = [],{},[]
        g_score={(r,c):float('inf') for r in range(self.grid_map.shape[0]) for c in range(self.grid_map.shape[1])}; g_score[start_g]=0
        f_score={(r,c):float('inf') for r in range(self.grid_map.shape[0]) for c in range(self.grid_map.shape[1])}; f_score[start_g]=self.heuristic(start_g,goal_g)
        heapq.heappush(open_set,(f_score[start_g],start_g))
        
        while open_set: # A* loop (copy from plan_path_astar)
            _,curr_g = heapq.heappop(open_set)
            if curr_g==goal_g:
                path_tmp=[]; curr_path_node=curr_g
                while curr_path_node in came_from: path_tmp.append(curr_path_node); curr_path_node=came_from[curr_path_node]
                path_tmp.append(start_g); path_reconstructed_grid_cells=path_tmp[::-1]; break
            for dr,dc in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]:
                n_g=(curr_g[0]+dr,curr_g[1]+dc)
                if 0<=n_g[0]<self.grid_map.shape[0] and 0<=n_g[1]<self.grid_map.shape[1] and self.grid_map[n_g[0],n_g[1]]==0:
                    cost=1 if abs(dr)+abs(dc)==1 else math.sqrt(2)
                    d_obs=self.dist_transform_map_grid[n_g[0],n_g[1]] if self.dist_transform_map_grid is not None else GRID_CELL_SIZE
                    penalty=self.obstacle_proximity_weight/(d_obs+1e-6) if d_obs<(GRID_CELL_SIZE*1.5) else 0
                    tent_g=g_score[curr_g]+cost+penalty
                    if tent_g < g_score[n_g]:
                        came_from[n_g]=curr_g; g_score[n_g]=tent_g; f_score[n_g]=tent_g+self.heuristic(n_g,goal_g)
                        if not any(item[1]==n_g for item in open_set): heapq.heappush(open_set,(f_score[n_g],n_g))
        if path_reconstructed_grid_cells:
            raw_wps=[(c*GRID_CELL_SIZE+GRID_CELL_SIZE//2,r*GRID_CELL_SIZE+GRID_CELL_SIZE//2) for r,c in path_reconstructed_grid_cells]
            return self.smooth_path(raw_wps,self.SMOOTHING_MAX_TURN_DEG,self.SMOOTHING_CHAMFER_FACTOR)
        return []

    # Función para navegar un paso en la ruta planificada
    def navigate_step(self):
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints) or self.planning_in_progress or not self.robot_pose_visually_confirmed: 
            self.current_mov_dir = 0
            return
        target_wp = self.waypoints[self.current_waypoint_index]
        if self.heuristic((self.robot_x, self.robot_y), target_wp) < GRID_CELL_SIZE * 0.75:
            self.current_waypoint_index += 1
            if self.current_waypoint_index >= len(self.waypoints):
                self.current_mov_dir = 0
                return
            target_wp = self.waypoints[self.current_waypoint_index]
        p1 = self.current_path_origin if self.current_waypoint_index == 0 else self.waypoints[self.current_waypoint_index - 1]
        p2 = target_wp
        dx_p, dy_p = p2[0] - p1[0], p2[1] - p1[1]
        path_ang = math.atan2(dy_p, dx_p)
        head_err = self.normalize_angle(path_ang - self.robot_theta)
        cte_n = dx_p * (p1[1] - self.robot_y) - (p1[0] - self.robot_x) * dy_p
        path_len_sq = dx_p ** 2 + dy_p ** 2
        cte = 0
        if path_len_sq > (1e-3) ** 2:
            cte = cte_n / math.sqrt(path_len_sq)
        steer_cte = math.atan2(self.K_cte_stanley * cte, ROBOT_VELOCITY + 1e-5)
        self.current_steer_angle = np.clip(head_err + steer_cte, -ROBOT_MAX_STEER_ANGLE, ROBOT_MAX_STEER_ANGLE)
        self.current_mov_dir = 1
        v = ROBOT_VELOCITY * self.current_mov_dir
        phi = self.current_steer_angle
        dx_r, dy_r = v * math.cos(self.robot_theta) * SIM_DT, v * math.sin(self.robot_theta) * SIM_DT
        dth_r = (v / ROBOT_WHEELBASE) * math.tan(phi) * SIM_DT if abs(ROBOT_WHEELBASE) > 1e-3 else 0
        px, py = self.robot_x + dx_r, self.robot_y + dy_r
        pth = self.normalize_angle(self.robot_theta + dth_r)
        coll = False
        if self.active_contours:
            for obs_c in self.active_contours:
                if cv2.pointPolygonTest(obs_c, (px, py), True) > -(ROBOT_RADIUS * 0.8):
                    coll = True
                    break
        if coll:
            print("¡Colisión Kinematica!")
            self.current_mov_dir = 0
            self.waypoints = []
            self.path_found = False
        else:
            self.robot_x, self.robot_y, self.robot_theta = px, py, pth
            if not self.robot_path_history or self.heuristic(self.robot_path_history[-1], (px, py)) > ROBOT_RADIUS * 0.2:
                self.robot_path_history.append((int(px), int(py)))
            # Enviar datos de control por socket
            self.enviar_control_por_socket(v, self.current_steer_angle, self.robot_x, self.robot_y, self.robot_theta)
    
    # Función para enviar datos de control por socket (placeholder, debe implementarse)
    def perform_detection_and_draw_elements(self, replan=False):
        if self.original_image is None:
            h_ph,w_ph = getattr(self.image, 'shape', (480,640,3))[:2]
            ph_img = np.ones((h_ph,w_ph,3),np.uint8)*128; cv2.putText(ph_img,"Esperando imagen...",(30,h_ph//2),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
            cv2.imshow(self.window_name,ph_img); return

        self.detect_robot_markers_and_pose(self.original_image)
        img4obs = self.original_image.copy()
        if self.robot_pose_visually_confirmed and self.robot_marker_contours_for_exclusion:
            cv2.drawContours(img4obs,self.robot_marker_contours_for_exclusion,-1,(255,255,255),cv2.FILLED)
        self.process_image_core(image_to_process=img4obs)
        replan_needed = replan

        if self.robot_pose_visually_confirmed:
            curr_pos=(self.robot_x,self.robot_y)
            if not self.mission_active and self.mission_stage==MissionStage.IDLE and not self.goal_point:
                self.target_obstacle_info=self.find_obstacle_with_max_side_difference()
                if self.target_obstacle_info:
                    self.mission_active=True; self.object_target_point=self.target_obstacle_info['position']; self.goal_point=self.object_target_point
                    self.start_point=(int(curr_pos[0]),int(curr_pos[1])); self.current_path_origin=self.start_point; self.mission_stage=MissionStage.GOING_TO_TARGET_OBSTACLE
                    replan_needed=True; self.waypoints=[]; self.path_found=False; self.planning_in_progress=False; self.path_to_final_destination_preview=[]
                    self.last_known_obstacle_target_for_drawing=self.target_obstacle_info
                else: self.mission_active=False; self.mission_stage=MissionStage.IDLE; self.last_known_obstacle_target_for_drawing=None
            if self.mission_active:
                if self.mission_stage==MissionStage.GOING_TO_TARGET_OBSTACLE:
                    if self.object_target_point:
                        if (not self.waypoints or self.current_waypoint_index>=len(self.waypoints)) and self.heuristic(curr_pos,self.object_target_point)<(ROBOT_RADIUS*2.0+GRID_CELL_SIZE*0.5):
                            print(f"Misión: Obst. {self.object_target_point} alcanzado."); self.path_to_final_destination_preview=[] # LIMPIAR PREVIEW
                            if self.dropoff_point is None: h_img,w_img=self.original_image.shape[:2]; self.set_dropoff_point((DROPOFF_X_OFFSET,h_img//2))
                            self.goal_point=self.dropoff_point; self.object_target_point=self.dropoff_point
                            self.start_point=(int(curr_pos[0]),int(curr_pos[1])); self.current_path_origin=self.start_point; self.mission_stage=MissionStage.GOING_TO_DROPOFF
                            replan_needed=True; self.waypoints=[]; self.path_found=False
                            self.last_known_obstacle_target_for_drawing=None; self.target_obstacle_info=None
                        elif (not self.path_found or not self.waypoints) and not self.planning_in_progress: replan_needed=True
                elif self.mission_stage==MissionStage.GOING_TO_DROPOFF:
                    if self.dropoff_point:
                        if (not self.waypoints or self.current_waypoint_index>=len(self.waypoints)) and self.heuristic(curr_pos,self.dropoff_point)<ROBOT_RADIUS*2:
                            print(f"Misión: Entrega {self.dropoff_point} OK."); self.mission_stage=MissionStage.MISSION_COMPLETE; self.mission_active=False
                            self.goal_point=None; self.object_target_point=self.dropoff_point; self.waypoints=[]; self.path_found=False
                        elif (not self.path_found or not self.waypoints) and not self.planning_in_progress: replan_needed=True
        
        if self.mission_active and self.robot_pose_visually_confirmed and self.goal_point and not self.planning_in_progress:
            replan_needed=True
            
        if replan_needed and self.robot_pose_visually_confirmed and self.goal_point and not self.planning_in_progress:
            self.start_point=(int(self.robot_x),int(self.robot_y)); self.current_path_origin=self.start_point
            self.build_grid_map()
            if self.grid_map is None: print("ERROR: Falló grid_map."); self.waypoints=[]; self.path_found=False
            else:
                adj_start=self._find_accessible_cell(self.start_point,"start_A*"); adj_goal=self._find_accessible_cell(self.goal_point,"goal_A*")
                if adj_start and adj_goal:
                    self.start_point,self.goal_point=adj_start,adj_goal
                    self.plan_path_astar()
                    if self.path_found and self.mission_stage==MissionStage.GOING_TO_TARGET_OBSTACLE and self.object_target_point and self.dropoff_point:
                        # print(f"DEBUG: Planificando preview path: {self.object_target_point} -> {self.dropoff_point}")
                        self.path_to_final_destination_preview = self.generate_path_for_visualization(self.object_target_point,self.dropoff_point)
                        # if not self.path_to_final_destination_preview: print(f"DEBUG: No se pudo generar preview path.")
                    elif self.mission_stage!=MissionStage.GOING_TO_TARGET_OBSTACLE: self.path_to_final_destination_preview=[]
                else: print("ERROR: No hallaron celdas accesibles para A*."); self.waypoints=[]; self.path_found=False
        
        if self.path_found and self.waypoints and self.current_waypoint_index<len(self.waypoints) and not self.planning_in_progress and self.robot_pose_visually_confirmed:
            self.navigate_step()
        
        disp_img = self.processed_image_base.copy() if self.processed_image_base is not None else np.zeros_like(self.original_image) if self.original_image is not None else np.zeros((480,640,3),dtype=np.uint8)
        if self.active_contours: cv2.drawContours(disp_img,self.active_contours,-1,(0,128,0),1)
        if self.last_known_obstacle_target_for_drawing and self.mission_stage==MissionStage.GOING_TO_TARGET_OBSTACLE and self.last_known_obstacle_target_for_drawing.get('type')=='max_side_diff_obstacle':
            tgt_info=self.last_known_obstacle_target_for_drawing; cv2.drawContours(disp_img,[tgt_info['contour']],-1,(255,0,255),3); cv2.putText(disp_img,f"Target(D:{tgt_info.get('side_difference',0):.0f})",(tgt_info['position'][0]-60,tgt_info['position'][1]-20),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,255),1)
        if self.dropoff_point:
            color_dp=(128,0,128)
            if (self.mission_stage==MissionStage.GOING_TO_DROPOFF and self.object_target_point==self.dropoff_point) or (not self.mission_active and self.goal_point==self.dropoff_point): color_dp=(255,100,255)
            cv2.circle(disp_img,self.dropoff_point,ROBOT_RADIUS,color_dp,-1); cv2.circle(disp_img,self.dropoff_point,ROBOT_RADIUS+3,color_dp,1); cv2.putText(disp_img,"DropOff",(self.dropoff_point[0]+ROBOT_RADIUS,self.dropoff_point[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        
        # Dibujar ruta de previsualización primero (para que la activa esté encima si se solapan)
        if self.mission_stage == MissionStage.GOING_TO_TARGET_OBSTACLE and self.path_to_final_destination_preview and len(self.path_to_final_destination_preview) >= 2:
            for i in range(len(self.path_to_final_destination_preview)-1): cv2.line(disp_img,tuple(map(int,self.path_to_final_destination_preview[i])),tuple(map(int,self.path_to_final_destination_preview[i+1])),(0,190,255),1) # Amarillo/Naranja delgado

        if self.path_found and self.waypoints and len(self.waypoints)>=2: # Ruta activa
            for i in range(len(self.waypoints)-1): cv2.line(disp_img,tuple(map(int,self.waypoints[i])),tuple(map(int,self.waypoints[i+1])),(255,255,0),2) # Cian, más gruesa
        
        if self.current_path_origin: cv2.circle(disp_img,self.current_path_origin,6,(0,165,255),-1); cv2.putText(disp_img,"Path Start",(self.current_path_origin[0]-25,self.current_path_origin[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,165,255),1)
        if self.goal_point:
            cv2.circle(disp_img,self.goal_point,8,(0,0,255),-1); cv2.putText(disp_img,"A* Goal",(self.goal_point[0]+10,self.goal_point[1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
            if self.object_target_point and self.goal_point!=self.object_target_point and (self.mission_stage==MissionStage.GOING_TO_TARGET_OBSTACLE or self.mission_stage==MissionStage.GOING_TO_DROPOFF):
                cv2.circle(disp_img,self.object_target_point,6,(255,0,128),2); cv2.putText(disp_img,"Mission Target",(self.object_target_point[0]+10,self.object_target_point[1]+15),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,128),1)
        if len(self.robot_path_history)>1:
            for i in range(len(self.robot_path_history)-1): cv2.line(disp_img,self.robot_path_history[i],self.robot_path_history[i+1],(0,100,255),1)
        if self.robot_pose_visually_confirmed or (self.robot_x!=-100):
            rb_center=(int(self.robot_x),int(self.robot_y)); cv2.circle(disp_img,rb_center,ROBOT_RADIUS,(0,0,200),-1)
            ox_end=int(self.robot_x+ROBOT_RADIUS*1.8*math.cos(self.robot_theta)); oy_end=int(self.robot_y+ROBOT_RADIUS*1.8*math.sin(self.robot_theta))
            cv2.line(disp_img,rb_center,(ox_end,oy_end),(255,255,255),2)
            if self.pink_marker_center: cv2.circle(disp_img,self.pink_marker_center,4,(203,192,255),-1)
            if self.yellow_marker_center: cv2.circle(disp_img,self.yellow_marker_center,4,(0,220,220),-1)
        s1,s2="",""
        if self.mission_active: s1=f"Mision: { {0:'Inactiva',1:'-> Obst.Obj.',2:'-> Entrega',3:'Mision OK!'}.get(self.mission_stage,'N/A') }" + (f" ({self.object_target_point[0]},{self.object_target_point[1]})" if self.object_target_point and self.mission_stage in [1,2] else "")
        elif not self.robot_pose_visually_confirmed: s1="Buscando Robot..."
        elif self.goal_point and not self.mission_active: s1=f"Manual -> ({self.goal_point[0]},{self.goal_point[1]})"
        else: s1="Modo Espera/Inactivo"
        if self.planning_in_progress: s2="Planificando Ruta A*..."
        elif not self.path_found and self.goal_point and self.robot_pose_visually_confirmed: s2="Ruta A* no encontrada/Obstaculo"
        elif self.path_found and self.waypoints: s2=f"Navegando... WP {self.current_waypoint_index+1}/{len(self.waypoints)}"
        cv2.putText(disp_img,s1,(10,disp_img.shape[0]-35),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0),1,cv2.LINE_AA)
        if s2: color_s2=(0,0,255) if "no encontrada" in s2 else (255,255,0); cv2.putText(disp_img,s2,(10,disp_img.shape[0]-15),cv2.FONT_HERSHEY_SIMPLEX,0.5,color_s2,1,cv2.LINE_AA)
        cv2.imshow(self.window_name,disp_img); self.final_display_image=disp_img
        # Enviar imagen procesada por socket UDP
        self.generar_plot_y_enviar()

    # Función para generar y enviar datos relevantes por socket UDP usando pickle
    def generar_plot_y_enviar(self):
        try:
            # Generar estructura de datos con información relevante
            data_dict = {
                "robot_pose": [self.robot_x, self.robot_y, self.robot_theta],
                "dropoff_point": self.dropoff_point,
                "robot_path_history": self.robot_path_history,
                "waypoints": self.waypoints,
                "contours": [cv2.boundingRect(c) for c in self.active_contours] if self.active_contours else []
            }
            payload = pickle.dumps(data_dict)
            if not hasattr(self, 'sock'):
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.destino = ("172.20.10.10", 5050)
            size = struct.pack("!I", len(payload))
            self.sock.sendto(size + payload, self.destino)
        except Exception as e:
            print(f"⚠️ Error al enviar datos pickle por socket: {e}")

    # Función para enviar datos de control por socket (placeholder, debe implementarse)
    def run(self):
        print(f"{time.time():.4f}: Entrando a app.run()")
        try:
            cv2.setMouseCallback(self.window_name, self.mouse_callback) # setup_windows_and_trackbars ya se llamó en __init__
            if not self.start_camera(): print("No se pudo iniciar la cámara. Finalizando."); return
            while True:
                if self.cap is None or not self.cap.isOpened():
                    print("Error: Cámara desconectada."); time.sleep(1)
                    if not self.start_camera(): print("Fallo al reiniciar cámara. Saliendo."); break 
                    else: print("Cámara reiniciada."); continue
                ret, frame = self.cap.read()
                if not ret: print("Error: No se pudo leer fotograma."); time.sleep(0.01); continue
                self.original_image=frame; self.image=self.original_image.copy()
                self.perform_detection_and_draw_elements(replan=False) 
                active_nav=(self.path_found and self.waypoints and self.current_waypoint_index<len(self.waypoints) and self.robot_pose_visually_confirmed)
                wait_time = 15 if not self.planning_in_progress and not active_nav else 5 if self.planning_in_progress else max(1,int(SIM_DT*1000*0.75))
                key = cv2.waitKey(wait_time) & 0xFF
                if key == ord('q'): print("Tecla 'q'. Finalizando..."); break
                elif key == ord('s'): 
                    if hasattr(self,'final_display_image'): self.save_result(self.final_display_image)
                elif key == ord('r'): print("Tecla 'r'. Reseteando detección."); self.reset_detection_parameters() 
                elif key == ord('m'): 
                    print("Tecla 'm'. Reseteando Misión.")
                    self.mission_active=False; self.mission_stage=MissionStage.IDLE; self.goal_point=None; self.object_target_point=None
                    self.target_obstacle_info=None; self.last_known_obstacle_target_for_drawing=None
                    self.waypoints=[]; self.path_found=False; self.path_to_final_destination_preview=[]; self.planning_in_progress=False
        except Exception as e: print(f"ERROR INESPERADO en run(): {e}"); traceback.print_exc()
        finally:
            print(f"\n{time.time():.4f}: Limpieza final...");
            if hasattr(self,'cap') and self.cap is not None and self.cap.isOpened(): print("Liberando cámara..."); self.cap.release()
            print("Destruyendo ventanas OpenCV..."); cv2.destroyAllWindows(); 
            if os.name!='nt': cv2.waitKey(1)
            print(f"{time.time():.4f}: Programa finalizado.")

# --- Main ---
if __name__ == "__main__":
    print(f"{time.time():.4f}: Script iniciado.")
    app = ObstacleAvoidanceNavigation()
    print(f"=== Detector y Navegación: Obst. Obj. (No-Borde) -> Entrega ===")
    print(f"  Intentando cámara 1, backend DSHOW (Windows) y fallback, resolución 1080p.")
    print(f"  Se mostrará ruta previsualizada al dropoff cuando se dirija al primer objeto.")
    print(f"  Se intentará encontrar ruta continuamente si el objetivo es inalcanzable temporalmente.")
    print(f"  Controles: 'q': Salir, 's': Guardar, 'r': Reset Detección, 'm': Reset Misión, Click Der: Meta Manual")
    print("Directorio de trabajo:", os.getcwd())
    app.run()