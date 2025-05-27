import cv2

def detectar_camara_valida():
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


if __name__ == "__main__":
    print("🔍 Escaneando todas las cámaras disponibles...\n")
    for i in range(5):
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        shape_info = None if frame is None else frame.shape
        print(f"Cámara {i}: ret={ret}, frame shape={shape_info}")
        if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            cv2.imshow(f"Cámara {i}", frame)
            print(f"✅ Mostrando imagen de cámara {i}. Presiona una tecla para cerrar la ventana.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()