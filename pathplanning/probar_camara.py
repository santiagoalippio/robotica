giimport cv2

def detectar_camara_valida():
    print("🔍 Buscando cámara válida...")

    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
                print(f"✅ Cámara {i} está funcionando.")
                cap.release()
                return i
            else:
                print(f"⚠️ Cámara {i} abierta, pero sin imagen válida.")
            cap.release()
        else:
            print(f"❌ Cámara {i} no se pudo abrir.")
    return -1  # Ninguna cámara válida encontrada