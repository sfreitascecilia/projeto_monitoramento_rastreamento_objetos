# utilizei o openvc-contrib-python ao invés do openvc-python
# para tentar usar o tracker
import cv2
from ultralytics import YOLO

# from google.colab.patches import cv2_imshow

# Carregar o modelo YOLOv8 pré-treinado
model = YOLO("yolov8n.pt")

# Configurar a captura de vídeo
video_source = 'people.mp4'
cap = cv2.VideoCapture(video_source)

# Inicializar o rastreador
tracker = cv2.TrackerCSRT.create()  # Usando um rastreador mais robusto

# Variáveis para controle do rastreamento
init_tracking = False
bbox = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o vídeo")
        break

    # Realizar a detecção com o modelo YOLOv8
    results = model(frame)
    # Assumindo que 'results' contém 'boxes' com coordenadas xyxy
    detections = results[0].boxes.xyxy.cpu().numpy()

    if not init_tracking:
        # Iniciar rastreamento com a primeira detecção
        if len(detections) > 0:
            # Selecionar a primeira detecção com confiança alta
            x1, y1, x2, y2 = detections[0][:4]
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            if bbox[2] > 0 and bbox[3] > 0:  # Verificar se a largura e altura são válidas, para não gerar erro
                tracker.init(frame, bbox)
                init_tracking = True

    if init_tracking:
        # Atualizar o rastreador com o frame atual
        success, bbox = tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in bbox]
            # Desenhar a caixa delimitadora
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Rastreamento', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
        else:
            init_tracking = False  # Se o rastreamento falhar, reiniciar o rastreamento

    # Mostrar o vídeo com as detecções e rastreamento
    cv2.imshow('Detecção e Rastreamento', frame)
    # cv2_imshow(frame)  # Para executar no Google Colab

    # Sair do loop quando a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
