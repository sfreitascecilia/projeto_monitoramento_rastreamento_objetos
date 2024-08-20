# utilizei o openvc-contrib-python ao invés do openvc-python
# para, posteriormente, tentar usar o tracker
import cv2
from ultralytics import YOLO
import random

# from google.colab.patches import cv2_imshow  # Para executar no Google Colab


# Abrir e ler as classes
with open("class_list.txt", "r") as my_file:
    class_list = my_file.read().splitlines()

# Gerar cores aleatórias para as classes
detection_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                    range(len(class_list))]

# Carregar o modelo YOLOv8n
model = YOLO("yolov8n.pt", "v8")

# Configurações de vídeo
frame_wid, frame_hyt = 640, 480
cap = cv2.VideoCapture("people.mp4")

if not cap.isOpened():
    print("Não foi possível abrir o(a) imagem/vídeo")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar o(a) imagem/vídeo")
        break

    # Redimensionar o quadro para otimizar
    frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Prever detecções
    detect_params = model.predict(source=[frame], conf=0.45, save=False, task="detect", max_det=2)

    # Processar resultados
    for result in detect_params[0]:
        boxes = result.boxes
        for box in boxes:
            clsID = int(box.cls[0].item())
            conf = box.conf[0].item()
            bb = box.xyxy[0].cpu().numpy()

            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), detection_colors[clsID], 3)

            # Exibir nome da classe e confiança
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame, f"{class_list[clsID]} {round(conf * 100, 1)}%",
                        (int(bb[0]), int(bb[1]) - 10), font, 1, (255, 255, 255), 2)

    # Exibir o quadro com detecções
    cv2.imshow('Detecção e Monitoramento', frame)
    # cv2_imshow(frame)  # Para executar no Google Colab

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
