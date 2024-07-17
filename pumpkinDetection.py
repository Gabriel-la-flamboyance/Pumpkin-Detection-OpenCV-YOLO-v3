import cv2  
import numpy as np          

# Charger le modèle YOLO avec les poids et la configuration
net = cv2.dnn.readNet('yolov3_training_final.weights', 'yolov3_testing.cfg')

# Lire les noms des classes depuis un fichier
classes = []
with open("classes.txt", "r") as f:
    classes = f.read().splitlines()

# Ouvrir le fichier vidéo
cap = cv2.VideoCapture('test5.mp4')
font = cv2.FONT_HERSHEY_PLAIN 
# Générer des couleurs aléatoires pour les boîtes de détection
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    # Lire une image de la vidéo
    _, img = cap.read()
    height, width, _ = img.shape

    # Préparer l'image pour YOLO
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    # Parcourir les sorties de chaque couche
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Calculer les coordonnées de la boîte englobante
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # Appliquer la suppression des non-maxima pour éviter les boîtes redondantes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            color = colors[i]
            # Dessiner la boîte englobante et le label sur l'image
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255,255,255), 2)
            
            ###########################################
            # Dessiner un cercle au centre de la boîte englobante
            x2 = x + int(w/2)
            y2 = y + int(h/2)
            cv2.circle(img, (x2, y2), 4, (0, 255, 0), -1)
            
            # Imprimer les coordonnées de la boîte englobante entière 
            print(x, y)
            
            # Ajouter les coordonnées du centre de la boîte sur l'image
            textCoor = "x: " + str(x2) + ", y: " + str(y2)
            cv2.putText(img, textCoor, (x2 - 10, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            ###########################################     

    # Afficher l'image avec les détections
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:  # Si la touche 'Esc' est pressée, quitter la boucle
        break

# Libérer la capture vidéo et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()
