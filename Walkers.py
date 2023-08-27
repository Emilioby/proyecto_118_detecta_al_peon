import cv2
#Seleccionar el haarscade:
haCu = input('ingresa la ruta del haarcascade')
newHaCu= haCu.replace('\\', '/')

# video aleatorio 
custom = input('ingresa el video a seleccionar')
newCustom= custom.replace('\\', '/')

# Crea nuestro body classifier
body_classifier = cv2.CascadeClassifier(newHaCu)

# Inicializa video capture para el archivo de video
cap = cv2.VideoCapture(newCustom)

# Pasa el bucle ya que el video se haya cargado correctamente
while True:
    # Lee el primer cuadro
    ret, frame = cap.read()

    if not ret:
        break

    # Convierte cada cuadro en escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pasa los cuadros a nuestro body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    # Extrae los cuadros delimitadores de los cuerpos identificados
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (222, 0,5), 2)

    # Muestra el cuadro con los rectángulos dibujados
    cv2.imshow('Detected Bodies', frame)

    if cv2.waitKey(1) == 32:  # 32 es la tecla espaciadora
        break

cap.release()
cv2.destroyAllWindows()