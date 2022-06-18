from cmath import rect
import cv2
import numpy as np

#PARAMETROS PARA LA FUNCION GRABCUT

#Imagen de entrada
imagen = cv2.imread(r'C:\Users\elycu\OneDrive\Escritorio\VisualStudio_Ejemplo\Practicas_Vision_7mo\shein.jpg')
#Rectangulo por medio de ROI
roi = cv2.selectROI(imagen)
rect = roi
#Mascara
mask = np.zeros(imagen.shape[:2],np.uint8)
#Background y foreground model 
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)
#Funcion
cv2.grabCut(imagen, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
imagen = imagen*mask2[:,:,np.newaxis]

#DETECCION DE ESQUINAS

#Convertir imagen a escala de grises
grises = cv2.cvtColor(imagen,cv2.COLOR_BGR2GRAY)
grises = np.float32(grises)
#Funcion
esquinas = cv2.goodFeaturesToTrack(grises, 60, 0.01, 5)
esquinas = np.int0(esquinas)
for esquina in esquinas:
    x,y = esquina.ravel()
    cv2.circle(imagen,(x,y),3,255,-1)
    
#MUESTRA DE RESULTADOS
img_recorte = imagen[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] 
cv2.imshow("Extracción de fondo con esquinas", img_recorte)

cv2.waitKey(0)
cv2.destroyAllWindows()