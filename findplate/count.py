import cv2 
from matplotlib import pyplot as plt 
img = cv2.imread("images\image3.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

haarplacapare = cv2.CascadeClassifier('haarcascade\haarplacapare.xml')
found = haarplacapare.detectMultiScale(img_gray,  minSize =(2, 22))
quant = len(found)
if quant != 0: 
    for (x, y, width, height) in found: 
      cv2.rectangle(img_rgb, (x, y),(x + height, y + width),(9, 9, 300), 5)
      
 
plt.imshow(img_rgb) 
plt.show()
