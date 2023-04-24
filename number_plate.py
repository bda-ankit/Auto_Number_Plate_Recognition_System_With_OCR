import numpy as np
import cv2
import imutils
import pytesseract
import pandas as pd
import time

pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# read and resize image to the required size
image = cv2.imread('MH12DE1433.jpg')
image = imutils.resize(image, width=500)
cv2.imshow("Original Image", image)

# convert to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Conversion", gray)

# blur to reduce noise
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Bilateral Filter", gray)

# perform edge detection
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("Canny Edges", edged)

# find contours in the edged image
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts=sorted(cnts, key = cv2.contourArea, reverse = True)[:50]

NumberPlateCnt = None 
count = 0

# loop over contours
for c in cnts:
    
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.01 * peri, True)
    
    # if the approximated contour has four points, then assume that screen is found
    if len(approx) == 4:  
        NumberPlateCnt = approx 
        break

# mask the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[NumberPlateCnt],0,255,-1)
new_image = cv2.bitwise_and(image,image,mask=mask)
cv2.namedWindow("Final Image",cv2.WINDOW_NORMAL)
cv2.imshow("Final Image",new_image)

# configuration for tesseract
config = ('-l eng --oem 1 --psm 3')

# run tesseract OCR on image
demo_number_plate = pytesseract.image_to_string(new_image, config=config)

target_number_plate = ['MH12DE1433']
print("Target Number Plate", "\t", "Predicted Target Number Plate", "\t", "Accuracy")  
print("-----------------------", "\t", "-------------------------", "\t\t", "---------")

predicted_target_number_plate = demo_number_plate.replace(" ", "")
predicted_target_number_plate = list(demo_number_plate.split())
for target_number_plate, predicted_target_number_plate in zip(target_number_plate, predicted_target_number_plate):
        acc = "0 %"  
        number_matches = 0
        if target_number_plate == predicted_target_number_plate:  
            acc = "100 %"  
        else:  
            if len(target_number_plate) == len(predicted_target_number_plate):
                for o, p in zip(target_number_plate, predicted_target_number_plate):
                    if o == p:  
                        number_matches += 1  
                acc = str(round((number_matches / len(target_number_plate)), 2) * 100)  
                acc += "%"  
print(target_number_plate, "\t\t", predicted_target_number_plate, "\t\t\t", acc)

# data is stored in CSV file
raw_data = {'Date':[time.asctime(time.localtime(time.time()))],'Target Number Plate':[target_number_plate],'Predicted Target Number Plate':[predicted_target_number_plate],'Accuracy':[acc]}
df = pd.DataFrame(raw_data)
df.to_csv('data.csv',mode='a')

import mysql.connector
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password=""
)

cursor = mydb.cursor()
cursor.execute("SHOW databases")
print("\n----Databases----")
for databases in cursor:
  print(databases)

cursor.execute("USE data")

print("\n----Tables inside Database[data]----")
cursor.execute("SHOW TABLES")
for tables in cursor:
  print(tables)

print("\n----Total records inside Table[target_info]----")
cursor.execute("SELECT *FROM target_info")
for fetch in cursor:
  print(fetch)

print("\n----Target Information----")
cursor.execute("SELECT target_name,target_gender,target_vehicle_type,target_number_plate FROM target_info where target_id='3'")
for target in cursor:
  print(target)











