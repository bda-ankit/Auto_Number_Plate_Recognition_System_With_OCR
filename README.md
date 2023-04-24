# Auto_Number_Plate_Recognition_System_With_OCR
Advance Python project on Auto Number Plate Recognition System With OCR

# Important Development Modules
1. OpenCV for identify license plates
2. Pytesseract for extract characters and numbers from the plate

# Component
Install Tesseract-OCR in your system<br />
Link - https://github.com/UB-Mannheim/tesseract/wiki

# Approach

1. Install OpenCV and Pytesseract
2. Read image path
3. Convert image to gray scale using OpenCV
4. Blur image to reduce noise 
5. Perform edge detection to recognize plate
6. Find contours in the edged image
7. Mask the part other than the number plate [Final Image]
8. Configuration for tesseract
9. Extract number plate using Pytesseract
10. Get accuracy of extracted number plates
11. Store logs in csv file

# Conclusion
We passed the final processed image to the Tesseract OCR engine to extract the number from the license number plate. We can perform this image process for license number plates with not 100% accuracy sometimes.
