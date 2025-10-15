import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import cv2
main_folder_path = r"C:\Users\intel\OneDrive\Desktop\aiml\trainingSet\trainingSet"
x = []
y = []

for label in os.listdir(main_folder_path):
    folder_path = os.path.join(main_folder_path, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))  
        x.append(img.flatten())
        y.append(int(label))

x = np.array(x)
y = np.array(y)


model = KNeighborsClassifier(n_neighbors=3)
model.fit(x, y)
print("Model trained successfully")


model_path = "knn_digit_model.joblib"
joblib.dump(model, model_path)
print("Model saved successfully")

camera = cv2.VideoCapture(0)
print("Starting camera..")

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to grab frame")
        break

   
    cv2.imshow("Live Feed - Press 's' to capture", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):  
        print("Capturing image..")
        break


camera.release()
cv2.destroyAllWindows()

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
roi = cv2.resize(gray, (50, 50))  
flattened = roi.flatten().reshape(1, -1)


predicted_digit = model.predict(flattened)[0]
print(f"Predicted Digit: {predicted_digit}")


cv2.imshow("Captured Digit", gray)
cv2.waitKey(0)
cv2.destroyAllWindows()