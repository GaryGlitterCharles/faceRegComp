import cv2
import dlib
from imutils import face_utils

# Load pre-trained face detector
detector = dlib.get_frontal_face_detector()

# Load pre-trained facial landmark predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load image
image = cv2.imread("face_image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = detector(gray, 0)

for face in faces:
    # Detect facial landmarks
    shape = predictor(gray, face)
    shape = face_utils.shape_to_np(shape)

    # Draw facial landmarks on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

# Display the image
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
