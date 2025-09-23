import cv2 ## image processing
from io import BytesIO ## in-memory file operations. BytesIO -> Input and Output
import numpy as np ## arrays of images

## Steps - Process Image
## 1) Take an image and store it in temporary memory
## 2) Convert that image into numpy array
## 3) Convert from numpy array to normal OpenCV friendly image
## 4) Convert that image to grayscale image -> best for facial detection
## 5) Load a pre-trained frontal face detector
## 6) Use detector to detect faces on a particular image
## 7) Detect the main subject (largest) face on that particular image
## 8) Detect coordinates of the face & draw a bounding box
## 9) Convert the image which is currently in OpenCV format into .jpg 
## 10) Return that .jpg image. 
## 11) Can't return the image directly, so we will return it in a form of bytes.
## -> so that later we can convert that into normal image

def process_image(image_file): ##image_file -> Image that user uploads
    ## To store the image when user uploads. We can't store all images in the local storage -> lack of storage
    ## Temporary Storage -> to store images for sometime in the form of cache file.
    in_memory_file = BytesIO() ## Cache memory
    image_file.save(in_memory_file) # save the file in the memory storage

    # To deal with the bytedata of a particular image. image_bytes is a variable. All files are now stored in there.
    image_bytes = in_memory_file.getvalue() # Retrieve the entire contents of the images files in the form of bites
    # Convert byte data into numpy arrays. OpenCV library can only deal with numpy arrays.
    nparr = np.frombuffer(image_bytes, np.uint8)

    ## Convert numpy array into opencv compatible format
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) ## Decoding numpy array to an actual image

    ## Convert BGR image to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ## Why we are converting -> Face detection -> Grayscale is the best

    ## load pre-trained face detection model
    ## Haar Cascade
    ## Only the front part of the face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    ## Detect grayscale images and store in faces variable
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0: ## No face detected
        return image_bytes, None
    
    # In case more than 1 face in the image, get the biggest face
    largest_face = max(faces, key = lambda r:r[2] * r[3])

    ## 4 things we use to detect the position and size of the image
    ## x-axis, y-axis, width, height
    ## Splitting the largest face
    ## (14, 20, 15, 18) -> x=14, y=20, w=15, h=18
    (x, y, w, h) = largest_face

    ## Create a rectangle on the face -> object detection model -> rectangular boxes
    ## On main image (img), x1 y1, x2 y2, color of the box, thickness of the box
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)

    ## Encoding the image from rectangle to JPEG format
    is_success, buffer = cv2.imencode(".jpg", img) ## after encoding, we store it inside the buffer variable

    # We return the encoded image, in the format of bytes so that we can convert them into normal image and the coordinates of the largest face
    return buffer.tobytes(), largest_face