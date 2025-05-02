import numpy as np
import cv2
from PIL import Image

# Load model files
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'
kernel_path = 'models/pts_in_hull.npy'

# Load neural net and cluster centers
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
points = np.load(kernel_path)

# Format cluster centers as blobs
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Main function to colorize a PIL image and return a PIL image
def colorize_image_pil(pil_img):
    # Convert PIL to OpenCV (RGB -> BGR)
    img = np.array(pil_img.convert("RGB"))[:, :, ::-1]
    img_float = img.astype("float32") / 255.0

    # Convert to LAB color space
    lab = cv2.cvtColor(img_float, cv2.COLOR_BGR2LAB)
    L = cv2.split(cv2.resize(lab, (224, 224)))[0]
    L -= 50

    # Predict a/b channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    # Combine with original L channel
    L_original = cv2.split(lab)[0]
    colorized = np.concatenate((L_original[:, :, np.newaxis], ab), axis=2)
    colorized_bgr = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized_bgr = np.clip(255.0 * colorized_bgr, 0, 255).astype("uint8")

    # Convert BGR to RGB and return as PIL
    return Image.fromarray(colorized_bgr[:, :, ::-1])
