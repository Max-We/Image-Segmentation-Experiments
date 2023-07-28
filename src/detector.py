import base64
import json
import os

import cv2
import cv2.aruco as aruco
import numpy as np
import requests
from sklearn.cluster import KMeans, DBSCAN


def detect_aruco_cv2(img):
    """
    Detects a marker in an image
    https://stackoverflow.com/questions/74964527/attributeerror-module-cv2-aruco-has-no-attribute-dictionary-get
    """
    # Find a marker
    for dictionary_name in range(0, 100):
        print(f"---\nUsing dictionary #{dictionary_name}")

        aruco_dict = aruco.getPredefinedDictionary(dictionary_name)
        parameters = aruco.DetectorParameters()

        detector = aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(img)

        if len(np.ravel(corners)) != 0:
            print(f"Found {len(np.ravel(corners))} confirmed points.")
            print(f"Drawing markers...")
            output_img = aruco.drawDetectedMarkers(img, corners, ids)
            break

        print(f"Found {len(np.ravel(rejected))} rejected points, no confirmed points.")

    if len(np.ravel(corners)) != 0:
        output_file = "../output/output.jpg"
        print("Saving result...")
        cv2.imwrite(output_file, output_img)
        print(f"Saved result to {output_file}")
        return

    print("No aruco codes found.")


def detect_coins_cv2(img):
    """
    Detects coins in the provided image
    https://www.perplexity.ai/search/2937bde1-d6f8-4029-a24e-8ec3e6dc7707?s=c
    """
    # Pre-process the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 11)
    cv2.imwrite("../output/coins_pre.jpg", blurred)

    # Detect circles
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, minDist=20, param1=100, param2=50, minRadius=1,
                               maxRadius=0)

    # Check if circles were found
    if circles is not None:
        # Convert the coordinates and radius to integers
        circles = np.round(circles[0, :]).astype("int")

        # Loop over the circles
        for (x, y, r) in circles:
            # Perform operations on each circle
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)

    output_file = "../output/coins.jpg"
    cv2.imwrite(output_file, img)
    print(f"Saved result to {output_file}")

def rectangle_detection(img):
    debug_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binarized = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binarized, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # detect all rectangles
    rois = []
    for contour in contours:
        if len(contour) < 4:
            continue
        cont_area = cv2.contourArea(contour)
        if not 1000 < cont_area < 15000: # roughly filter by the volume of the detected rectangles
            continue
        cont_perimeter = cv2.arcLength(contour, True)
        (x, y), (w, h), angle = rect = cv2.minAreaRect(contour)
        rect_area = w * h
        if cont_area / rect_area < 0.8: # check the 'rectangularity'
            continue
        rois.append(rect)

    # save intermediate results in the debug folder
    rois_img = cv2.drawContours(debug_img, contours, -1, (0, 0, 230))
    rois_img = cv2.drawContours(rois_img, [cv2.boxPoints(rect).astype('int32') for rect in rois], -1, (0, 230, 0))
    return rois_img


def detect_legos_cv2(img):
    """
    Detects legos in an image using basic image manipulations with OpenCV

    This function calculates the threshold, finds contours and tries to identify lego bricks from the contour shapes

    Adapted in parts from: https://www.perplexity.ai/search/02fdb65b-e370-4a6e-93cd-e82f0e8407e7?s=c
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("../output/legos_gray.jpg", gray)

    # Threshold -> Blur -> Threshold yields good results:  https://stackoverflow.com/a/57200704
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 51, 3)
    blur = cv2.GaussianBlur(thresh, (31, 31), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cv2.imwrite("../output/legos_thresh.jpg", thresh)

    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through the contours
    area_sizes = []
    poly_approximations = []
    for cnt in contours:
        # Approximate the contour shape to a polygon
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

        # Check if the polygon has minimum four sides, which indicates that it is a rectangle
        # In isometric perspectives, lego cubes can have up to 6 sides
        if 6 >= len(approx) >= 4:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            # Filter out pixel-artefacts
            if area > 1250:
                # Save for the next step
                area_sizes.append(area)
                poly_approximations.append(approx)

    # LEGO bricks are roughly the same size
    # To filter out false positives, cluster all detected areas by size and pick the cluster with most elements in it
    data = np.array(area_sizes).reshape(-1, 1) # Reshape for SciKit K-Means (expects 2D input)
    kmeans = KMeans(n_clusters=3) # Todo: Try out other clustering algorithms, specifically for 1D data
    kmeans.fit(data)

    # Find the cluster with the most elements in it
    labels = kmeans.labels_
    unique_labels, counts = np.unique(labels, return_counts=True)
    most_elements_cluster = unique_labels[np.argmax(counts)]

    # Draw the elements from the largest cluster
    for i, label in enumerate(labels):
        if label == most_elements_cluster:
            cv2.drawContours(img, [poly_approximations[i]], 0, (0, 255, 0), 2)

    output_file = "../output/legos.jpg"
    cv2.imwrite(output_file, img)
    print(f"Saved result to {output_file}")


def detect_objects_yolo(img):
    """
    Uses YOLOv3 to detect objects in an image
    Adapted from: https://opencv-tutorial.readthedocs.io/en/latest/yolo/yolo.html
    """
    if not os.path.exists('src/yolo/yolov3.weights'):
        print("Please download yolov3.weights from https://pjreddie.com/media/files/yolov3.weights "
              "and place them under the ./yolo folder")
        return

    # Load names of classes and get random colors
    classes = open('yolo/coco.names').read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

    # Give the configuration and weight files for the model and load the network.
    net = cv2.dnn.readNetFromDarknet('yolo/yolov3.cfg', 'yolo/yolov3.weights')
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # determine the output layer
    ln = net.getLayerNames()
    # print(len(net.getUnconnectedOutLayers()))
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    outputs = net.forward(ln)

    def trackbar2(x):
        confidence = x / 100
        r = r0.copy()
        for output in np.vstack(outputs):
            if output[4] > confidence:
                x, y, w, h = output[:4]
                p0 = int((x - w / 2) * 416), int((y - h / 2) * 416)
                p1 = int((x + w / 2) * 416), int((y + h / 2) * 416)
                cv2.rectangle(r, p0, p1, 1, 1)

    r0 = blob[0, 0, :, :]

    boxes = []
    confidences = []
    classIDs = []
    h, w = img.shape[:2]

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > 0.5:
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    output_file = "./output/objects-yolo.jpg"
    cv2.imwrite(output_file, img)
    print(f"Saved result to {output_file}")


def detect_legos_yolo_custom(img):
    """
    Uses a custom trained YOLO network to detect lego bricks in an image

    Details about the model are on Huggingface:
    - Huggingface model card: https://huggingface.co/mw00/yolov7-lego
    - Huggingface space: https://huggingface.co/spaces/mw00/yolov7-lego
    """

    # Encode the image to a base64 string
    retval, buffer_img = cv2.imencode('.jpg', img)
    base64_img = base64.b64encode(buffer_img)

    # This is a custom trained YOLO network to detect Lego bricks for this project
    # The details about the training data as well as the performance is listed in the model card.
    # There is also the notebook included in which the model has been trained.
    # The HF-space is configured to return only predictions with a confidence > 80% from the model
    # Huggingface model card: https://huggingface.co/mw00/yolov7-lego
    # Huggingface space: https://huggingface.co/spaces/mw00/yolov7-lego
    url = "https://mw00-yolov7-lego.hf.space/api/predict"
    img_format = "data:image/jpg;base64,"

    # Build request
    payload = {
        "data": [
            img_format + base64_img.decode('utf-8'),
            "zero-shot-1000-single-class"  # this is the only trained model atm
        ]
    }
    headers = {'Content-Type': 'application/json'}

    # Send request
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    # Evaluate response
    if not response.ok:
        print(f"Error making request to model endpoint: {response.text}")
        return

    response_data = response.json()
    output_base64_img = response_data["data"][0]

    output_img_bytes = base64.b64decode(output_base64_img.split(",")[1])
    output_img_array = np.frombuffer(output_img_bytes, dtype=np.uint8)
    output_img = cv2.imdecode(output_img_array, flags=cv2.IMREAD_COLOR)

    # Save result as file
    output_file = "../output/lego-yolo.jpg"
    cv2.imwrite(output_file, output_img)
    print(f"Saved result to {output_file}")
