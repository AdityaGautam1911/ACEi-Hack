import cv2

# Load pre-trained model (You can change the model if needed)
net = cv2.dnn.readNet(cv2.samples.findFile("yolov3.weights"), cv2.samples.findFile("yolov3.cfg"))

# Load classes (coco.names contains the class names)
with open(cv2.samples.findFile("coco.names"), "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set up video capture (replace 'your_video.mp4' with the actual video file path)
cap = cv2.VideoCapture('your_video.mp4')

while cv2.waitKey(1) < 0:
    # Read a frame from the video
    ret, frame = cap.read()

    # Break the loop if the video is finished
    if not ret:
        break

    # Get the height and width of the frame
    height, width, _ = frame.shape

    # Convert the frame to a blob
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Set the input to the model
    net.setInput(blob)

    # Get the output layer names
    out_layer_names = net.getUnconnectedOutLayersNames()

    # Run forward pass to get detections
    detections = net.forward(out_layer_names)

    # Loop over the detections and draw bounding boxes around people
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'person':
                # Calculate bounding box coordinates
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                w = int(obj[2] * width)
                h = int(obj[3] * height)

                # Draw bounding box
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("People Detection", frame)

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
