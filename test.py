from ultralytics import YOLO
import cv2

# Load your YOLOv8 model (change the path if needed)
model = YOLO("D:\BGK\8th sem\Pedestrian detection-improved\models\yolov10x.pt")  # Replace with your actual model path

# Load the video file (change "video.mp4" to your video path or 0 for webcam)
cap = cv2.VideoCapture("D:\BGK\8th sem\Pedestrian detection-improved\videos\test.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Run inference on the frame
    results = model(frame)[0]

    # Annotate the frame with detection boxes, labels, etc.
    annotated_frame = results.plot()

    # Show the annotated frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # Press ESC to quit
    if cv2.waitKey(1) == 27:
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()