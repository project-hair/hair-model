import random
import cv2
from ultralytics import YOLO

def perform_object_detection():
    # Function to read class data from file
    def read_class_data(file_path):
        class_list = []
        with open(file_path, "r") as file:
            data = file.read()
            class_list = data.split("\n")
        return class_list

    # Read shape class data
    shape_class_list = read_class_data("utils/coco.txt")

    # Read gender class data
    gender_class_list = read_class_data("utils/gender.txt")

    # Generate random colors for class list
    detection_colors = []
    for i in range(len(shape_class_list)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        detection_colors.append((b, g, r))

    # Load a pretrained YOLOv8n shape_model
    shape_model = YOLO("./utils/yolo8n-shapes.pt", "v8")
    gender_model = YOLO("./utils/yolov8-gender.pt", "v8")

    # Vals to resize video frames | small frame optimise the run
    frame_wid = 640
    frame_hyt = 480

    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("img.jpg")

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        #  resize the frame | small frame optimize the run
        # frame = cv2.resize(frame, (frame_wid, frame_hyt))

        # Predict on image
        shape_detect_params = shape_model.predict(source=[frame], conf=0.45, save=False)
        gender_detect_params = gender_model.predict(source=[frame], conf=0.45, save=False)

        # Convert tensor array to numpy
        SDP = shape_detect_params[0].numpy()
        GDP = gender_detect_params[0].numpy()

        if len(SDP) != 0 and len(GDP) != 0:
            for j in range(len(shape_detect_params[0])):
                for k in range(len(gender_detect_params[0])):
                    boxes = shape_detect_params[0].boxes
                    box = boxes[j]
                    s_clsID = box.cls.numpy()[0]
                    s_cs = shape_class_list[int(s_clsID)]

                    g_boxes = gender_detect_params[0].boxes
                    g_box = g_boxes[k]
                    g_clsID = g_box.cls.numpy()[0]
                    g_cs = gender_class_list[int(g_clsID)]

                    # print(g_cs, s_cs)

        if len(SDP) != 0:
            for i in range(len(shape_detect_params[0])):
                # print(i)

                boxes = shape_detect_params[0].boxes
                box = boxes[i]  # returns one box
                clsID = box.cls.numpy()[0]
                conf = box.conf.numpy()[0]
                bb = box.xyxy.numpy()[0]

                cv2.rectangle(
                    frame,
                    (int(bb[0]), int(bb[1])),
                    (int(bb[2]), int(bb[3])),
                    detection_colors[int(clsID)],
                    3,
                )

                # Display class name and confidence
                font = cv2.FONT_HERSHEY_COMPLEX
                cv2.putText(
                    frame,
                    gender_class_list[int(g_clsID)] + "_" + shape_class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                    (int(bb[0]), int(bb[1]) - 10),
                    font,
                    1,
                    (255, 255, 255),
                    2,
                )

        # Display the resulting frame
        cv2.imshow("ObjectDetection", frame)

        # Terminate run when "Q" pressed
        if cv2.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start object detection
perform_object_detection()
