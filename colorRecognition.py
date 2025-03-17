# importing OpenCV library
import cv2
import numpy as np
import time

# Initialize the camera
cam_port = 0
cam = cv2.VideoCapture(cam_port)

# Check if camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

red_positions = []
blue_positions = []

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

print_interval = 5
last_print_time = time.time()

fourcc = cv2.VideoWriter.fourcc(*'mp4v')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))

while True:
    # Read the frame from the camera
    result, frame = cam.read()
    if not result:
        print("Error: Could not read camera frame.")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)


    lower_red1 = np.array([0, 120, 70])  # first red range
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])  # Second red range
    upper_red2 = np.array([180, 255, 255])
    lower_blue = np.array([55, 70, 50])  # Lower and upper bound of blue
    upper_blue = np.array([140, 255, 255])

    BlueMask = cv2.inRange(hsv, lower_blue, upper_blue)
    RedMask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    RedMask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    RedMask = RedMask1 + RedMask2  # Combine both masks

    # Find contours of detected red areas
    BlueContours, _ = cv2.findContours(BlueMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    RedContours, _ = cv2.findContours(RedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    blue_boxes = 0
    for i, contour in enumerate(BlueContours):
        area = cv2.contourArea(contour)
        if area > 2000:  # Filter out small noise
            x, y, w, h = cv2.boundingRect(contour)
            blue_positions.append((x, y, w, h))
            cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)  # Green dot
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue box
            cv2.putText(frame, f"Blue Box", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 0), 2)
            blue_boxes += 1

    red_boxes = 0
    for i, contour in enumerate(RedContours):
        area = cv2.contourArea(contour)
        if area > 2000:  # Filter out small noise
            x, y, w, h = cv2.boundingRect(contour)

            red_positions.append((x, y, w, h))
            cv2.circle(frame, (x + w // 2, y + h // 2), 5, (0, 255, 0), -1)  # Green dot
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # red box
            cv2.putText(frame, f"Red Box", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 2)
            red_boxes += 1

    if time.time() - last_print_time >= print_interval:
        # Print positions of red boxes
        for x, y, w, h in red_positions:
            center_x = x + w // 2
            center_y = y + h // 2
            #print(f"Red box: ({x},{y},{w},{h}) center: ({center_x}, {center_y})")


        # print positions of blue boxes
        for x, y, w, h in blue_positions:
            center_x = x + w // 2
            center_y = y + h // 2
            #print(f"blue box: ({x},{y},{w},{h}) center: ({center_x}, {center_y})")
        last_print_time = time.time()


    out.write(frame)
    cv2.imshow('Camera', frame)
    cv2.imshow('hsv', hsv)
    cv2.imshow('RedContours', RedMask)
    cv2.imshow('BlueContours', BlueMask)
    # Wait for key press and close windows
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()
out.release()