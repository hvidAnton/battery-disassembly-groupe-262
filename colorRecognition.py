import cv2
import numpy as np
import time

class CameraHandler:
    def __init__(self, cam_port=1):
        self.cam = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW)
        if not self.cam.isOpened():
            raise TypeError("Error: Could not open camera.")

        self.frame_width = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output.avi', self.fourcc, 20.0, (self.frame_width, self.frame_height))

    def get_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            raise TypeError("Error: Could not read camera frame.")
        return frame

    def release(self):
        self.cam.release()
        self.out.release()


class ObjectDetector:
    def __init__(self):
        self.red_positions = []
        self.blue_positions = []
        self.red_center_points = []
        self.blue_center_points = []

        # Red color ranges
        self.lower_red1 = np.array([0, 120, 70])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

        # Blue color ranges
        self.lower_blue = np.array([100, 150, 50])
        self.upper_blue = np.array([130, 255, 255])

    def process_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Masks
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = red_mask1 + red_mask2

        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.detect_objects(blue_contours, frame, (255, 0, 0), "Blue Box", self.blue_positions, self.blue_center_points)
        self.detect_objects(red_contours, frame, (0, 0, 255), "Red Box", self.red_positions, self.red_center_points)

    def detect_objects(self, contours, frame, color, label, positions, center_points):
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(contour)
                positions.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                centers = self.get_centers(x, y, w, h)
                for center_x, center_y in centers:
                    center_points.append((center_x, center_y))
                    cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)

    def get_centers(self, x, y, w, h):
        n_centers = max(1, round(w / h))
        print(n_centers)
        step = w / (2 * n_centers)
        return [(int(x + step * (2 * i + 1)), int(y + h / 2)) for i in range(n_centers)]


class Logger:
    def __init__(self, print_interval=2):
        self.print_interval = print_interval
        self.last_print_time = time.time()

    def log_positions(self, red_positions, blue_positions):
        if time.time() - self.last_print_time >= self.print_interval:
            for x, y, w, h in red_positions:
                center_x = x + w // 2
                center_y = y + h // 2
                #print(f"Red box: ({x},{y},{w},{h}) center: ({center_x}, {center_y})")

            for x, y, w, h in blue_positions:
                center_x = x + w // 2
                center_y = y + h // 2
                #print(f"Blue box: ({x},{y},{w},{h}) center: ({center_x}, {center_y})")

            self.last_print_time = time.time()


def main():
    try:
        camera_handler = CameraHandler()
        object_detector = ObjectDetector()
        logger = Logger()

        while True:
            frame = camera_handler.get_frame()
            object_detector.process_frame(frame)
            logger.log_positions(object_detector.red_positions, object_detector.blue_positions)
            cv2.imshow('Camera', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): #quit with "q"
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        camera_handler.release()


if __name__ == "__main__":
    main()

