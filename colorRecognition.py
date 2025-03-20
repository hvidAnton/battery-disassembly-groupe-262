from email.policy import default
from operator import index

import cv2
import numpy as np
import time


class CameraHandler:
    def __init__(self, cam_port=0):
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

        # Reset positions to store only the current frame's data
        self.red_positions = []
        self.blue_positions = []

        # Masks
        blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        red_mask = red_mask1 + red_mask2

        #definde contours
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.blue_center_points.clear(), self.red_center_points.clear()
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
        n_centers = max(1, round((w * 1.2) / h))
        step = w / (2 * n_centers)
        return [(int(x + step * (2 * i + 1)), int(y + h / 2)) for i in range(n_centers)]

class Logger:
    def __init__(self, grid_cols=4, grid_rows=2):
        self.grid_cols = grid_cols  # 4
        self.grid_rows = grid_rows  # 2

    def get_grid_state(self, red_positions, blue_positions , red_center_points, blue_center_points):
        grid_state = [["Empty" for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        print(grid_state)

        min_x = min((x for x, y, w, h in red_positions + blue_positions), default=0)
        min_y = min((y for x, y, w, h in red_positions + blue_positions), default=0)
        max_w = max((x + w for x, y, w, h in red_positions + blue_positions), default=0)
        max_h = max((y + h for x, y, w, h in red_positions + blue_positions), default=0)

        # Determine cell width and height for the 2x4 grid
        cell_width = (max_w - min_x) / self.grid_cols
        cell_height = (max_h - min_y) / self.grid_rows


        for center_x, center_y in red_center_points + blue_center_points:
            col = min(self.grid_cols - 1, max(0, int((center_x - min_x) / cell_width)))
            row = min(self.grid_rows - 1, max(0, int((center_y - min_y) / cell_height)))

            # Determine the color based on which list the center came from
            if (center_x, center_y) in red_center_points:
                color = "Red"
            elif (center_x, center_y) in blue_center_points:
                color = "Blue"
            else:
                raise TypeError(f"Error: center not in red_positions or blue_positions.{center_x, center_y} {red_center_points + blue_center_points}")

            grid_state[row][col] = color

        return grid_state

def get_grid_index_state(self, x, y, cell_width, cell_height):
    col = min(self.grid_cols - 1, max(0, int(x // cell_width)))
    row = min(self.grid_rows - 1, max(0, int(y // cell_height)))
    return row * self.grid_cols + col




def main():
    try:
        camera_handler = CameraHandler()
        object_detector = ObjectDetector()
        logger = Logger()

        while True:
            frame = camera_handler.get_frame()
            object_detector.process_frame(frame)


            if cv2.waitKey(1) & 0xFF == ord('a'):
                grid_state = logger.get_grid_state(object_detector.red_positions, object_detector.blue_positions, object_detector.red_center_points, object_detector.blue_center_points)
                print("Grid State:", grid_state)


            cv2.imshow('Camera', frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):  # quit with "q"
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        camera_handler.release()


if __name__ == "__main__":
    main()
