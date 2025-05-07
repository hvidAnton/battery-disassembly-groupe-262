import cv2
import numpy as np
import time

# CameraHandler class is responsible for managing the camera input and video output
class CameraHandler:
    def __init__(self, cam_port=0):
        # Initialize the camera
        self.cam = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW)
        if not self.cam.isOpened():
            raise TypeError("Error: Could not open camera.")

        # Reduce resolution for faster processing
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Capture frame dimensions and video writer setup
        self.frameWidth = int(self.cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frameHeight = int(self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        self.out = cv2.VideoWriter('output.avi', self.fourcc, 20.0, (self.frameWidth, self.frameHeight))

    # Capture a single frame from the camera
    def get_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            raise TypeError("Error: Could not read camera frame.")
        return frame

    # Release the camera and video writer resources
    def release(self):
        self.cam.release()
        self.out.release()

# ObjectDetector class is responsible for detecting red and blue objects in a video frame
class ObjectDetector:
    def __init__(self):
        # Initialize lists to store detected positions and center points
        self.red_positions = []
        self.blue_positions = []
        self.red_center_points = []
        self.blue_center_points = []

        # Define HSV color ranges for red and blue detection
        self.lowerRed1 = np.array([0, 120, 70])  # Lower range for red color
        self.upperRed1 = np.array([10, 255, 255])  # Upper range for red color
        self.lowerRed2 = np.array([170, 120, 70])  # Secondary lower range for red
        self.upperRed2 = np.array([180, 255, 255])

        self.lowerBlue = np.array([100, 150, 50])  # Lower range for blue color
        self.upperBlue = np.array([130, 255, 255])  # Upper range for blue color

    # Process a video frame to detect objects
    def process_frame(self, frame):
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Reset positions for the current frame
        self.red_positions = []
        self.blue_positions = []

        # Create masks for detecting red and blue objects
        blue_mask = cv2.inRange(hsv, self.lowerBlue, self.upperBlue)
        red_mask1 = cv2.inRange(hsv, self.lowerRed1, self.upperRed1)
        red_mask2 = cv2.inRange(hsv, self.lowerRed2, self.upperRed2)
        red_mask = red_mask1 + red_mask2

        # Find contours for blue and red objects
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Clear previous center points
        self.blue_center_points.clear(), self.red_center_points.clear()

        # Detect and annotate the objects in the frame
        self.detect_objects(blue_contours, frame, (255, 0, 0), "Blue Box", self.blue_positions, self.blue_center_points)
        self.detect_objects(red_contours, frame, (0, 0, 255), "Red Box", self.red_positions, self.red_center_points)

    # Helper method to detect objects and draw annotations on the frame
    def detect_objects(self, contours, frame, color, label, positions, center_points):
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 2000:  # Filter out small areas
                x, y, w, h = cv2.boundingRect(contour)
                positions.append((x, y, w, h))
                # Draw bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Calculate and annotate the center points
                centers = self.get_centers(x, y, w, h)
                for centerX, centerY in centers:
                    center_points.append((centerX, centerY))
                    cv2.circle(frame, (centerX, centerY), 5, (0, 255, 0), -1)

    # Static method to calculate center points of an object
    @staticmethod
    def get_centers(x, y, w, h):
        n_centers = max(1, round((w * 1.2) / h))  # Determine number of centers based on aspect ratio
        step = w / (2 * n_centers)  # Calculate step size
        return [(int(x + step * (2 * i + 1)), int(y + h / 2)) for i in range(n_centers)]

# Logger class is responsible for maintaining and updating the grid state
class Logger:
    def __init__(self, grid_cols=4, grid_rows=2):
        self.grid_cols = grid_cols  # Number of columns in the grid
        self.grid_rows = grid_rows  # Number of rows in the grid

    # Generate the state of the grid based on object positions
    def get_grid_state(self, red_positions, blue_positions, red_center_points, blue_center_points):
        grid_state = [["Empty" for _ in range(self.grid_cols)] for _ in range(self.grid_rows)]
        red_box_positions = []

        # Return empty grid if no objects are detected
        if not red_positions and not blue_positions:
            print("No objects detected. Returning empty grid.")
            return grid_state

        # Calculate grid boundaries
        min_x = min((x for x, y, w, h in red_positions + blue_positions), default=0)
        min_y = min((y for x, y, w, h in red_positions + blue_positions), default=0)
        max_w = max((x + w for x, y, w, h in red_positions + blue_positions), default=0)
        max_h = max((y + h for x, y, w, h in red_positions + blue_positions), default=0)

        if max_w == min_x or max_h == min_y:  # Handle potential division by zero
            print("Warning: Division by zero detected, returning empty grid.")
            return grid_state

        # Determine cell size for the 2x4 grid
        cell_width = (max_w - min_x) / self.grid_cols
        cell_height = (max_h - min_y) / self.grid_rows

        # Populate the grid with detected object colors
        for centerX, centerY in red_center_points + blue_center_points:
            col = min(self.grid_cols - 1, max(0, int((centerX - min_x) / cell_width)))
            row = min(self.grid_rows - 1, max(0, int((centerY - min_y) / cell_height)))

            # Determine object color and update grid state
            if (centerX, centerY) in red_center_points:
                red_box_positions.append([col, row])
                color = "Red"
            elif (centerX, centerY) in blue_center_points:
                color = "Blue"
            else:
                print(f"Error: Unexpected center point {centerX, centerY}.")
                continue

            grid_state[row][col] = color

        return red_box_positions

    # Calculate grid index based on x, y coordinates
    def get_grid_index_state(self, x, y, cell_width, cell_height):
        col = min(self.grid_cols - 1, max(0, int(x // cell_width)))
        row = min(self.grid_rows - 1, max(0, int(y // cell_height)))
        return row * self.grid_cols + col

# Main function to run the object detection and logging
def main():
    try:
        camera_handler = CameraHandler()  # Initialize camera handler
        object_detector = ObjectDetector()  # Initialize object detector
        logger = Logger()  # Initialize logger

        while True:
            frame = camera_handler.get_frame()  # Get a frame from the camera
            key = cv2.waitKey(1) & 0xFF

            if key == ord('a'):  # Start object detection when 'a' is pressed
                start = time.time()
                object_detector.process_frame(frame)  # Process the frame for object detection
                grid_state = logger.get_grid_state(
                    object_detector.red_positions,
                    object_detector.blue_positions,
                    object_detector.red_center_points,
                    object_detector.blue_center_points
                )
                print("Grid State:", grid_state)
                print("Detection Time: {:.3f} seconds".format(time.time() - start))

                cv2.imshow('Processed Frame', frame)
            else:
                cv2.imshow('Camera', frame)  # Show raw camera feed

            if key == ord('q'):  # Quit the program when 'q' is pressed
                break
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()  # Close all OpenCV windows
        camera_handler.release()  # Release camera resources

if __name__ == "__main__":
    main()
