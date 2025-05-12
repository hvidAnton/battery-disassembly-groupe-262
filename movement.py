import colorRecognition
import socket
import urx
import time

class RobotControl:
    def __init__(self, gripper = None):
        try:
            print("Attempting to connect to robot...")
            self.rob = urx.Robot("192.168.0.100")
            print("Connected!")
            self.gripper = gripper or GripperControl()

            # Minimal setup commands
            self.rob.set_tcp((0, 0, 0.142, 0, 0, 0))
            self.rob.set_payload(2, (0, 0, 0.1))
            time.sleep(0.5)  # Longer wait time after initial setup
            # Define base coordinates for the grid
            self.x_base = 0.30697110556293716  # Base x-coordinate
            self.y_base = -0.42583099865300345  # Base y-coordinate
            self.z_base = 0.025299262123034774  # Base z-coordinate (table height)

            self.place_x_base = 0.3794308774795742
            self.place_y_base = -0.18
            self.place_z_base = 0.0222

            # Define cell dimensions
            self.cell_width = 0.0695  # Cell width in meters
            self.cell_height = 0.0475  # Cell height in meters

            # Define approach distance
            self.approach_distance = 0.05  # 5cm above object

            # Define timing for simulated pickup/place
            self.action_pause = 0.1  # Pause at pickup/place points

            # Speed settings - using very conservative values
            self.default_acc = 3
            self.default_vel = 2

            self.home_position = [0.3321, -0.2866, 0.2435, -2.2962, -2.1409, -0.0055]

        except Exception as e:
            print(f"Error during initialization: {e}")
            raise

    def move_safely(self, position, acc=None, vel=None, msg="Moving", tolerance=0.005, move = "movel"):
        acc = acc if acc is not None else self.default_acc
        vel = vel if vel is not None else self.default_vel

        print(f"{msg} to {position}")
        self.rob.movel(position, acc=acc, vel=vel, wait=False)

        timeout = 15  # seconds
        poll_interval = 0.002
        stable_count_required = 5

        start_time = time.time()
        stable_count = 0
        while time.time() - start_time < timeout:
            current_pose = self.rob.getl()
            diff = [abs(current_pose[i] - position[i]) for i in range(6)]
            within_tol = all(d <= tolerance for d in diff)
            if within_tol:
                stable_count += 1
                #print(f"Pose within tolerance: {stable_count}/{stable_count_required}")
                if stable_count >= stable_count_required:
                    print("Movement complete and stable.")
                    return True
            else:
                if stable_count > 0:
                    print("Drift detected, resetting stability counter.")
                stable_count = 0

            # Optional: Check if robot velocity is near zero (requires RTDE or get actual TCP speed)
            time.sleep(poll_interval)

        print("Timeout: Robot did not reach target pose in time.")
        return False

    def close(self):

        """Safely close the robot connection"""
        print("Closing robot connection...")
        try:
            self.rob.close()
            print("Robot connection closed successfully")
        except Exception as e:
            print(f"Error closing robot connection: {e}")

    def get_current_pos(self):
        """Get current robot position with error handling"""
        try:
            pos = self.rob.getl()
            print(f"Current position: {pos}")
            return pos
        except Exception as e:
            print(f"Error getting position: {e}")
            return None

    def run_remove_lid(self):
        print("\n--- Starting lid removal operation ---")

        lid_pick_position = [0.34549378894943444, -0.3595, 0.040825714378415756, 2.2975385808058735,
                             2.1422846609552124, 0.0]
        lid_place_position = [0.12724451121394884, -0.3595, 0.013353012293422756, 2.2976229488468403,
                              2.142284028861523, 0.0]

        # Move above pickup point
        approach_above = lid_pick_position.copy()
        approach_above[2] += self.approach_distance

        self.move_safely(approach_above, msg="Approaching lid pickup")
        self.move_safely(lid_pick_position, acc=self.default_acc, vel=self.default_vel, msg="Picking up lid")

        self.gripper.close()

        self.move_safely(approach_above, msg="Retracting lid")

        # Move to place position
        approach_place = lid_place_position.copy()
        approach_place[2] += self.approach_distance

        self.move_safely(approach_place, msg="Approaching lid place")
        self.move_safely(lid_place_position, acc=self.default_acc, vel=self.default_vel, msg="Placing lid")

        self.gripper.open()

        self.move_safely(approach_place, msg="Retracting after lid place")

        print("Lid removal complete")

    def run_single_pickup_place(self, pickup_col=0, pickup_row=0, place_col=0, place_row=0):
        """Run a single pickup and place operation with extensive error handling"""
        print(f"\n--- Starting pickup/place operation [{pickup_col},{pickup_row}] to [{place_col}] ---")
        try:
            # Calculate pickup coordinates
            pickup_x = self.x_base + (pickup_col * self.cell_width)
            pickup_y = self.y_base + (pickup_row * self.cell_height)
            pickup_z = self.z_base

            # Calculate place coordinates
            place_x = self.place_x_base - (place_row * self.cell_width)
            place_y = self.place_y_base + (place_col * self.cell_height)
            place_z = self.place_z_base


            # Calculate approach positions
            pickup_approach = (pickup_x, pickup_y, pickup_z + self.approach_distance, 0, 3.14, 0)
            place_approach = (place_x, place_y, place_z + self.approach_distance, 0, 3.14, 0)

            # Full positions
            pickup_position = (pickup_x, pickup_y, pickup_z, 0, 3.14, 0)
            place_position = (place_x, place_y, place_z, 0, 3.14, 0)

            # Execute the sequence with confirmation at each step
            steps = [
                ("move to pickup approach", pickup_approach),
                ("move to pickup position", pickup_position ),
                ("move back to pickup approach", pickup_approach),
                ("move to place approach", place_approach),
                ("move to place position", place_position),
                ("move back to place approach", place_approach)
            ]

            for step_name, position in steps:
                print(f"\nStep: {step_name}")
                # Use lower speed for actual pickup/place movements
                if "to pickup position" in step_name or "to place position" in step_name:
                    acc, vel = self.default_acc / 2, self.default_vel / 2  # pickup or place set the speed in half
                else:
                    acc, vel = self.default_acc, self.default_vel  # set the speed to normal

                success = self.move_safely(position, acc, vel, msg=f"Executing {step_name}")  # try top move
                if not success:
                    print(f"Failed at step: {step_name}")
                    return False

                if step_name == "move to pickup position" or step_name == "move to place position":
                    if 'pickup' in step_name:
                        print("pickup")
                        self.gripper.close()
                    elif 'place' in step_name:
                        print("place")
                        self.gripper.open()
                    else:
                        pass
                    time.sleep(self.action_pause)

            print("\nOperation completed successfully!")
            return True

        except Exception as e:
            print(f"\nError during operation: {e}")
            return False

class GripperControl:
    _instance = None  # class-level variable for singleton behavior
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GripperControl, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return  # avoid re-initializing
        self._initialized = True

        ROBOT_IP = "192.168.0.100"
        PORT = 30002  # URScript interface port

        print("Connecting to gripper control server")
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ROBOT_IP, PORT))
        time.sleep(1)

        self.s.send(b'set_digital_out(5, True)\n')
        time.sleep(0.05)
        self.s.send(b'set_digital_out(6, True)\n')
        time.sleep(0.25)

    def open(self):
        self.s.send(b'set_digital_out(7, False)\n')
        time.sleep(0.05)
        self.s.send(b'set_digital_out(6, True)\n')
        time.sleep(0.25)

    def close(self):
        self.s.send(b'set_digital_out(6, False)\n')
        time.sleep(0.05)
        self.s.send(b'set_digital_out(7, True)\n')
        time.sleep(0.25)

    def shutdown(self):
        self.s.close()
        print("Gripper socket closed.")

def run_color_pickup_place_sequence(robot):
    """Runs the color pickup/place sequence"""
    try:
        # Remove the lid first
        robot.run_remove_lid()

        # Initialize the camera handler and object detector
        camera_handler = colorRecognition.CameraHandler()
        object_detector = colorRecognition.ObjectDetector()
        logger = colorRecognition.Logger()
        print("Capturing frame and detecting objects...")
        frame = camera_handler.get_frame()
        object_detector.process_frame(frame)
        # Get the red box positions from the grid state
        red_positions = logger.get_grid_state(
            object_detector.red_positions,
            object_detector.blue_positions,
            object_detector.red_center_points,
            object_detector.blue_center_points
        )
        time.sleep(1)
        print(f"Detected red positions: {red_positions[0]}")
        if not red_positions:
            print("No red objects detected. Aborting operation.")
        else:
            # Use detected positions for pickup
            place_positions = list(range(len(red_positions)))
            print(f"Running pickup/place sequences based on detected objects:")
            for i, (pickup_col, pickup_row) in enumerate(red_positions):
                place_row = i // 2
                place_col = i % 2
                success = robot.run_single_pickup_place(pickup_row, pickup_col, place_row, place_col)
                if not success:
                    print("Aborting sequence due to failure.")
                    break
        # Clean up camera resources
        camera_handler.release()

    except Exception as e:
        print(f"Error in color detection sequence: {e}")
    finally:
        # Ensure camera resources are released
        try:
            camera_handler.release()
        except:
            pass


def main():
    robot = None
    gripper = None
    try:
        print("=== UR Robot Color Detection Control Program ===")
        robot = RobotControl()
        gripper = GripperControl()

        gripper.open()
        robot.move_safely(robot.home_position, 0.2, 0.2, "home position")

        while True:
            input("\nPress Enter to run the color pickup/place sequence...")
            run_color_pickup_place_sequence(robot)
            robot.move_safely(robot.home_position, 0.2, 0.2, "home position")

    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
    finally:
        if gripper:
            gripper.shutdown()
        if robot:
            robot.close()


if __name__ == "__main__":
    main()
