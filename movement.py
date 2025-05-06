import colorRecognition
import numpy as np
import urx
import time

class RobotControl:
    def __init__(self):


        self.rob = urx.Robot("192.168.0.100")
        self.rob.set_tcp((0, 0, 0.142, 0, 0, 0))  # add tool schunk
        self.rob.set_payload(2, (0, 0, 0.1))
        time.sleep(0.2)  # leave some time to robot to process the setup commands

        # Define base coordinates for the grid
        self.x_base = 0.305704  # Base x-coordinate
        self.y_base = -0.288771 # Base y-coordinate
        self.z_base = 0.28589  # Base z-coordinate (table height)

        # Define cell dimensions in robot coordinates
        self.cell_width = 0.04695  # Cell width in meters
        self.cell_height = 0.069813  # Cell height in meters

        # Define approach distance (how far above object to position before going down)
        self.approach_distance = 0.010  # 5cm above object

        # Define home position
        self.home_position = (340.479962, -360.100958, 93.090019, -127.279181, 127.279191, -0.122425)


    def move_to_home(self):
        print("home position")
        self.rob.movel(self.home_position, 0.3, 0.2)

    def close(self):
        self.rob.close()

    def move(self, a=0.3, v=0.2):
        # Get red box positions from color recognition module
        print("Getting red box positions from camera...")
        red_positions = colorRecognition.get_grid_state()

        print(f"Detected {len(red_positions)} red positions: {red_positions}")

        self.move_to_home()
        i = 0
        for col, row in red_positions:
            # Calculate pickup position
            pickup_x = self.x_base + (col * self.cell_width)
            pickup_y = self.y_base + (row * self.cell_height)
            pickup_z = self.z_base

            # Calculate approach positions (slightly above objects)
            pickup_approach = (pickup_x, pickup_y, pickup_z + self.approach_distance, 0, 3.14, 0)

            # Destination position (here we're placing all objects at column 0)
            drop_x = self.x_base
            drop_y = self.y_base + (i * self.cell_height)
            drop_z = self.z_base
            drop_approach = (drop_x, drop_y, drop_z + self.approach_distance, 0, 3.14, 0)
            print(f"Moving from position [{col}, {row}] to position [i]")
            i += 1


            # Move to approach position above pickup
            self.rob.movel(pickup_approach, a, v)

            # Move down to pickup object
            pickup_position = (pickup_x, pickup_y, pickup_z, 0, 3.14, 0)
            self.rob.movel(pickup_position, a / 2, v / 2)

            # Activate gripper
            time.sleep(0.5)  # Replace

            self.rob.movel(pickup_approach, a / 2, v / 2)

            # Move to approach position
            self.rob.movel(drop_approach, a, v)

            # place object
            drop_position = (drop_x, drop_y, drop_z, 0, 3.14, 0)
            self.rob.movel(drop_position, a / 2, v / 2)  # Slower for precision

            # open gripper
            time.sleep(0.5)  # Replace with actual gripper control

            # back up
            print("Moving back up...")
            self.rob.movel(drop_approach, a / 2, v / 2)

            # Return to home position between operations
            self.move_to_home()

    def wait_for_program(self):
        while True:
            time.sleep(0.1)  # sleep first since the robot may not have processed the command yet
            if self.rob.is_program_running():
                break


# Example usage
if __name__ == "__main__":
    try:
        print("Initializing robot control...")
        robot = RobotControl()

        print("Moving to home position...")
        robot.move_to_home()

        print("Starting pick and place operations...")
        robot.move()

        print("Operations completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            print("Closing robot")
            robot.close()
        except:
            pass
        print("Program terminated")