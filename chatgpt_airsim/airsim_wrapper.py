import airsim
import csv
import os
import cv2
import time
import math
import logging
import numpy as np
import torch
from PIL import ImageEnhance
from datetime import datetime
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models.segmentation as models
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from utils import log_metrics_to_csv, metrics_csv_path


import torch
import torch.nn as nn
import torch.nn.functional as F
# _________dictionaries________________________________________________________________________________________
objects_dict = {
    'trees1':'tree_group_152',
    'trees2':'tree_group_11',
    'trees3':'tree_group_15',
    'trees4':'tree_group_33',
    'trees5':'tree_group_5',
    'agave':'agave_5',
    'water':'WaterBodyCustom_1',
    'fire':'Burn_BP',
    'tower1':'SM_Tower_9',
    'tower2':'SM_Tower2_12',
}
# _________________________________________________________________________________________________
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the GrassfireCNN architecture
class FireCNN(nn.Module):
    def __init__(self):
        super(FireCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) #1st conv layer : 3 input channels (RGB), 16 ouput channels, 3x3 kernel - Output: 640 x 640 x 16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) #max pooling layer to reduce spatial dimensions - Output: 320 x 320 x 16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 2nd conv layer: 16 input channels, 32 output channels - Output: 320 x 320 x 32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 3rd conv layer: 32 input channels, 64 output channels - Output: 320 x 320 x 64
        # fully connected layers
        self.fc1 = nn.Linear(64 * 80 * 80, 128)  # Adjust input size if image size changes
        self.fc2 = nn.Linear(128, 1)  # Binary classification
        self.dropout = nn.Dropout(0.5) # 50% dropout

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # apply 1st conv layer followed by ReLU activation and pooling - Output: 320 x 320 x 16
        x = self.pool(F.relu(self.conv2(x))) # apply 2nd conv layer followed by ReLU activation and pooling - Output: 160 x 160 x 32
        x = self.pool(F.relu(self.conv3(x))) # apply 3rd conv layer followed by ReLU activation and pooling - Output: 80 x 80 x 64
        x = x.view(-1, 64 * 80 * 80) # flatten to shape (batch_size, 64 * 80 * 80)
        x = self.dropout(F.relu(self.fc1(x)))  # applying dropout
        
        #x = F.relu(self.fc1(x)) # apply 1st fully connected layer with ReLU activation
        x = torch.sigmoid(self.fc2(x)) # apply output layer with sigmoid activation for binary classification
        return x

# Load the CNN model and weights
def load_cnn_model(model_path=r'fire_cnn_.pth'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"File not found at specified path: {model_path}")
    
    model = FireCNN()
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)  # Explicitly set weights_only=True
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# _______original functions (altered for swarm)__________________________________________________________________________________________
class AirSimWrapper:
    def __init__(self, drone_id, other_drones=None):
        self.drone_id = drone_id
        self.other_drones = other_drones if other_drones else []
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.drone_id)
        self.client.armDisarm(True, vehicle_name=self.drone_id)
        self.objects_dict = objects_dict
        self.flight_start_time = time.time()  # Start tracking flight time
        self.battery_level = 100  # Simulated battery
        self.battery_drain_rate = 1  # Rate at which battery drains
        


        # Load CNN model and set to evaluation mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cnn_model = load_cnn_model().to(self.device)
        self.cnn_model.eval()  # Ensure model is in eval mode


        # Define the transform for scene images
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # SIMULTANEOUS TAKEOFF
    def takeoff(self, altitude=0):
        """Initiates asynchronous takeoff for simultaneous takeoff."""
        self.client.takeoffAsync(vehicle_name=self.drone_id)
        
    # SIMULTANEOUS LANDING
    def land(self):
        """Initiates asynchronous landing for simultaneous landings."""
        self.client.landAsync(vehicle_name=self.drone_id)       

    def detect_nearby_drones(self, radius):
        """
        Check for nearby drones within a given radius.
        """
        current_position = self.get_drone_position()
        
        for other_drone_id in self.other_drones:  # Assume self.other_drones is a list of other drone instances
            other_position = other_drone_id.get_drone_position()
            distance = self.calculate_deviation(current_position, other_position)
            
            if distance < radius:
                return True  # Nearby drone detected
        return False

    def fly_to(self, point, speed=5):
        """
        Commands this drone to fly to a target point.
        Ensures Z-axis coordinates are adjusted for AirSim.
        """
        target_x, target_y, target_z = point

        # Flip Z-coordinate if it's positive to match AirSim's coordinate system
        if target_z > 0:
            target_z = -target_z

        print(f"{self.drone_id} flying to ({target_x}, {target_y}, {target_z}) at speed {speed} m/s.")
        self.client.moveToPositionAsync(target_x, target_y, target_z, speed, vehicle_name=self.drone_id).join()


    def get_drone_position(self, vehicle_name=None):
        if vehicle_name is None:
            vehicle_name = self.drone_id
        pose = self.client.simGetVehiclePose(vehicle_name=vehicle_name)
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def fly_path(self, points, vehicle_name=None):
        if vehicle_name is None:
            vehicle_name = self.drone_id
        airsim_points = [airsim.Vector3r(point[0], point[1], point[2]) for point in points]
        self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1, vehicle_name=vehicle_name)

    def set_yaw(self, yaw, vehicle_name=None):
        if vehicle_name is None:
            vehicle_name = self.drone_id
        self.client.rotateByYawRateAsync(yaw, 5, vehicle_name=vehicle_name).join()

    def get_yaw(self, vehicle_name=None):
        if vehicle_name is None:
            vehicle_name = self.drone_id
        orientation_quat = self.client.simGetVehiclePose(vehicle_name=vehicle_name).orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return yaw

    def get_position(self, object_name):
        """ Get position of an object (e.g., fire) in the environment """
        print(f"Getting position for {object_name}...")  
        object_pose = self.client.simGetObjectPose(self.objects_dict[object_name]).position
        position = [object_pose.x_val, object_pose.y_val, object_pose.z_val]
        print(f"Position for {object_name}: {position}")  
        return position

# _________added functions________________________________________________________________________________________
# _________metric logging________________________________________________________________________________________
   
    # def log_metric_to_csv(self, metric_name, value, interaction=None, comments=None):
    #     """
    #     Log any metric to a shared CSV file for all drones with extended columns.
    #     - metric_name: Name of the metric (e.g., "Chat Response Time")
    #     - value: The numerical value of the metric
    #     - interaction: Details about the command or interaction (optional)
    #     - comments: Additional notes or comments (optional)
    #     """
    #     file_path = "swarm_metrics.csv"
    #     file_exists = os.path.isfile(file_path)
        
    #     with open(file_path, mode='a', newline='') as file:
    #         writer = csv.writer(file)
            
    #         # Write header only if the file is new
    #         if not file_exists:
    #             writer.writerow(["Timestamp", "Drone ID", "Metric Name", "Value", "Interaction", "Comments"])
            
    #         # Log entry with the specified fields and placeholders for unused columns
    #         writer.writerow([
    #             datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    #             self.drone_id,
    #             metric_name,
    #             value,
    #             interaction if interaction else "",
    #             comments if comments else ""
    #         ])

# _________formation functions________________________________________________________________________________________
 
    def form_v_formation(client, num_drones, spacing=5.0, altitude=10.0, angle=30):
        """
        Commands a swarm of drones to form a V-formation.

        Parameters:
            client (list): List of AirSim client objects for each drone.
            num_drones (int): Total number of drones in the swarm.
            spacing (float): Distance between drones in the arms of the V (in meters).
            altitude (float): Altitude at which the V should form (in meters).
            angle (float): Angle of the V-formation (in degrees).

        """
        # Calculate the angle in radians
        angle_rad = np.radians(angle)
        
        # Calculate the initial positions for the V formation
        positions = []
        center_drone_index = num_drones // 2  # The central point of the V
        
        for i in range(num_drones):
            # Determine if the drone is on the left or right side of the V
            offset = abs(i - center_drone_index)
            
            # x position moves out from the center based on the offset
            x = offset * spacing * np.cos(angle_rad)
            
            # y position shifts left or right based on which side of the V the drone is on
            y = (-1 if i < center_drone_index else 1) * offset * spacing * np.sin(angle_rad)
            
            # z position is the same for all drones (altitude)
            z = -altitude  # Negative for upward direction in AirSim
            
            positions.append((x, y, z))
        
        # Move each drone to the calculated position
        for i, pos in enumerate(positions):
            client[i].moveToPositionAsync(pos[0], pos[1], pos[2], velocity=5).join()
        
        # Allow some time for drones to stabilize
        time.sleep(2)

        print(f"Drones have formed a V-formation at {altitude} meters altitude.")

    def calculate_deviation(self, target_position, actual_position):
        """Calculate Euclidean deviation between target and actual positions."""
        return math.sqrt(
            (actual_position[0] - target_position[0]) ** 2 +
            (actual_position[1] - target_position[1]) ** 2 +
            (actual_position[2] - target_position[2]) ** 2
    )

    def grid_formation(client, num_drones, grid_spacing=5.0, altitude=10.0):
        """
        Commands a swarm of drones to form a grid formation.

        Parameters:
            client (list): List of AirSim client objects for each drone.
            num_drones (int): Total number of drones in the swarm.
            grid_spacing (float): Distance between drones in the grid (in meters).
            altitude (float): Altitude at which the grid should form (in meters).

        """
        # Calculate grid dimensions based on the number of drones
        grid_size = int(np.ceil(np.sqrt(num_drones)))
        
        # Initialize target positions
        positions = []
        
        # Calculate the positions in a grid
        for i in range(num_drones):
            row = i // grid_size
            col = i % grid_size
            x = col * grid_spacing
            y = row * grid_spacing
            z = -altitude  # Negative for upward direction in AirSim
            positions.append((x, y, z))
        
        # Move each drone to the calculated position
        for i, pos in enumerate(positions):
            client[i].moveToPositionAsync(pos[0], pos[1], pos[2], velocity=5).join()
        
        # Allow some time for drones to stabilize
        time.sleep(2)

        print(f"Drones have formed a {grid_size}x{grid_size} grid at {altitude} meters altitude.")

# ________battery functions_________________________________________________________________________________________

    def decrease_battery(self):
        """
        Decreases the battery level based on battery drain rate.
        If battery falls below 20%, initiate return to base.
        """
        self.battery_level -= self.battery_drain_rate
        self.battery_level = max(0, self.battery_level)  # Prevent battery from going below 0
        
        # Notify user and return to base if battery is critically low
        if self.battery_level <= 20:
            print(f"Warning: {self.drone_id} battery is low ({self.battery_level}%). Returning to base.")
            self.return_to_base()

    def return_to_base(self):
        """Commands the drone to return to the base location (assumed to be origin)."""
        base_position = [0, 0, -10]  # Set the altitude for landing at base
        print(f"{self.drone_id} returning to base at {base_position}.")
        self.client.moveToPositionAsync(base_position[0], base_position[1], base_position[2], 5, vehicle_name=self.drone_id).join()
        self.land()

# __________fire detection functions______________________________________________________________________________________

    def capture_scene_image(self):
        """Captures a scene image and runs it through the CNN to detect fire."""
        try:
            time.sleep(3)  # Optional delay to stabilize the drone

            for attempt in range(3):
                start_time = time.time()  # Start timing for capture and detection

                # Capture the image from the drone's camera
                response = self.client.simGetImage("camera_name", airsim.ImageType.Scene, vehicle_name=self.drone_id)
                if response:
                    img_data = np.frombuffer(response, dtype=np.uint8)
                    decoded_image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

                    if decoded_image is not None:
                        print(f"{self.drone_id}: Successfully captured scene image on attempt {attempt + 1}.")

                        # Apply sharpening filter
                        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                        sharpened_image = cv2.filter2D(decoded_image, -1, sharpen_kernel)

                        # Convert BGR to RGB
                        rgb_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB)

                        # Resize to fit the model's input requirements (640x640)
                        resized_image = cv2.resize(rgb_image, (640, 640), interpolation=cv2.INTER_LINEAR)

                        # Convert to tensor
                        tensor_image = transforms.ToTensor()(resized_image).unsqueeze(0).to(self.device)

                        # Run the tensor through the CNN model for fire detection
                        with torch.no_grad():
                            output = self.cnn_model(tensor_image)

                        fire_detected = (output.item() > 0.7)  # Assuming 0.7 threshold for binary classification
                        end_time = time.time()  # End timing
                        detection_time = end_time - start_time

                        # Save the image to a file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        save_dir = f"./images/{self.drone_id}"
                        os.makedirs(save_dir, exist_ok=True)  # Create a directory for each drone if it doesn't exist
                        save_path = os.path.join(save_dir, f"{self.drone_id}_{timestamp}.png")
                        cv2.imwrite(save_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
                        print(f"Image saved for {self.drone_id} at {save_path}")

                        # Log detection attempt
                        log_metrics_to_csv(metrics_csv_path, "Fire Detection Attempt", fire_detected,
                                    drone_id=self.drone_id, event_description=f"Attempt {attempt + 1}")


                        return fire_detected, decoded_image  # Return both fire detection status and the image

                    else:
                        print(f"{self.drone_id}: Decoding failed; retrying...")
                else:
                    print(f"{self.drone_id}: Image capture response was empty; retrying...")

                time.sleep(1)  # Delay before retrying

            print(f"{self.drone_id}: Failed to capture image after 3 attempts.")
            return False, None  # Return no detection and no image on failure

        except Exception as e:
            print(f"{self.drone_id}: Error capturing scene image: {e}")

        return False, None  # Return no detection and no image on failure

    def search_for_fire(self, search_area, fire_detected_flag, altitude=5, capture_interval=5):
        print(f"{self.drone_id} starting fire search...")
        #self.takeoff()
        time.sleep(3)  # Wait for the drone to stabilize after takeoff

        for point in search_area:
            # Check if fire was already detected by another drone
            if fire_detected_flag[0]:  
                print(f"{self.drone_id} aborting search as fire was detected by another drone.")
                break

            # Fly to each point in the search area
            target_position = [point[0], point[1], -altitude]
            print(f"{self.drone_id} flying to {target_position}")
            self.fly_to(target_position)
            print(f"{self.drone_id} reached {target_position}")

            # Multi-image verification process
            fire_confirmed = self.multi_image_verification(target_position)

            # If fire is confirmed, update the flag and stop further search
            if fire_confirmed:
                print(f"Fire confirmed by {self.drone_id} at position {target_position}!")
                fire_detected_flag[0] = True  # Update flag to notify other drones
                break  # Exit search as fire is confirmed
            else:
                print(f"No fire detected by {self.drone_id} at {target_position}")

            # Wait before moving to the next waypoint if no fire was detected
            time.sleep(capture_interval)

        # Final message if this drone completes the search without detecting fire
        if not fire_detected_flag[0]:
            print(f"No fire detected in search area for {self.drone_id}.")


    def multi_image_verification(self, target_position):
        """Captures multiple images at slight position adjustments to verify fire detection."""
        detections = 0
        adjustments = [(0, 0), (5, 0)]  # Adjust positions slightly for verification , (5, 0), (-5, 0)
        
        for dx, dy in adjustments:
            # Adjust position and capture image
            adjusted_position = [target_position[0] + dx, target_position[1] + dy, target_position[2]]
            self.fly_to(adjusted_position)
            print(f"{self.drone_id} capturing image at {adjusted_position} for verification...")

            fire_detected, captured_image = self.capture_scene_image()
            
            if fire_detected:
                    detections += 1
                    print(f"Fire detected in verification image at {adjusted_position}.")
            
            # Require at least 3 positive detections to confirm fire
            if detections >= 3:
                return True
        
        return False  # Return false if not enough detections confirm fire


    def start_swarm_search(self, drones, search_area, fire_detected_flag, altitude=5, capture_interval=5):
        """
        Initiates a fire search with all drones in the swarm using multi-threading.
        Each drone searches the specified area independently until fire is detected by any drone.
        """
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(drone.search_for_fire, search_area, fire_detected_flag, altitude, capture_interval)
                for drone in drones.values()
            ]
            
            # Continuously check for completion or fire detection
            for future in as_completed(futures):
                if fire_detected_flag[0]:  # If fire is detected, terminate all threads
                    print("Fire detected, stopping all drone searches.")
                    break  # Exit once fire is detected
                future.result()  # Wait for each thread to finish (unless break is triggered)

        if fire_detected_flag[0]:
            print("Fire detected by one or more drones. Search terminated.")
        else:
            print("No fire detected by any drone in the swarm.")


    def check_and_recover_from_collision(self, reset_position=None):
        """Checks for collision and attempts to recover the drone if a collision is detected."""
        collision_info = self.client.simGetCollisionInfo(vehicle_name=self.drone_id)
        
        if collision_info.has_collided:
            print(f"{self.drone_id} has collided! Attempting recovery...")
            
            # Option 1: Move the drone slightly upward to clear the collision.
            current_position = self.get_drone_position()
            if reset_position:
                target_position = reset_position
                print(f"{self.drone_id}: Moving to reset position to recover.")
            else:
                target_position = [current_position[0], current_position[1], current_position[2] - 5]
                print(f"{self.drone_id}: Moving upward to recover from collision.")

            self.client.moveToPositionAsync(target_position[0], target_position[1], target_position[2], 5, vehicle_name=self.drone_id).join()
            time.sleep(1)  # Wait a moment to ensure it moves out of collision

            # Re-arm if disarmed after collision
            self.client.armDisarm(True, vehicle_name=self.drone_id)
            
            # Re-takeoff if needed
            self.takeoff()
            print(f"{self.drone_id}: Recovery completed. Resuming operations.")

        return not collision_info.has_collided

# __________(future A* pathfinding functions)_______________________________________________________________________________________

    def navigate_with_astar(self, start, goal, grid):
            # Use A* to get the path from start to goal
                path = astar(grid, start, goal)
                if path:
                    for point in path:
                        self.fly_to(point)
                else:
                    print(f"No valid path found for {self.drone_id} from {start} to {goal}")

class Node:
    def __init__(self, position, parent=None):
        self.position = position  # (x, y, z) position in 3D space
        self.parent = parent  # Parent node to trace the path
        self.g = 0  # Cost from start to this node
        self.h = 0  # Heuristic cost to the goal
        self.f = 0  # Total cost (g + h)

    def __eq__(self, other):
        return self.position == other.position  # Nodes are equal if they have the same position

def heuristic(node, goal):
    """Calculate the Euclidean distance between the current node and the goal."""
    return math.sqrt((node.position[0] - goal.position[0])**2 + (node.position[1] - goal.position[1])**2 + (node.position[2] - goal.position[2])**2)

def get_neighbors(node, grid):
    """Generate neighbors based on current node's position. Only consider valid moves within the grid."""
    directions = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]  # 6 directions (up, down, left, right, up in z, down in z)
    neighbors = []
    
    for direction in directions:
        new_position = (node.position[0] + direction[0], node.position[1] + direction[1], node.position[2] + direction[2])

        # Ensure the new position is within the grid bounds and is not an obstacle
        if is_valid(new_position, grid):
            neighbors.append(Node(new_position, parent=node))

    return neighbors

def is_valid(position, grid):
    """Check if the position is within the grid bounds and is not an obstacle."""
    x, y, z = position
    # Check bounds
    if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and 0 <= z < len(grid[0][0]):
        # Check if the cell is not an obstacle (assuming obstacles are marked with 1)
        if grid[x][y][z] == 0:
            return True
    return False

def astar(grid, start, goal):
    """A* pathfinding algorithm."""
    start_node = Node(start)
    goal_node = Node(goal)

    open_list = []
    closed_list = []

    open_list.append(start_node)

    while open_list:
        # Sort open_list to find the node with the lowest f cost
        open_list.sort(key=lambda node: node.f)
        current_node = open_list.pop(0)  # Remove node with lowest f cost

        # If goal is reached, reconstruct the path
        if current_node == goal_node:
            path = []
            while current_node is not None:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path (from start to goal)

        # Add the current node to closed_list (processed nodes)
        closed_list.append(current_node)

        # Generate neighbors for the current node
        neighbors = get_neighbors(current_node, grid)
        
        for neighbor in neighbors:
            if neighbor in closed_list:
                continue  # Skip neighbors already evaluated

            # Calculate g, h, and f costs for the neighbor
            neighbor.g = current_node.g + 1  # Assuming uniform cost of 1 for moving to a neighbor
            neighbor.h = heuristic(neighbor, goal_node)  # Heuristic based on Euclidean distance
            neighbor.f = neighbor.g + neighbor.h

            # If the neighbor is already in open_list with a lower f cost, skip it
            if any(open_node for open_node in open_list if neighbor == open_node and neighbor.g > open_node.g):
                continue

            # Add the neighbor to the open_list for future evaluation
            open_list.append(neighbor)

    return None  # No path found

