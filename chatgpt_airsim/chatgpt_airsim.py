import openai
import csv
import time
import re
import argparse
from airsim_wrapper import AirSimWrapper, load_cnn_model
import os
from concurrent.futures import ThreadPoolExecutor
import json
from torchvision import transforms
import numpy as np
from utils import log_metrics_to_csv, setup_csv_logger, metrics_csv_path


# ________ definitions _________________________________________________________________________________________________________________________________________________
num_drones = 5  # Number of drones in the swarm
metrics_csv_path = 'test.csv'  # Use this consistently throughout
successful_commands = 0  # To track successful commands
failed_commands = 0  # To track failed commands
fire_detected_flag = [False]
# _________________________________________________________________________________________________

# Setup command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default=r"SwarmPrompt\chatgpt_airsim\prompts\airsim_basic.txt")
parser.add_argument("--sysprompt", type=str, default=r"SwarmPrompt\chatgpt_airsim\system_prompts\airsim_basic.txt")
args = parser.parse_args()

with open(r"SwarmPrompt\chatgpt_airsim\config.json", "r") as f:
    config = json.load(f)

print("Initializing ChatGPT...")
openai.api_key = config["OPENAI_API_KEY"]

with open(args.sysprompt, "r") as f:
    sysprompt = f.read()

chat_history = [
    {"role": "system", "content": sysprompt},
    {"role": "user", "content": "Search for fire"},
    {"role": "assistant", "content": """```python
for drone in drones.values():
    drone.search_in_formation(drones, [0, 0, -10], search_radius=100, num_points=6)
```"""}
]


def ask(prompt):
    try:

        start_time = time.time()  # Start timing for ResponseTime metric

        chat_history.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=chat_history,
            temperature=0
        )
        
        response = completion.choices[0].message.content 
        chat_history.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        end_time = time.time()  # End timing
        response_time = end_time - start_time  # Calculate response time
        log_metrics_to_csv(metrics_csv_path, "Chat Response Time", response_time, user_command=prompt, chatgpt_response=response)
        return response
    except Exception as e:
        return "Sorry, there was an error processing your request."


print(f"Done.")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)

# ________ back functions _________________________________________________________________________________________________________________________________________________
def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)
        if full_code.startswith("python"):
            full_code = full_code[7:]
        return full_code
    return None

def execute_python_code(code, context):
    global successful_commands, failed_commands
    
    context["cnn_model"] = cnn_model
    context["transform"] = transform
    
    try:
        exec(code, {}, context)
        successful_commands += 1
        log_metrics_to_csv(metrics_csv_path, "Successful Commands", successful_commands)
        return True
    except Exception as e:
        print(f"Error executing command: {e}")
        failed_commands += 1
        log_metrics_to_csv(metrics_csv_path, "Failed Commands", failed_commands, reason=str(e))
        return False

def log_interaction(user_input, interpreted_command, correct):
    with open(metrics_csv_path, "a", newline='') as file:
        writer = csv.writer(file)
        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), user_input, interpreted_command, correct])
# ________ calculating metrics _________________________________________________________________________________________________________________________________________________

def track_response_time(start_time, end_time):
    response_time = end_time - start_time
    log_metrics_to_csv(metrics_csv_path, "Command Response Time", response_time)
    return response_time

def calculate_coverage(drones):
    # Simplified example; could calculate area by tracking visited coordinates
    total_area_covered = 0
    for drone in drones.values():
        # Example: calculate the area covered based on path or waypoints visited
        total_area_covered += len(drone.visited_positions) * 10  # Example calculation
    log_metrics_to_csv(metrics_csv_path, "Area Coverage", total_area_covered)


# ________ logging metrics _________________________________________________________________________________________________________________________________________________
def track_system_metrics(drone, metric_name, value):
    log_metrics_to_csv(metrics_csv_path, metric_name, value, drone_id=drone.drone_id)

# def setup_csv_logger():
#     if not os.path.exists(metrics_csv_path):  # Avoid overwriting if file already exists
#         with open(metrics_csv_path, mode='w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(['Timestamp', 'Metric Name', 'Value', 'User Command', 'ChatGPT Response', 'Feedback', 'Rating', 'Reason'])

# def log_metrics_to_csv(file_name, metric_name, value, user_command=None, chatgpt_response=None, start_time=None, end_time=None, rating=None, reason=None, drone_id=None, event_description=None):
#     with open(file_name, mode='a', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow([
#             time.strftime("%Y-%m-%d %H:%M:%S"), metric_name, value, user_command if user_command else '',
#             chatgpt_response if chatgpt_response else '', start_time if start_time else '',
#             end_time if end_time else '', rating if rating else '', reason if reason else '',
#             drone_id if drone_id else '', event_description if event_description else ''
#         ])

def check_battery_levels(drones):
    """
    Decrease battery for each drone and check if any battery level is below the critical threshold.
    If so, return that drone to base.
    """
    for drone in drones.values():  # Iterate over the values, not the keys
        # Decrease battery
        drone.battery_level -= drone.battery_drain_rate
        drone.battery_level = max(0, drone.battery_level)  # Ensure battery doesn't go below 0

        # Check if battery is critically low
        if drone.battery_level <= 20:
            print(f"Warning: {drone.drone_id} battery is low ({drone.battery_level}%). Returning to base.")
            drone.return_to_base()  # Command the drone to return to base
        else:
            print(f"{drone.drone_id} battery level: {drone.battery_level}%")


def collect_user_feedback():
    """Collect user feedback at the end of a simulation."""
    rating = input("Rate your satisfaction with the system (1-5): ")
    #reason = input("Reason for your rating: ")
    #improvement_suggestion = input("Any improvements you'd suggest? ")
    log_metrics_to_csv(metrics_csv_path, "User Satisfaction", rating)#, reason=reason, improvement=improvement_suggestion
setup_csv_logger()

# Load CNN model and set up the transformation pipeline
cnn_model = load_cnn_model()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# _________________________________________________________________________________________________________________________________________________________________________

class colors:
    RED = "\033[31m"
    ENDC = "\033[m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"

print("Initializing AirSim...")
drones = {
    "Drone1": AirSimWrapper("Drone1"),
    "Drone2": AirSimWrapper("Drone2"),
    "Drone3": AirSimWrapper("Drone3"),
    "Drone4": AirSimWrapper("Drone4"),
    "Drone5": AirSimWrapper("Drone5")
}
print("Done.")   
# _________________________________________________________________________________________________________________________________________________________________________

# Define start_swarm_search and add it to local_context
def start_swarm_search(drones, search_area, fire_detected_flag):
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(drone.search_for_fire, search_area, fire_detected_flag) for drone in drones.values()]
        for future in futures:
            future.result()

# Prepare the context with additional swarm-level command
local_context = {
    "drones": drones,
    "num_drones": num_drones,
    "cnn_model": cnn_model,
    "transform": transform,
    "fire_detected_flag": fire_detected_flag,
    "start_swarm_search": AirSimWrapper.start_swarm_search,
}
# _________________________________________________________________________________________________________________________________________________________________________



with open(args.prompt, "r") as f:
    prompt = f.read()

ask(prompt)
print("Welcome to the AirSim chatbot! I am ready to help you with your AirSim questions and commands.")
while True:
    question = input(colors.YELLOW + "AirSim> " + colors.ENDC)
        
    if question == "quit" or question == "exit":
        # collect_user_feedback()  # Collect feedback before exiting
        # print("Thank you for your feedback!")
        # print("Exiting...")
        break

    if question == "!clear":
        os.system("cls" if os.name == "nt" else "clear")
        continue

    response = ask(question)
    print(f"\n{response}\n")

    code = extract_python_code(response)

    if code is not None:
        print("Please wait while I run the code in AirSim...")
        local_context = {"drones": drones, "num_drones": num_drones}

        exec_start_time = time.time()

        if execute_python_code(code, local_context):
            print("Command executed successfully.")
        else:
            print("Command execution failed.")
            
        exec_end_time = time.time()
        execution_time = exec_end_time - exec_start_time
        log_metrics_to_csv(metrics_csv_path, "Command Execution Time", execution_time)
        print(f"Execution Time: {execution_time:.3f} seconds")

        #check_battery_levels(drones)

        

    
