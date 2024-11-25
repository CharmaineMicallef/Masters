import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class FireCNN(nn.Module):
    def __init__(self):
        super(FireCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 80 * 80, 128)
        self.fc2 = nn.Linear(128, 1)  # Binary classification

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 80 * 80)  # Flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Sigmoid for binary classification
        return x

# Load the model
model = FireCNN()
model.load_state_dict(torch.load(r'fire_cnn_.pth', map_location=torch.device('cpu')))
model.eval()

# Define the preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Function to predict fire or no-fire
def predict_fire(image_path, threshold=0.7):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Transform and add batch dimension

    # Run the image through the model
    with torch.no_grad():
        output = model(image)
    
    fire_prob = output.item()  # Since output is a single value after sigmoid
    
    if fire_prob > threshold:
        return f"Fire detected with probability {fire_prob:.2f}"
    else:
        return f"No fire detected with probability {1 - fire_prob:.2f}"

# Test with a scene image
# Uncomment the appropriate image path based on your test scenario
image_path = r'Scene\Scene_Rain1\auto\winter\400ft\down\pic_521.png' # No fire, day
# image_path = r'Scene\Scene_Rain1\auto\winter\400ft\down\pic_531.png'  # Fire, day
# image_path = r'Scene\Night_Scene_1_SingleFire\auto\winter\400ft\down\img_1.png'  # No fire, night 
# image_path = r'Scene\Night_Scene_1_\auto\winter\400ft\down\image_71.png'  # Fire, night

# Predict and print the


# Test with segmentation images

#image_path = r'Segmentation\Segmentation_Rain1\auto\winter\400ft\down\Segmentation_Rain1_214.png' #pic w/o fire day
#image_path = r'Segmentation\Segmentation_Rain1\auto\winter\400ft\down\Segmentation_Rain1_246.png'  # pic w. fire day - Not detected
#image_path = r'Segmentation\Night_Segmentation_1_SingleFire\auto\winter\400ft\down\seg_00639.png'  # pic w/o fire night 
#image_path = r'Segmentation\Night_Segmentation_1_SingleFire\auto\winter\400ft\down\seg_00608.png'  # pic w. fire night - Not detected

# Test with greyscale images

#image_path = r'IR\IR\auto\winter\400ft\down\ir_00219.png' #pic w/o fire day
#image_path = r'IR\IR_Rain1\auto\winter\400ft\down\ir_00063.png'  # pic w. fire day - Not detected
#image_path = r'IR\Night_IR_1_SingleFire\auto\winter\400ft\down\ir_00571.png'  # pic w/o fire night 
#image_path = r'IR\Night_IR_1_SingleFire\auto\winter\400ft\down\ir_00574.png'  # pic w. fire night - Not detected





print(predict_fire(image_path))
