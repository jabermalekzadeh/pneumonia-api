from fastapi import FastAPI, File, UploadFile
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

# Define FastAPI app
app = FastAPI()

# Load model
device = torch.device("cpu")
class PneumoniaCNN(torch.nn.Module):
    def __init__(self):
        super(PneumoniaCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2)
        )
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(128, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc_layers(self.conv_layers(x))

model = PneumoniaCNN()
model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = "Pneumonia" if output.item() > 0.5 else "Normal"

    return {"prediction": prediction}
