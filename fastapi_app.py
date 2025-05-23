from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
from train import LogisticRegression  # the model is defined in train.py

app = FastAPI()

# FIXME: Load your trained model here
print("Loading model...")
model = LogisticRegression(input_dim=28*28, output_dim=10)  # Replace with actual model loading
model.load_state_dict(torch.load("model_epoch_19.pth"))  # Load the model weights
model.eval()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert('L')  # Convert to grayscale
    # FIXME: Implement preprocessing of the image
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    image_tensor = transform(image)
    
    # FIXME: Implement model inference
    with torch.no_grad():
        output = model(image_tensor)#.unsqueeze(0))
        prediction = output.argmax(dim=1, keepdim=True).item()
    
    # FIXME: Return the prediction in the appropriate format
    return JSONResponse(content={"prediction": prediction})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)