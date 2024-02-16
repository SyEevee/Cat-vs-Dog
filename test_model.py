# loading the model and testing it on the test data

import torch
import torchvision.transforms as transforms
from PIL import Image

# Setting the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loading the saved model
model = torch.load('cats_vs_dogs.pth')

def predict(img_path):
# Loading the test image
    test_image = img_path

    # Transforming the image
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
    ])

    # Loading the image
    test_image = test_transform(Image.open(test_image).convert('RGB')).unsqueeze(0)

    # Passing the image to the model
    model.eval()
    with torch.no_grad():
        predictions = model(test_image.to(device)).argmax(dim=1)
        if predictions.item() == 1:
            print('dog')
        else:
            print('cat')

predict('dataset/single_prediction/cat_or_dog_1.jpg')