# Import necessary libraries
import torch
from PIL import Image
import torchvision.transforms as transforms

def img_to_tensor(path_to_image):
    # Read a PIL image
    image = Image.open('../../test_img.jpg')
    
    image = image.resize((256, 256)) # resize to 256 x 256
    
    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
        transforms.PILToTensor()
    ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)
    img_tensor = torch.unsqueeze(img_tensor, dim=0) # add dimension to tensor
    print(img_tensor[0, :, 0, 0])
    img_tensor = img_tensor[:, [2, 1, 0], :, :]
    print(img_tensor[0, :, 0, 0])
    return img_tensor
