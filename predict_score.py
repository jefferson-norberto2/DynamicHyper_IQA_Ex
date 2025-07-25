from IQANet import IQANet_DDF_Hyper, TargetNet
import torch
from PIL import Image
from torchvision import transforms
import argparse

parser = argparse.ArgumentParser(description='BLIND NATURAL IMAGE QUALITY PREDICTION USING '
                                'CONVOLUTIONAL NEURAL NETWORKS AND WEIGHTED SPATIAL POOLING')

parser.add_argument('--image_path', '-i', type=str, default='img/01.png', help='Path to the input image')
parser.add_argument('--model_path', '-m', type=str, default='checkpoints/training_48_epochs.pth.tar', help='Path to the model checkpoint')

def load_image(image_path: str):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Add batch dimension
    image = transform(image).unsqueeze(0)  

    return image.cuda()

def predict_score(image_path: str, model_path: str):
    # Load the model state
    state_dict = torch.load(model_path, map_location='cpu')['state_dict']
    
    # Create IQANet_DDF_Hyper model
    model = IQANet_DDF_Hyper(128, 24, 192, 64).cuda()
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Load and preprocess the image
    image = load_image(image_path)

    with torch.no_grad():
        # Generate dynamic parameters
        paras = model(image)

        # Create TargetNet using the parameters
        model_target = TargetNet(paras).cuda()
        
        # Remove gradients for TargetNet parameters
        for param in model_target.parameters():
            param.requires_grad = False

        # Pass the input vector through TargetNet
        output = model_target(paras['target_in_vec'])

        # Return the predicted score
        return output.squeeze().cpu().item()

if __name__ == "__main__":
    args = parser.parse_args()

    checkpoint_path = args.model_path
    image_path = args.image_path

    print(f"Predicting score for image: {image_path}")
    score = predict_score(image_path, checkpoint_path)
    print(f"Predicted score: {score:.2f}")
    