"""Image preprocessing / transforms."""
from torchvision import transforms

def get_preprocess_pipeline():
    """Return torchvision transform pipeline used for both train & infer."""
    return transforms.Compose([
        transforms.ToTensor(),  # converts [0,255] uint8 -> [0,1] float32
    ])
