"""Dataset wrappers."""
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image
from .preprocessing import get_preprocess_pipeline

class CocoUnlabeledDataset(Dataset):
    """Wrap COCO *unlabeled2017* JPEGs as a PyTorch Dataset."""
    def __init__(self, root: Path):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.jpg"))
        self.transform = get_preprocess_pipeline()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = read_image(str(path)).float() / 255.0
        img = self.transform(img) if self.transform else img
        return img, path.name
