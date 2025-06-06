import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from pathlib import Path


class TinyImageMLP:
    """
    End-to-end helper:
      • Loads & preprocesses one RGB image
      • Flattens it
      • Feeds it through a simple MLP
      • Returns softmax probabilities
    """

    def __init__(self, img_size: int = 224, num_classes: int = 10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Preprocessing pipeline
        mean = (0.485, 0.456, 0.406)  # ImageNet stats (good default)
        std = (0.229, 0.224, 0.225)
        self.transform = T.Compose(
            [
                T.Resize(img_size),  # shortest side → img_size
                T.CenterCrop(img_size),  # (img_size, img_size)
                T.ToTensor(),  # scales to [0, 1], shape (C,H,W)
                T.Normalize(mean, std),  # keeps shape
            ]
        )

        # 2. A small naive model
        flat_dim = 3 * img_size * img_size  # C × H × W after crop
        self.model = nn.Sequential(
            nn.Flatten(start_dim=1),  # (N, flat_dim)
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        ).to(self.device)

        # 3. Randomly initializing weights with something nicer
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.softmax = nn.Softmax(dim=1)

        # 4. Setting eval mode by default
        self.model.eval()

    # public helpers
    @torch.no_grad()
    def predict(self, img_path: str | Path) -> torch.Tensor:
        """
        Args
        ----
        img_path : str or Path
            Path to an image (any format supported by PIL).

        Returns
        -------
        probs : torch.Tensor, shape (1, num_classes)
            Softmax probabilities over the classes.
        """
        # 1) Load ≫ 2) preprocess ≫ 3) add batch dim
        pil_img = Image.open(img_path).convert("RGB")
        x = self.transform(pil_img).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        # 4) forward pass (already flattened by nn.Flatten)
        logits = self.model(x)  # (1, num_classes)

        # 5) turn into probabilities for human inspection
        probs = self.softmax(logits)  # (1, num_classes)

        return probs.cpu()

    # training stub
    def train_step(self, batch_imgs, batch_labels, optimizer, criterion):
        """
        A barebones single training step to show how you'd fit the network.

        batch_imgs  : torch.Tensor (N, 3, H, W)  – pre-transformed
        batch_labels: torch.Tensor (N,)          – class indices
        """
        self.model.train()
        optimizer.zero_grad()
        logits = self.model(batch_imgs.to(self.device))
        loss = criterion(logits, batch_labels.to(self.device))
        loss.backward()
        optimizer.step()
        self.model.eval()
        return loss.item()


if __name__ == "__main__":
    classifier = TinyImageMLP(img_size=224, num_classes=5)  # say 5 classes
    probs = classifier.predict("assets/kumiko-shimizu-unsplash.jpg")
    print("Class probabilities:", probs.numpy())
    print("Predicted class idx:", probs.argmax(dim=1).item())

    # NOT YET TRAINED
