from medmnist import PneumoniaMNIST
from torchvision import transforms
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = PneumoniaMNIST(
    split="train",
    transform=transform,
    download=True,
    size=224
)

val_dataset = PneumoniaMNIST(
    split="val",
    transform=transform,
    download=True,
    size=224
)

test_dataset = PneumoniaMNIST(
    split="test",
    transform=transform,
    download=True,
    size=224
)

print("Train:", len(train_dataset))
print("Val:", len(val_dataset))
print("Test:", len(test_dataset))

img, label = train_dataset[0]

print("Image shape:", img.shape)
print("Label:", label)

plt.imshow(img.squeeze(), cmap="gray")
plt.axis("off")
plt.show()
