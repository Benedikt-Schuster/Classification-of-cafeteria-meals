import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torch.utils.data import Dataset
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())  # Sollte True ausgeben, wenn Nvidia Grafikkarte vorhanden

# Klasse Net definiert 
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        logits = self.fc2(x)
        return {"logits": logits}

# Modell laden
model_save_path = "trained_complete_model.pth"
net = torch.load(model_save_path)

# In den Evaluationsmodus setzen
net.eval()

print("Modell wurde erfolgreich geladen!")


# Dataset-Klasse definieren
class CanteenFoodDataset(Dataset):
    def __init__(self, root_dir):
        """
        Custom dataset for canteen food images. Automatically loads images
        and their corresponding label from a folder structure.
        """
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._load_images_from_folders()

    def _load_images_from_folders(self):
        for class_idx, class_name in enumerate(sorted(os.listdir(self.root_dir))):
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = class_idx
                self.idx_to_class[class_idx] = class_name
                for file_name in os.listdir(class_path):
                    file_path = os.path.join(class_path, file_name)
                    if os.path.isfile(file_path):
                        self.image_paths.append(file_path)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        return image, label

# Dataset initialisieren
images_folder = r"D:\Studium\Master DS\1. Semester\DS & AI\Bilder"  # Passe den Pfad zu deinem Bilderordner an
dataset = CanteenFoodDataset(images_folder)

# Testen, ob das Dataset korrekt geladen wurde
print(f"Anzahl der Bilder im Dataset: {len(dataset)}")

###########################################################
# Klassenbeschriftung hier für alle Klassen richtig außer 1 & 10
# Klasse 1 = erkannte Klasse 0; Klasse 10 = erkannte Klasse 1
###########################################################

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import torch.nn.functional as F  

# Zielgerät definieren
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Grad-CAM Wrapper-Modell erstellen
class GradCAMWrapper(torch.nn.Module):
    def __init__(self, original_model):
        super(GradCAMWrapper, self).__init__()
        self.original_model = original_model

    def forward(self, x):
        output = self.original_model(x)  
        return output['logits']  

# trainiertes Modell laden
net = torch.load("trained_complete_model.pth") 
net.eval()

# Wrapper-Modell verwenden
gradcam_model = GradCAMWrapper(net)

# Grad-CAM initialisieren
target_layer = gradcam_model.original_model.conv2  
cam = GradCAM(model=gradcam_model, target_layers=[target_layer])

# Testbild manuell festlegen
example_image_path = r"D:\Studium\Master DS\1. Semester\DS & AI\Bilder\class_6\IMG_5291.jpg"
print(f"Ausgewähltes Bild: {example_image_path}")

# Bildvorbereitung
example_image = Image.open(example_image_path).convert("RGB")

# Transformationen wie beim Training anwenden
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Eingabe-Batch vorbereiten
input_tensor = transform(example_image).unsqueeze(0).to(DEVICE)

# Grad-CAM ausführen
grayscale_cam = cam(input_tensor=input_tensor, targets=None)

# Vorhersage berechnen
with torch.no_grad():
    output = gradcam_model(input_tensor)  # Modellvorhersage (Logits)
    probabilities = F.softmax(output, dim=1)  # Wahrscheinlichkeiten berechnen
    predicted_class = torch.argmax(probabilities, dim=1).item()  # Klasse mit höchster Wahrscheinlichkeit
    predicted_probability = probabilities[0, predicted_class].item() * 100  # Prozentwert

# Klasse und Wahrscheinlichkeit anzeigen
print(f"Vorhergesagte Klasse: {predicted_class}")
print(f"Wahrscheinlichkeit: {predicted_probability:.2f}%")

# Wahrscheinlichkeiten und Klassen sortiert ausgeben
all_classes = list(range(probabilities.shape[1]))  # Klassen-IDs (0, 1, ..., N-1)
all_probabilities = probabilities.cpu().numpy().flatten()  # Wahrscheinlichkeiten in ein Array umwandeln
sorted_predictions = sorted(zip(all_classes, all_probabilities), key=lambda x: x[1], reverse=True)

print("\nVorhersagen aller Klassen:")
for cls, prob in sorted_predictions:
    print(f"Klasse {cls}: {prob * 100:.2f}%")

# Visualisierung vorbereiten
example_image_np = np.array(example_image.resize((224, 224))) / 255.0
visualization = show_cam_on_image(example_image_np, grayscale_cam[0], use_rgb=True)

# 1. Aktivierungswerte für die Farbskala vorbereiten
activation_map = grayscale_cam[0]  # Grad-CAM Heatmap (Grayscale)
norm = Normalize(vmin=0, vmax=1)  # Werte normalisieren (zwischen 0 und 1)
colormap = cm.jet  # Farbskala (Jet ist gängig für Heatmaps)

# 2. Plot mit Farbskala
fig, ax = plt.subplots(figsize=(8, 8))

# Zeige das Bild mit der Heatmap
im = ax.imshow(visualization)
ax.axis("off")
ax.set_title(f"Grad-CAM Visualisierung: Klasse {predicted_class}, {predicted_probability:.2f}%")

# 3. Farbskala hinzufügen
cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax)
cbar.set_label("Aktivierungsintensität (Grad-CAM)")

plt.show()

# Balkendiagramm der Wahrscheinlichkeiten
plt.figure(figsize=(10, 5))
bars = plt.bar(all_classes, all_probabilities * 100)
plt.xlabel("Klasse")
plt.ylabel("Wahrscheinlichkeit (%)")
plt.title("Wahrscheinlichkeiten für alle Klassen")

# Prozentwerte über die Balken schreiben
for bar, prob in zip(bars, all_probabilities):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, f"{prob * 100:.1f}%", ha='center')

plt.show()
