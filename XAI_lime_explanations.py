import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
import torch
import torch.nn as nn
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import time
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#Defining the model
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=64 * 56 * 56, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return {'logits': x}
#Instance of the Conv Net
net = Net(num_classes=10) #Amount of classes
#Loading the model
model_save_path = ".venv/Model/net_model.pth"
net.load_state_dict(torch.load(model_save_path))
net.eval()  #set the model to evaluation mode
print("Modell erfolgreich geladen und bereit für die Verwendung.")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load the image
def get_image(path):
    with open(os.path.abspath(path), 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
img = get_image('.venv/Bilderzsm/Komplex_Bild_nur_Tab.png')
plt.imshow(img)
plt.show()

#Define the transformation for the input image
def get_input_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    return transf

#Transform the input image
transform = get_input_transform()
input_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Unsqueezing adds a batch dimension

#Make a prediction
with torch.no_grad():
    output = net(input_tensor)

logits = output['logits']
probabilities = torch.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()

#Probabilities and class labels
class_labels = [1,10,2,3,4,5,6,7,8,9]  #Class labels
probabilities_list = probabilities.squeeze().tolist()  #Convert the tensor to a list

#Output the prediction results
print("Vorhersageergebnisse:")
for i, (label, prob) in enumerate(zip(class_labels, probabilities_list)):
    print(f"Klasse {label}: {prob:.2%} Wahrscheinlichkeit")

print(f"\nDie vorhergesagte Klasse ist: Klasse {class_labels[predicted_class]} mit {probabilities_list[predicted_class]:.2%} Wahrscheinlichkeit.")

#Define the transformation for the PIL image
def get_pil_transform():
    transf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224)
    ])

    return transf

#Define the transformation for the input image
def get_preprocess_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf

pill_transf = get_pil_transform()
preprocess_transform = get_preprocess_transform()

#Define the function to make a prediction
def batch_predict(images):
    net.eval()
    batch = torch.stack(tuple(preprocess_transform(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    batch = batch.to(device)

    with torch.no_grad():
        outputs = net(batch)
        logits = outputs['logits']  #Extract logits from the dictionary
        probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()

#Make a prediction
test_pred = batch_predict([pill_transf(img)])
test_pred.squeeze().argmax()

#Create the explainer
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(np.array(pill_transf(img)),
                                         batch_predict,
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=5000) #Amount of samples send to the model



#Superpixel-visualization (positive_only=True)
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
img_boundry1 = mark_boundaries(temp/255.0, mask)
plt.title('Superpixel-Visualisierung Top 5 Features')
plt.imshow(img_boundry1)
plt.show()

#Heatmap of the weights of the superpixels
ind = explanation.top_labels[0]
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
plt.imshow(heatmap, cmap='RdYlGn', vmin=-heatmap.max(), vmax=heatmap.max())
plt.colorbar()
plt.title('Heatmap Gewichte Superpixel')
plt.show()



