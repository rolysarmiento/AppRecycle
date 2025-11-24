from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
import base64
import os
from PIL import Image
from io import BytesIO
import torch
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F


# ============================================
# CARGAR MODELOS DISPONIBLES
# ============================================

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # LeNet original usa 1 canal, aquí lo adapto a RGB (3)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Cambia según tamaño después de conv (224x224 → 53x53)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)  # 10 clases (puedes cambiarlo)

    def forward(self, x):
        x = F.relu(self.conv1(x))  
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))  
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model_lenet = LeNet()
model_alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
model_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model_inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
model_mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model_efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

labels_lenet = [f"Clase {i}" for i in range(10)]
labels_lenet = ["vidrio", "papel", "plastico", "metal", "organico", "carton"]

# Diccionario general de modelos
MODELOS = {
    1: {
        "model": model_lenet,
        "labels": labels_lenet
    },
    2: {
        "model": model_alexnet,
        "labels": models.AlexNet_Weights.DEFAULT.meta["categories"]
    },
    3: {
        "model": model_resnet,
        "labels": models.ResNet50_Weights.DEFAULT.meta["categories"]
    },
    4: {
        "model": model_inception,
        "labels": models.Inception_V3_Weights.DEFAULT.meta["categories"]
    },
    5: {
        "model": model_mobilenet,
        "labels": models.MobileNet_V2_Weights.DEFAULT.meta["categories"]
    },
    6: {
        "model": model_efficientnet,
        "labels": models.EfficientNet_B0_Weights.DEFAULT.meta["categories"]
    },
}

# Transformación estándar
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def inicio(request):
    return render(request, 'index.html')


def analisis(request):
    if request.method == 'POST':
        data = request.POST.get('imagen')

        # Si no llega una imagen → cargar imagen por defecto
        if not data:
            ruta = os.path.join(settings.BASE_DIR, "static", "img", "analizar.png")
            image = Image.open(ruta).convert("RGB")
        else:
            # Si llega base64
            if data.startswith('data:image'):
                data = data.split(',')[1]

            try:
                image_bytes = base64.b64decode(data)
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
            except Exception as e:
                return JsonResponse({"error": "Imagen no válida: " + str(e)}, status=400)

        # Obtener id del modelo
        id_modelo = int(request.POST.get('id_modelo', 2))

        # Verificar si el modelo existe
        if id_modelo not in MODELOS:
            return JsonResponse({"error": "Modelo no válido"}, status=400)

        modelo_info = MODELOS[id_modelo]
        model = modelo_info["model"]
        labels = modelo_info["labels"]

        if model is None:
            return JsonResponse({"error": "Modelo aún no implementado (por ejemplo LeNet)"}, status=400)

        # Poner en evaluación
        model.eval()

        try:
            # Preparar imagen
            tensor = transform(image).unsqueeze(0)

            # Ejecutar predicción
            with torch.no_grad():
                output = model(tensor)

            _, idx = torch.max(output, 1)
            predicted_label = labels[idx.item()]

            return JsonResponse({"resultado": predicted_label})

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Método no permitido"}, status=405)
