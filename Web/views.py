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
# DICCIONARIO MEJORADO: Clasificación de reciclaje
# ============================================
CLASIFICACION = {
    # Reciclables
    "plastic": {"es": "Plástico", "Reciclable": True},
    "plastic bag": {"es": "Bolsa plástica", "Reciclable": True},
    "paper": {"es": "Papel", "Reciclable": True},
    "cardboard": {"es": "Cartón", "Reciclable": True},
    "glass": {"es": "Vidrio", "Reciclable": True},
    "metal": {"es": "Metal", "Reciclable": True},
    "can": {"es": "Lata", "Reciclable": True},
    "bottle": {"es": "Botella", "Reciclable": True},
    "tin": {"es": "Hojalata", "Reciclable": True},
    "aluminum": {"es": "Aluminio", "Reciclable": True},
    
    # No Reciclables
    "organic": {"es": "Orgánico", "Reciclable": False},
    "food waste": {"es": "Residuo orgánico", "Reciclable": False},
    "food": {"es": "Alimento", "Reciclable": False},
    "dirty paper": {"es": "Papel sucio", "Reciclable": False},
    "dirty plastic": {"es": "Plástico sucio", "Reciclable": False},
    "trash": {"es": "Basura", "Reciclable": False},
    "garbage": {"es": "Basura", "Reciclable": False},
}

# Palabras clave para búsqueda flexible
KEYWORDS_RECICLABLE = {
    "plastic": True,
    "paper": True,
    "cardboard": True,
    "glass": True,
    "metal": True,
    "can": True,
    "bottle": True,
    "aluminum": True,
    "tin": True,
    "container": True,
    "box": True,
    
    "food": False,
    "organic": False,
    "waste": False,
    "dirty": False,
    "trash": False,
    "garbage": False,
    "banana": False,
    "apple": False,
    "orange": False,
}

# ============================================
# MODELO LeNet
# ============================================
class LeNet(nn.Module):
    def __init__(self, num_classes=6):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

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

# ============================================
# INICIALIZACIÓN DE MODELOS (solo una vez)
# ============================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_lenet = LeNet(num_classes=6).to(device)
model_alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(device)
model_resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)
model_inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT).to(device)
model_mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT).to(device)
model_efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(device)

# Poner todos en modo evaluación desde el inicio
model_lenet.eval()
model_alexnet.eval()
model_resnet.eval()
model_inception.eval()
model_mobilenet.eval()
model_efficientnet.eval()

# Labels para LeNet
labels_lenet = ["vidrio", "papel", "plastico", "metal", "organico", "carton"]

# Diccionario de modelos
MODELOS = {
    1: {
        "model": model_lenet,
        "labels": labels_lenet,
        "name": "LeNet"
    },
    2: {
        "model": model_alexnet,
        "labels": models.AlexNet_Weights.DEFAULT.meta["categories"],
        "name": "AlexNet"
    },
    3: {
        "model": model_resnet,
        "labels": models.ResNet50_Weights.DEFAULT.meta["categories"],
        "name": "ResNet50"
    },
    4: {
        "model": model_inception,
        "labels": models.Inception_V3_Weights.DEFAULT.meta["categories"],
        "name": "InceptionV3"
    },
    5: {
        "model": model_mobilenet,
        "labels": models.MobileNet_V2_Weights.DEFAULT.meta["categories"],
        "name": "MobileNetV2"
    },
    6: {
        "model": model_efficientnet,
        "labels": models.EfficientNet_B0_Weights.DEFAULT.meta["categories"],
        "name": "EfficientNetB0"
    },
}

# Transformaciones
transform_standard = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_inception = transforms.Compose([
    transforms.Resize((299, 299)),  # Inception requiere 299x299
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================
# FUNCIÓN MEJORADA: Clasificar reciclabilidad
# ============================================
def clasificar_reciclable(predicted_label):
    """
    Determina si un objeto es reciclable basándose en su etiqueta.
    Usa búsqueda flexible con palabras clave.
    """
    label_lower = predicted_label.lower()
    
    # 1. Buscar coincidencia exacta en diccionario
    if label_lower in CLASIFICACION:
        info = CLASIFICACION[label_lower]
        return {
            "nombre_es": info["es"],
            "reciclable": "Reciclable" if info["reciclable"] else "No reciclable",
            "es_reciclable": info["reciclable"]
        }
    
    # 2. Buscar por palabras clave (búsqueda flexible)
    for keyword, es_reciclable in KEYWORDS_RECICLABLE.items():
        if keyword in label_lower:
            return {
                "nombre_es": predicted_label,
                "reciclable": "Reciclable" if es_reciclable else "No reciclable",
                "es_reciclable": es_reciclable
            }
    
    # 3. Si no se encuentra, retornar como desconocido
    return {
        "nombre_es": predicted_label,
        "reciclable": "Desconocido",
        "es_reciclable": None
    }

# ============================================
# VISTAS
# ============================================
def inicio(request):
    return render(request, 'index.html')


def analisis(request):
    if request.method == 'POST':
        data = request.POST.get('imagen')

        # Cargar imagen por defecto si no llega ninguna
        if not data:
            ruta = os.path.join(settings.BASE_DIR, "static", "img", "analizar.png")
            image = Image.open(ruta).convert("RGB")
        else:
            # Procesar base64
            if data.startswith('data:image'):
                data = data.split(',')[1]

            try:
                image_bytes = base64.b64decode(data)
                image = Image.open(BytesIO(image_bytes)).convert('RGB')
            except Exception as e:
                return JsonResponse({"error": f"Imagen no válida: {str(e)}"}, status=400)

        # Obtener ID del modelo
        id_modelo = int(request.POST.get('id_modelo', 1))

        # Verificar si existe el modelo
        if id_modelo not in MODELOS:
            return JsonResponse({"error": "Modelo no válido"}, status=400)

        modelo_info = MODELOS[id_modelo]
        model = modelo_info["model"]
        labels = modelo_info["labels"]

        # Seleccionar transformación según modelo
        if id_modelo == 4:  # Inception
            transform = transform_inception
        else:
            transform = transform_standard

        try:
            # Preparar imagen
            tensor = transform(image).unsqueeze(0).to(device)

            # Ejecutar predicción
            with torch.no_grad():
                output = model(tensor)
                
                # Si es Inception y está en entrenamiento, tiene salidas auxiliares
                if isinstance(output, tuple):
                    output = output[0]

            # Obtener probabilidades
            probabilities = F.softmax(output, dim=1)
            confidence, idx = torch.max(probabilities, 1)
            
            predicted_label = labels[idx.item()]
            confidence_percent = confidence.item() * 100

            # Clasificar si es reciclable
            clasificacion = clasificar_reciclable(predicted_label)

            return JsonResponse({
                "resultado": predicted_label,
                "resultado_es": clasificacion["nombre_es"],
                "reciclable": clasificacion["reciclable"],
                "es_reciclable": clasificacion["es_reciclable"],
                "confianza": f"{confidence_percent:.2f}%",
                "modelo_usado": modelo_info["name"]
            })

        except Exception as e:
            return JsonResponse({"error": f"Error en predicción: {str(e)}"}, status=500)

    return JsonResponse({"error": "Método no permitido"}, status=405)