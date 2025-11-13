from django.shortcuts import render
from django.http import HttpResponse

import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from io import BytesIO

from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

import torch
from torchvision import models, transforms


def inicio(request):
    return render(request, 'index.html')

# Cargar modelo AlexNet preentrenado
model = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
model.eval()

# Transformaciones para preparar la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def analisis(request):
    if request.method == 'POST':
        data = request.POST.get('imagen')  # Recibes el base64 desde el frontend
        
        # Verifica que exista la imagen
        if not data:
            return JsonResponse({'error': 'No se recibió ninguna imagen'}, status=400)

        # Si viene en formato data:image/jpeg;base64, eliminar el encabezado
        if data.startswith('data:image'):
            data = data.split(',')[1]

        try:
            # Decodificar el base64 a bytes
            image_data = base64.b64decode(data)
            
            # Abrir como imagen PIL
            image = Image.open(BytesIO(image_data)).convert('RGB')

            # Transformar imagen para AlexNet
            input_tensor = transform(image).unsqueeze(0)

            # Pasar por el modelo
            with torch.no_grad():
                output = model(input_tensor)
            
            # Obtener clase predicha
            _, predicted_idx = torch.max(output, 1)
            class_idx = predicted_idx.item()

            # Obtener etiquetas (usando las de ImageNet)
            labels = models.AlexNet_Weights.DEFAULT.meta["categories"]
            predicted_label = labels[class_idx]

            return JsonResponse({'resultado': predicted_label})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Método no permitido'}, status=405)
