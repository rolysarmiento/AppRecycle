from django.shortcuts import render
from django.http import HttpResponse

import base64
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from io import BytesIO


def inicio(request):
    return render(request, 'index.html')

def analisis(request):
    
    if request.method == "POST":
        imagen_base64 = request.POST.get("imagen")
        if not imagen_base64:
            return JsonResponse({"error": "No se recibió imagen"}, status=400)

        # Decodificar base64
        header, data = imagen_base64.split(',', 1)
        imagen_bytes = base64.b64decode(data)
        imagen = Image.open(BytesIO(imagen_bytes))

        # Aquí puedes analizar la imagen con IA, OpenCV, etc.
        resultado = "Objeto detectado: cama"

        return JsonResponse({"resultado": resultado})