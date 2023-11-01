import os
from django.shortcuts import redirect, render
from django.http import HttpResponse
import torch

from django.http import JsonResponse
# Create your views here.


from pore2d.detection import detect_objects
from django.shortcuts import render



def home(request):
    return render(request, 'home.html')

from django.core.files.storage import default_storage
from django.conf import settings

def app(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        save_path = default_storage.save(image_file.name, image_file)
    
        # Perform object detection on the saved image
        detection_results = detect_objects(save_path)
        
        context = {
            'image_path': save_path,
            'detection_results': detection_results,
        }
        return render(request, 'detected_image.html', context)

    return render(request, 'app.html')

def detected_image(request):
    # Retrieve the image path and detection results from the query parameters
    image_path = request.GET.get('image_path')
    detection_results = request.GET.get('detection_results')

    # Render the detected_image template with the image path and detection results
    context = {
        'image_path': image_path,
        'detection_results': detection_results,
    }
    return render(request, 'detected_image.html', context)

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')


