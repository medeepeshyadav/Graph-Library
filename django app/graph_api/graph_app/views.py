import mimetypes
from django.forms import model_to_dict
from django.urls import reverse
from django.shortcuts import render, redirect
from django.http.response import HttpResponse
from .forms import AForm, BForm
from .models import Order, Parameter
import os
import uuid
from .graphlibrary import FeatureExtractor, PrepareData
import pandas as pd
import shutil

def handle_uploaded_file(f):
    if not os.path.isdir('./graph_app/uploaded'):
        os.makedirs('./graph_app/uploaded')

    with open('graph_app/uploaded/'+f.name, 'wb') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

    return f.name

def api(request):
    context = {}
    guid = uuid.uuid4()

    if request.POST:
        form = AForm(request.POST, request.FILES)
        if form.is_valid():
            g_type = form.cleaned_data["graph_type"]
            f_type = form.cleaned_data["feature_type"]
            size = form.cleaned_data["test_size"]
            workers = form.cleaned_data["n_jobs"]
            file_name = handle_uploaded_file(request.FILES['input_file'])
            ext = file_name.split(".")[-1]
            os.rename('./graph_app/uploaded/'+file_name, './graph_app/uploaded/'+str(guid)+'.'+ext)
            
            o = Order.objects.create(
                file_id = guid,
                in_file_loc='graph_app/uploaded/'+str(guid)+'.'+ext,
                out_file_loc = None,
                status='ready',
                graph_type=g_type,
                feature_type=f_type,
                test_size=size,
                n_jobs=workers,
            )

            o.save()
        
        check_form = BForm()
        # return redirect(reverse(check_status, kwargs={'key': o.file_id}))
        return render(request, 'upload_successful.html', {'key':o.file_id, 'chkform':check_form})

    else:
        form = AForm()
    context['form'] = form
    return render(request, 'index.html', context)


def check_status(request, key):
    """generate id and display to user for further ref."""
    return render(request, 'upload_successful.html', {'key': key})
    
# processing orders
def status(request):
    """Enter id and show status
    or Download""" 
    if request.method == 'POST':
        form = BForm(request.POST)

        if form.is_valid():
            key = form.cleaned_data['key']

        o = Order.objects.get(file_id=key)
        if o.status == 'done':            
            return render(request, 'download.html', {'path': o.out_file_loc})
        
        return render(request, 'display.html', {'status': o.status})
    
def download_file(request, filedir):
    # define django project base directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # open the file for reading its content
    path = open(filedir, 'rb')

    # set the mime type
    mime_type, _ = mimetypes.guess_type(filedir)

    # set the return value of the HttpResponse
    response = HttpResponse(path, content_type=mime_type)

    # set http header
    response['Content-Disposition'] = "attachment; filename=%s" %filedir.split("/")[-1]

    # return the response value
    return response
