from django.urls import path
from . import views
app_name = 'graph_app'
urlpatterns = [
    path('', views.api),
    path('status/', views.status, name='display_query'),
    path('check_status/<uuid:key>/', views.check_status, name='check_status'),
    # to download file
    # <path: "kwarg name"> <--- this accepts the file path argument to be supplied to the view.
    path(r'download/<path:filedir>', views.download_file, name='download_file'),
    
    
]

