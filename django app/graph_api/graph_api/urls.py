from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', view=include('graph_app.urls')),
    path('status/', view=include('graph_app.urls', namespace='graph_app')),
]
