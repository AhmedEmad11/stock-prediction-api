from django.urls import path

from .views import facebook, overview

urlpatterns = [
    path('', overview, name='overview'),
    path('facebook', facebook, name='facebook')
]