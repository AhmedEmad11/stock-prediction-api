from django.urls import path

from .views import facebook

urlpatterns = [
    path('facebook', facebook, name='facebook')
]