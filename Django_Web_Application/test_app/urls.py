from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_page),
    path('commandsrec/', views.commands_page),
    path('transformer/', views.transformer_page),
]