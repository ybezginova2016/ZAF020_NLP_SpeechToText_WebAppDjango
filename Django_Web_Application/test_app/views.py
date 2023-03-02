from django.shortcuts import render

# Create your views here.

def home_page(request):
    return render(request, 'test_app/home.html')

def commands_page(request):
    return render(request, 'test_app/commands_page.html')

def transformer_page(request):
    return render(request, 'test_app/transformer_page.html')
