from django.contrib import admin
from .models import PredictionHistory


@admin.register(PredictionHistory)
class PredictionHistoryAdmin(admin.ModelAdmin):
    list_display = ['disease_name', 'confidence', 'model_type', 'created_at']
    list_filter = ['model_type', 'created_at']
    search_fields = ['disease_name']
    readonly_fields = ['created_at']

