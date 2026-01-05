"""
Django models for Skin Disease Detection app.

For this project, we don't need database models as we're doing
real-time predictions. But we can add models for storing prediction
history if needed in the future.
"""
from django.db import models


class PredictionHistory(models.Model):
    """Optional: Store prediction history for analytics."""
    image_path = models.CharField(max_length=500)
    disease_name = models.CharField(max_length=200)
    confidence = models.FloatField()
    model_type = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Prediction Histories'

    def __str__(self):
        return f"{self.disease_name} ({self.confidence:.2%}) - {self.created_at}"

