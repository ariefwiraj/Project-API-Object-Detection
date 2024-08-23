from django.db import models
import json

class DetectionResult(models.Model):
    id = models.AutoField(primary_key=True)  # Auto-increment ID for primary key
    id_predictions = models.IntegerField(default=0)  # Provide a default value
    class_name = models.CharField(max_length=255)
    bounding_box_x = models.IntegerField()
    bounding_box_y = models.IntegerField()
    bounding_box_width = models.IntegerField()
    bounding_box_height = models.IntegerField()
    score = models.FloatField()
    mask_data = models.TextField(blank=True, default='[]')  # Stores mask data as JSON string
    image_base64 = models.TextField(blank=True, default='')  # Stores image encoded in base64
    file_name = models.CharField(max_length=255, blank=True)

    def save(self, *args, **kwargs):
        # Convert mask_data to JSON string before saving
        if isinstance(self.mask_data, list):
            self.mask_data = json.dumps(self.mask_data)
        super().save(*args, **kwargs)

    def get_mask_data(self):
        # Convert JSON string back to list
        return json.loads(self.mask_data)

    def __str__(self):
        return f'{self.class_name} - {self.id}'
