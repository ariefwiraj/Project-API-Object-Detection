# Generated by Django 4.2.13 on 2024-08-23 08:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("detection", "0006_detectionresult_file_name_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="detectionresult",
            name="id_predictions",
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name="detectionresult",
            name="image_base64",
            field=models.TextField(blank=True, default=""),
        ),
        migrations.AlterField(
            model_name="detectionresult",
            name="mask_data",
            field=models.TextField(blank=True, default="[]"),
        ),
    ]