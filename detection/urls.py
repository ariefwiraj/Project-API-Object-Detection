from django.urls import path
from .views import ObjectDetectionView, ImageWithAnnotationsView, index_view, ImageListView, DetectionDetailView, ImageWithAnnotationsBoundingBoxView, ImageWithAnnotationsMasksView, ObjectDetectionFromEncodedView



urlpatterns = [
    path('', index_view, name='index'),
    path('detect/', ObjectDetectionView.as_view(), name='object-detection'),
    path('image-with-annotations/', ImageWithAnnotationsView.as_view(), name='image_with_annotations'),
    path('images-list/', ImageListView.as_view(), name='images_list'),  # New endpoint for image list
    path('detection-detail/', DetectionDetailView.as_view(), name='detection-detail'),
    path('image-with-annotations-bbox/', ImageWithAnnotationsBoundingBoxView.as_view(), name='image-with-annotations-bbox'),
    path('image-with-annotations-masks/', ImageWithAnnotationsMasksView.as_view(), name='image-with-annotations-masks'),
    path('detect-annotator/<str:image_name>/', ObjectDetectionFromEncodedView.as_view(), name='detect-annotator'),
]
