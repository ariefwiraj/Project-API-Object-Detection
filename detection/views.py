# detection/views.py
import os
import cv2
import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import ImageSerializer
import mrcnn.model as modellib
from django.http import HttpResponse, JsonResponse
from mrcnn import visualize
import time
import uuid

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import random
import base64

from .models import DetectionResult, EncodedImage
from django.shortcuts import render

def index_view(request):
    return render(request, 'index.html')

class ImageListView(APIView):
    def get(self, request, *args, **kwargs):
        # Ambil parameter filename dari query string
        file_name = request.query_params.get('file_name', None)

        # Jika filename disediakan, filter berdasarkan filename
        if file_name:
            detections = DetectionResult.objects.filter(file_name=file_name).values('id_predictions', 'image_base64')
        else:
            # Jika tidak ada filename, ambil semua data
            detections = DetectionResult.objects.all().values('id_predictions', 'image_base64')

        return Response(list(detections), status=status.HTTP_200_OK)


# Define the class labels
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

model.load_weights(filepath="mask_rcnn_coco.h5", 
                   by_name=True)


class ObjectDetectionFromEncodedView(APIView):
    def get(self, request, image_name, *args, **kwargs):
        try:
            # Ambil gambar berdasarkan image_name dari tabel EncodedImage
            encoded_image_record = EncodedImage.objects.filter(image_name=image_name).last()
            if not encoded_image_record:
                return Response({"error": "Encoded image with the specified name not found."}, status=status.HTTP_404_NOT_FOUND)

            # Decode base64 menjadi gambar
            image_data = base64.b64decode(encoded_image_record.image_base64)
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except Exception as e:
            return Response({"error": "Failed to process image data.", "details": str(e)}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Lakukan deteksi objek
            results = model.detect([image], verbose=1)
            r = results[0]
        except Exception as e:
            return Response({"error": "Object detection failed.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            # Generate a unique id_predictions using current timestamp or a counter
            id_predictions = int(time.time())  # Gunakan timestamp sebagai ID atau logika lain yang Anda inginkan

            # Generate a unique file name based on ID and timestamp
            unique_file_name = f'detection_{id_predictions}_{uuid.uuid4().hex[:8]}.jpg'

            response_data = {
                'file_name': unique_file_name,  # Tambahkan file_name di bagian atas
                'id_predictions': id_predictions,
                'width': image.shape[1],
                'height': image.shape[0],
                'predictions': []
            }

            # Encode the processed image to base64
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            image_base64 = base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            return Response({"error": "Failed to encode image.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            for i in range(len(r['rois'])):
                y1, x1, y2, x2 = r['rois'][i]
                x = int(x1)
                y = int(y1)
                width = int(x2 - x1)
                height = int(y2 - y1)

                mask = r['masks'][:, :, i]
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                polygons = []
                for contour in contours:
                    polygon = [{'x': int(point[0][0]), 'y': int(point[0][1])} for point in contour]
                    polygons.append(polygon)

                detection_result = DetectionResult.objects.create(
                    id_predictions=id_predictions,  # Gunakan ID yang dihasilkan untuk set ini
                    class_name=CLASS_NAMES[r['class_ids'][i]],
                    bounding_box_x=x,
                    bounding_box_y=y,
                    bounding_box_width=width,
                    bounding_box_height=height,
                    score=float(r['scores'][i]),
                    mask_data=polygons,
                    image_base64=image_base64,  # Simpan gambar dalam format base64 untuk setiap deteksi
                    file_name=unique_file_name  # Simpan nama file unik
                )

                response_data['predictions'].append({
                    'id': detection_result.id,
                    'class_name': detection_result.class_name,
                    'bounding_box': {
                        'x': detection_result.bounding_box_x,
                        'y': detection_result.bounding_box_y,
                        'width': detection_result.bounding_box_width,
                        'height': detection_result.bounding_box_height
                    },
                    'score': detection_result.score,
                    'mask': detection_result.get_mask_data(),
                    'image_base64': detection_result.image_base64,  # Sertakan base64 dalam respons
                })
        except Exception as e:
            return Response({"error": "Failed to save detection results to the database.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response(response_data, status=status.HTTP_200_OK)




class ObjectDetectionView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ImageSerializer(data=request.data)
        
        # Validasi data yang diterima
        if serializer.is_valid():
            try:
                image = serializer.validated_data['image']
                image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                return Response({"error": "Failed to process image data.", "details": str(e)}, status=status.HTTP_400_BAD_REQUEST)

            try:
                # Lakukan deteksi objek
                results = model.detect([image], verbose=1)
                r = results[0]
            except Exception as e:
                return Response({"error": "Object detection failed.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            try:
                # Generate a unique id_predictions using current timestamp or a counter
                id_predictions = int(time.time())  # Gunakan timestamp sebagai ID atau logika lain yang Anda inginkan

                # Generate a unique file name based on ID and timestamp
                unique_file_name = f'detection_{id_predictions}_{uuid.uuid4().hex[:8]}.jpg'

                response_data = {
                    'file_name': unique_file_name,  # Tambahkan file_name di bagian atas
                    'id_predictions': id_predictions,
                    'width': image.shape[1],
                    'height': image.shape[0],
                    'predictions': []
                }

                # Encode the processed image to base64
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                image_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                return Response({"error": "Failed to encode image.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            try:
                for i in range(len(r['rois'])):
                    y1, x1, y2, x2 = r['rois'][i]
                    x = int(x1)
                    y = int(y1)
                    width = int(x2 - x1)
                    height = int(y2 - y1)

                    mask = r['masks'][:, :, i]
                    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    polygons = []
                    for contour in contours:
                        polygon = [{'x': int(point[0][0]), 'y': int(point[0][1])} for point in contour]
                        polygons.append(polygon)

                    detection_result = DetectionResult.objects.create(
                        id_predictions=id_predictions,  # Gunakan ID yang dihasilkan untuk set ini
                        class_name=CLASS_NAMES[r['class_ids'][i]],
                        bounding_box_x=x,
                        bounding_box_y=y,
                        bounding_box_width=width,
                        bounding_box_height=height,
                        score=float(r['scores'][i]),
                        mask_data=polygons,
                        image_base64=image_base64,  # Simpan gambar dalam format base64 untuk setiap deteksi
                        file_name=unique_file_name  # Simpan nama file unik
                    )

                    response_data['predictions'].append({
                        'id': detection_result.id,
                        'class_name': detection_result.class_name,
                        'bounding_box': {
                            'x': detection_result.bounding_box_x,
                            'y': detection_result.bounding_box_y,
                            'width': detection_result.bounding_box_width,
                            'height': detection_result.bounding_box_height
                        },
                        'score': detection_result.score,
                        'mask': detection_result.get_mask_data(),
                        'image_base64': detection_result.image_base64,  # Sertakan base64 dalam respons
                    })
            except Exception as e:
                return Response({"error": "Failed to save detection results to the database.", "details": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            return Response(response_data, status=status.HTTP_201_CREATED)
        else:
            return Response({"error": "Invalid input data.", "details": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)


def generate_random_color():
    # Generate random values for R, G, B components
    r = random.randint(150, 255)  # Ensure R value is between 150 and 255
    g = random.randint(150, 255)  # Ensure G value is between 150 and 255
    b = random.randint(150, 255)  # Ensure B value is between 150 and 255
    return (r, g, b)

class ImageWithAnnotationsView(APIView):
    def get(self, request, *args, **kwargs):
        detection_id = request.query_params.get('id')
        id_predictions = request.query_params.get('id_predictions')
        include_all = request.query_params.get('include_all', 'false').lower() == 'true'
        include_masks = request.query_params.get('include_masks', 'false').lower() == 'true'

        if not detection_id and not id_predictions:
            return Response({"error": "No detection ID or ID Predictions provided"}, status=status.HTTP_400_BAD_REQUEST)

        if detection_id:
            detections = DetectionResult.objects.filter(id=detection_id)
        elif id_predictions:
            detections = DetectionResult.objects.filter(id_predictions=id_predictions)

        if not detections.exists():
            return Response({"error": "Detection results not found"}, status=status.HTTP_404_NOT_FOUND)

        # Assume all detections share the same base64 image, take the first one
        base64_image = detections.first().image_base64

        # Decode the base64 image to a NumPy array for OpenCV
        image_data = base64.b64decode(base64_image)
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return Response({"error": "Decoded image is None"}, status=status.HTTP_404_NOT_FOUND)

        # Convert image from BGR to RGB for OpenCV processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for detection in detections:
            # Generate a random color for each detection
            color = generate_random_color()
            color_bgr = (color[2], color[1], color[0])  # Convert RGB to BGR for OpenCV

            # Initialize x and y for text positioning, default to bounding box coordinates
            x = detection.bounding_box_x
            y = detection.bounding_box_y

            if include_all or not include_masks:
                # Draw bounding box
                width = detection.bounding_box_width
                height = detection.bounding_box_height
                cv2.rectangle(image_rgb, (x, y), (x + width, y + height), color_bgr, 2)

            if include_all or include_masks:
                # Create a transparent overlay for the mask
                overlay = image_rgb.copy()
                mask_data = detection.get_mask_data()
                for mask in mask_data:
                    points = np.array([(point['x'], point['y']) for point in mask], dtype=np.int32)
                    cv2.fillPoly(overlay, [points], color_bgr)  # Fill mask area with color

                # Blend overlay with original image to achieve transparency
                alpha = 0.4  # Set transparency level (0.0 to 1.0)
                cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0, image_rgb)

                # Draw masks as outlines with the same color as fill
                for mask in mask_data:
                    points = np.array([(point['x'], point['y']) for point in mask], dtype=np.int32)
                    cv2.polylines(image_rgb, [points], isClosed=True, color=color_bgr, thickness=2)

            # Add class name and score if `include_all` is true or bounding box is drawn
            if include_all or not include_masks:
                cv2.putText(image_rgb, f'{detection.class_name}: {detection.score:.2f}', 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)

        # Convert image back to BGR for display
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.jpg', image_bgr)
        response = HttpResponse(img_encoded.tobytes(), content_type="image/jpeg")
        return response

class DetectionDetailView(APIView):
    def get(self, request, *args, **kwargs):
        detection_id = request.query_params.get('id')
        id_predictions = request.query_params.get('id_predictions')
        ids = request.query_params.get('ids')
        file_name = request.query_params.get('file_name')  # Get file_name from query parameters

        if not detection_id and not id_predictions and not ids and not file_name:
            return Response({"error": "No detection ID, ID Predictions, IDs, or file name provided"}, status=status.HTTP_400_BAD_REQUEST)

        if ids:
            # Handle multiple IDs
            ids_list = ids.split(',')
            detections = DetectionResult.objects.filter(id__in=ids_list)
        elif detection_id:
            detections = DetectionResult.objects.filter(id=detection_id)
        elif id_predictions:
            detections = DetectionResult.objects.filter(id_predictions=id_predictions)
        elif file_name:
            detections = DetectionResult.objects.filter(file_name=file_name)  # Filter based on file name

        if not detections.exists():
            return Response({"error": "Detection results not found"}, status=status.HTTP_404_NOT_FOUND)

        # Prepare a list of results to return
        results = []
        for detection in detections:
            results.append({
                'id': detection.id,  # Include the ID
                'class_name': detection.class_name,
                'bounding_box': {
                    'x': detection.bounding_box_x,
                    'y': detection.bounding_box_y,
                    'width': detection.bounding_box_width,
                    'height': detection.bounding_box_height
                },
                'score': detection.score,
                'mask_coordinates': detection.get_mask_data(),
                'file_name': detection.file_name  # Include the file name in the response
            })

        return Response(results, status=status.HTTP_200_OK)


    
    
class ImageWithAnnotationsBoundingBoxView(APIView):
    def get(self, request, *args, **kwargs):
        ids = request.query_params.get('ids')
        if not ids:
            return Response({"error": "No detection IDs provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            ids_list = list(map(int, ids.split(',')))  # Convert the IDs to integers
        except ValueError:
            return Response({"error": "Invalid detection IDs"}, status=status.HTTP_400_BAD_REQUEST)

        detections = DetectionResult.objects.filter(id__in=ids_list)
        if not detections.exists():
            return Response({"error": "No detection results found for the provided IDs"}, status=status.HTTP_404_NOT_FOUND)

        # Create a blank image for combining all results
        combined_image = None

        for detection in detections:
            base64_image = detection.image_base64
            image_data = base64.b64decode(base64_image)
            image_np = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            if image is None:
                continue  # Skip if image decoding fails

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            color = generate_random_color()
            color_bgr = (color[2], color[1], color[0])
            x = detection.bounding_box_x
            y = detection.bounding_box_y
            width = detection.bounding_box_width
            height = detection.bounding_box_height
            cv2.rectangle(image_rgb, (x, y), (x + width, y + height), color_bgr, 2)

            if combined_image is None:
                combined_image = image_rgb
            else:
                combined_image = np.maximum(combined_image, image_rgb)

        if combined_image is None:
            return Response({"error": "No valid images to combine"}, status=status.HTTP_404_NOT_FOUND)

        image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.jpg', image_bgr)
        response = HttpResponse(img_encoded.tobytes(), content_type="image/jpeg")
        return response


class ImageWithAnnotationsMasksView(APIView):
    def get(self, request, *args, **kwargs):
        ids = request.query_params.get('ids')
        include_all = request.query_params.get('include_all', 'false').lower() == 'true'
        include_masks = request.query_params.get('include_masks', 'false').lower() == 'true'

        if not ids:
            return Response({"error": "No detection IDs provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            ids_list = list(map(int, ids.split(',')))  # Convert the IDs to integers
        except ValueError:
            return Response({"error": "Invalid detection IDs"}, status=status.HTTP_400_BAD_REQUEST)

        detections = DetectionResult.objects.filter(id__in=ids_list)
        if not detections.exists():
            return Response({"error": "Detection results not found"}, status=status.HTTP_404_NOT_FOUND)

        # Assume all detections share the same base64 image, take the first one
        base64_image = detections.first().image_base64

        # Decode the base64 image to a NumPy array for OpenCV
        image_data = base64.b64decode(base64_image)
        image_np = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image is None:
            return Response({"error": "Decoded image is None"}, status=status.HTTP_404_NOT_FOUND)

        # Convert image from BGR to RGB for OpenCV processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        for detection in detections:
            # Generate a random color for each detection
            color = generate_random_color()
            color_bgr = (color[2], color[1], color[0])  # Convert RGB to BGR for OpenCV

            # Initialize x and y for text positioning, default to bounding box coordinates
            x = detection.bounding_box_x
            y = detection.bounding_box_y

            if include_all or not include_masks:
                # Create a transparent overlay for the mask
                overlay = image_rgb.copy()
                mask_data = detection.get_mask_data()
                for mask in mask_data:
                    points = np.array([(point['x'], point['y']) for point in mask], dtype=np.int32)
                    cv2.fillPoly(overlay, [points], color_bgr)  # Fill mask area with color

                # Blend overlay with original image to achieve transparency
                alpha = 0.4  # Set transparency level (0.0 to 1.0)
                cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0, image_rgb)

                # Draw masks as outlines with the same color as fill
                for mask in mask_data:
                    points = np.array([(point['x'], point['y']) for point in mask], dtype=np.int32)
                    cv2.polylines(image_rgb, [points], isClosed=True, color=color_bgr, thickness=2)

        # Convert image back to BGR for display
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.jpg', image_bgr)
        response = HttpResponse(img_encoded.tobytes(), content_type="image/jpeg")
        return response


