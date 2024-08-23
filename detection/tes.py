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
            color_bgr = (color[2], color[1], color[0])  # Convert RGB to BGR for OpenCV

            # Initialize x and y for text positioning, default to bounding box coordinates
            x = detection.bounding_box_x
            y = detection.bounding_box_y

            # if include_all or not include_masks:
            #     # Draw bounding box
            #     width = detection.bounding_box_width
            #     height = detection.bounding_box_height
            #     cv2.rectangle(image_rgb, (x, y), (x + width, y + height), color_bgr, 2)

            
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

        if combined_image is None:
            return Response({"error": "No valid images to combine"}, status=status.HTTP_404_NOT_FOUND)

        image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.jpg', image_bgr)
        response = HttpResponse(img_encoded.tobytes(), content_type="image/jpeg")
        return response


class ImageWithAnnotationsMasksView(APIView):
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

            # if include_all or not include_masks:
            #     # Draw bounding box
            #     width = detection.bounding_box_width
            #     height = detection.bounding_box_height
            #     cv2.rectangle(image_rgb, (x, y), (x + width, y + height), color_bgr, 2)

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

            # # Add class name and score if `include_all` is true or bounding box is drawn
            # if include_all or not include_masks:
            #     cv2.putText(image_rgb, f'{detection.class_name}: {detection.score:.2f}', 
            #                 (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_bgr, 2)

        # Convert image back to BGR for display
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, img_encoded = cv2.imencode('.jpg', image_bgr)
        response = HttpResponse(img_encoded.tobytes(), content_type="image/jpeg")
        return response