#!/bin/bash

# Run yolo_label_converter.py
echo "Running yolo_label_converter.py..."
python yolo_label_converter.py
echo "yolo_label_converter.py completed."

# Run yolo_inference_and_postprocessing.py
echo "Running yolo_inference_and_postprocessing.py..."
python yolo_inference_and_postprocessing.py
echo "yolo_inference_and_postprocessing.py completed."

# Run pr_curve_validation.py
echo "Running pr_curve_validation.py..."
python pr_curve_validation.py
echo "pr_curve_validation.py completed."
