import os
import cv2
import argparse
from ultralytics import YOLO  # Assuming YOLOv8 is being used from the `ultralytics` library

def run_inference(model, image):
    # Perform YOLOv8 inference
    results = model(image)
    return results

def draw_boxes(image, results, label_map):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class label

            label = f"{label_map[cls]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

def main(input_dir, output_dir, person_model_path, ppe_model_path):
    # Load YOLOv8 models
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the mapping from YOLO class IDs to human-readable labels
    label_map = {0: 'person', 1: 'hard-hat', 2: 'gloves', 3: 'mask', 4: 'glasses', 5: 'boots', 6: 'vest', 7: 'ppe-suit', 8: 'ear-protector', 9: 'safety-harness'}

    # Iterate through all images in the input directory
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        # Run person detection model on full image
        person_results = run_inference(person_model, image)

        # Process each detected person
        for result in person_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
                cls = int(box.cls[0])  # Class label

                if person_model.names[cls] == 'person':  # Only consider detections for 'person'
                    # Crop the image to the person's bounding box
                    cropped_img = image[y1:y2, x1:x2]

                    # Run PPE detection on the cropped image
                    ppe_results = run_inference(ppe_model, cropped_img)

                    # Map PPE detections back to the original image
                    for ppe_result in ppe_results:
                        ppe_boxes = ppe_result.boxes
                        for ppe_box in ppe_boxes:
                            ppe_x1, ppe_y1, ppe_x2, ppe_y2 = map(int, ppe_box.xyxy[0])
                            ppe_conf = ppe_box.conf[0]
                            ppe_cls = int(ppe_box.cls[0])

                            ppe_x1 += x1
                            ppe_x2 += x1
                            ppe_y1 += y1
                            ppe_y2 += y1

                            # Draw PPE detection boxes on the original image
                            label = f"{label_map[ppe_cls]} {ppe_conf:.2f}"
                            cv2.rectangle(image, (ppe_x1, ppe_y1), (ppe_x2, ppe_y2), (0, 0, 255), 2)
                            cv2.putText(image, label, (ppe_x1, ppe_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Save the resulting image
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, image)
        print(f"Processed and saved {img_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for person and PPE detection using YOLOv8.")
    parser.add_argument('--input_dir', type=str, required=True, help="Directory containing input images.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save output images.")
    parser.add_argument('--person_model', type=str, required=True, help="Path to the person detection model.")
    parser.add_argument('--ppe_model', type=str, required=True, help="Path to the PPE detection model.")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.person_model, args.ppe_model)
