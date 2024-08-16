import os
import cv2
import xml.etree.ElementTree as ET
import argparse
from ultralytics import YOLO  # Assuming YOLOv8 is being used from the `ultralytics` library


def parse_xml_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    annotations = []

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        annotations.append({'name': name, 'bbox': (xmin, ymin, xmax, ymax)})

    return annotations


def save_yolo_annotation(filename, annotations, img_width, img_height):
    with open(filename, 'w') as f:
        for annotation in annotations:
            name = annotation['name']
            bbox = annotation['bbox']
            xmin, ymin, xmax, ymax = bbox
            # Convert to YOLO format (class_id, x_center, y_center, width, height)
            x_center = (xmin + xmax) / 2.0 / img_width
            y_center = (ymin + ymax) / 2.0 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            class_id = name_to_id[name]
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


def process_images(image_dir, annotation_dir, model, output_image_dir, output_annotation_dir):
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        xml_path = os.path.join(annotation_dir, img_name.replace('.jpg', '.xml'))

        image = cv2.imread(img_path)
        img_height, img_width = image.shape[:2]

        # Perform YOLOv8 inference
        results = model(image)

        # Process YOLO results
        persons = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
                conf = box.conf[0]  # Confidence score
                cls = int(box.cls[0])  # Class label

                if model.names[cls] == 'person':  # Only consider detections for 'person'
                    persons.append((x1, y1, x2, y2))

        annotations = parse_xml_annotation(xml_path)

        for i, (xmin_person, ymin_person, xmax_person, ymax_person) in enumerate(persons):
            # Crop the image around the detected person
            cropped_img = image[ymin_person:ymax_person, xmin_person:xmax_person]
            cropped_img_path = os.path.join(output_image_dir, f"{img_name.replace('.jpg', f'_{i}.jpg')}")
            cv2.imwrite(cropped_img_path, cropped_img)

            # Adjust gear annotations relative to the cropped image
            modified_annotations = []
            for annotation in annotations:
                if annotation['name'] in name_to_id:
                    xmin_gear, ymin_gear, xmax_gear, ymax_gear = annotation['bbox']

                    # Check if gear is within the person's bounding box
                    if xmin_person <= xmin_gear <= xmax_person and ymin_person <= ymin_gear <= ymax_person:
                        xmin_gear_cropped = xmin_gear - xmin_person
                        ymin_gear_cropped = ymin_gear - ymin_person
                        xmax_gear_cropped = xmax_gear - xmin_person
                        ymax_gear_cropped = ymax_gear - ymin_person

                        # Ensure the bounding box fits within the cropped image
                        xmin_gear_cropped = max(0, xmin_gear_cropped)
                        ymin_gear_cropped = max(0, ymin_gear_cropped)
                        xmax_gear_cropped = min(cropped_img.shape[1], xmax_gear_cropped)
                        ymax_gear_cropped = min(cropped_img.shape[0], ymax_gear_cropped)

                        modified_annotations.append({
                            'name': annotation['name'],
                            'bbox': (xmin_gear_cropped, ymin_gear_cropped, xmax_gear_cropped, ymax_gear_cropped)
                        })

            # Save the modified annotations in YOLO format
            annotation_output_path = os.path.join(output_annotation_dir, f"{img_name.replace('.jpg', f'_{i}.txt')}")
            save_yolo_annotation(annotation_output_path, modified_annotations, cropped_img.shape[1], cropped_img.shape[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images and annotations using YOLOv8.")
    parser.add_argument('--image_dir', required=True, help='Path to the directory containing images.')
    parser.add_argument('--annotation_dir', required=True, help='Path to the directory containing XML annotations.')
    parser.add_argument('--output_image_dir', required=True, help='Path to the directory to save cropped images.')
    parser.add_argument('--output_annotation_dir', required=True, help='Path to the directory to save YOLO annotations.')
    parser.add_argument('--model_path', required=True, help='Path to the trained YOLOv8 model.')

    args = parser.parse_args()

    # Ensure output directories exist
    os.makedirs(args.output_image_dir, exist_ok=True)
    os.makedirs(args.output_annotation_dir, exist_ok=True)

    # Load YOLOv8 person detection model
    model = YOLO(args.model_path)

    # Define the mapping from class names to YOLO class IDs
    name_to_id = {
        'person': 0,
        'hard-hat': 1,
        'gloves': 2,
        'mask': 3,
        'glasses': 4,
        'boots': 5,
        'vest': 6,
        'ppe-suit': 7,
        'ear-protector': 8,
        'safety-harness': 9,
    }

    # Process all images and annotations
    process_images(args.image_dir, args.annotation_dir, model, args.output_image_dir, args.output_annotation_dir)
