import os
import xml.etree.ElementTree as ET
import argparse


def convert_pascal_voc_to_yolo(label_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for label_file in os.listdir(label_folder):
        if label_file.endswith('.xml'):
            tree = ET.parse(os.path.join(label_folder, label_file))
            root = tree.getroot()

            img_width = int(root.find('size/width').text)
            img_height = int(root.find('size/height').text)

            yolo_labels = []

            for obj in root.findall('object'):
                class_name = obj.find('name').text

                if class_name == "person":
                    xmin = int(obj.find('bndbox/xmin').text)
                    ymin = int(obj.find('bndbox/ymin').text)
                    xmax = int(obj.find('bndbox/xmax').text)
                    ymax = int(obj.find('bndbox/ymax').text)

                    center_x = (xmin + xmax) / 2 / img_width
                    center_y = (ymin + ymax) / 2 / img_height
                    width = (xmax - xmin) / img_width
                    height = (ymax - ymin) / img_height

                    # YOLOv8 format: class_id center_x center_y width height
                    yolo_labels.append(f"0 {center_x} {center_y} {width} {height}")

            # Determine the corresponding txt file name
            output_file = os.path.join(output_folder, label_file.replace('.xml', '.txt'))

            # Write to the file if there are any labels, otherwise create an empty file
            with open(output_file, 'w') as f:
                if yolo_labels:
                    f.write("\n".join(yolo_labels))
                else:
                    f.write("")


def main():
    parser = argparse.ArgumentParser(description='Convert Pascal VOC annotations to YOLOv8 format')
    parser.add_argument('label_folder', type=str, help='Path to the folder containing Pascal VOC annotations')
    parser.add_argument('output_folder', type=str, help='Path to the folder where YOLOv8 annotations will be saved')

    args = parser.parse_args()
    convert_pascal_voc_to_yolo(args.label_folder, args.output_folder)


if __name__ == '__main__':
    main()
