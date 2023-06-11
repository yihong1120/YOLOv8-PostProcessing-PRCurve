import os
import pickle

def convert_yolo_coordinates(folder_path, img_width=640, img_height=640):
    # Store all data in an array
    all_labels = []

    # Read all txt files in the folder
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Sort file names
    file_names.sort()

    # Read all txt files in the folder
    for file_name in file_names:
        with open(os.path.join(folder_path, file_name), 'r') as f:
            lines = f.readlines()
            bounding_boxes = []
            for line in lines:
                # Split each line of data and convert to floating point number
                data = list(map(float, line.strip().split()))
                # Convert YOLO dimensions to regular coordinates
                class_id = int(data[0])
                x_center = data[1] * img_width
                y_center = data[2] * img_height
                width = data[3] * img_width
                height = data[4] * img_height
                # Calculate the coordinates of the upper left and lower right corners
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                # Add the data to the array
                bounding_boxes.append([class_id, x1, y1, x2, y2])
        all_labels.append(bounding_boxes)

    return all_labels

def save_as_pickle(data, file_name):
    # Save the list as a pickle file
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)

def main():
    folder_path = r'/route/to/your/valid/labels'
    all_labels = convert_yolo_coordinates(folder_path)
    save_as_pickle(all_labels, '../y_true.pkl')

if __name__ == "__main__":
    main()
