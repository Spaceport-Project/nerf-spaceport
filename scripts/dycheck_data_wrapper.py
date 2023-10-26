import json
import os
from PIL import Image

#This script is a wrapper for nerfplayer dycheck data parser, which gets spaceport cage data and calibration infos
#and makes necessary modifications on image names and creates metadata.json, train.json, val.json, dataset.json, scene.json
#extra.json files for training nerfplayer on multiple camera data using nerfstudio data managers and data parsers.
#It also creates camera json files for each camera using calibration infos.

# Load static variables from JSON
with open('config.json', 'r') as f:
    config = json.load(f)

DATA_DIR_IN = config['DATA_DIR_IN']
DATA_DIR_OUT = config['DATA_DIR_OUT']
JSON_DIR_OUT = config['JSON_DIR_OUT']
CALIB_METHOD = config['CALIB_METHOD']
CALIB_JSON_DIR = config['CALIB_JSON_DIR']


def rename_images(image_dir, output_dir):
    """
    Rename and move images from image_dir to output_dir. 

    Parameters:
    - image_dir (str): Path to the directory containing images to be renamed.
    - output_dir (str): Path to the directory where renamed images will be stored.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Dictionary to hold the mapping of serial numbers to new camera indices
    serial_number_mapping = {}

    # List to hold the file details
    file_details = []

    # Iterate over all files in the image directory
    for filename in os.listdir(image_dir):
        # Split the filename to extract frame_idx and serial_number
        parts = filename.split('_')
        if len(parts) >= 3:
            frame_idx = int(parts[1])
            serial_number = parts[2]
            
            # If the serial number is not in the mapping, assign a new camera index to it
            if serial_number not in serial_number_mapping:
                serial_number_mapping[serial_number] = len(serial_number_mapping)
            
            # Append the details to the file details list
            file_details.append((filename, serial_number, frame_idx))

    # Sort the file details by frame index
    file_details.sort(key=lambda x: x[2])
    frame_idx_offset = file_details[0][2]

    img_save_dir = os.path.join(output_dir, '1x')
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    # Check if output directory exists, if not, create it
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    # Iterate over the sorted file details and save the images with new names
    for i, (filename, serial_number, frame_idx) in enumerate(file_details):
        frame_idx_str = str(frame_idx - frame_idx_offset).zfill(5)
        new_cam_idx = str(serial_number_mapping[serial_number]).zfill(2)
        new_filename = f"{new_cam_idx}_{frame_idx_str}.png"

        # Open and save the image instead of copying directly
        with Image.open(os.path.join(image_dir, filename)) as img:
            img.save(os.path.join(img_save_dir, new_filename))

        print(f"Saved {filename} as {new_filename}")

    json_output_path = os.path.join(output_dir, 'serial_number_mapping.json')
    # Save serial_number_mapping to a JSON file
    with open(json_output_path, 'w') as json_file:
        json.dump(serial_number_mapping, json_file, indent=4)

    print(f"Saved serial number mapping to {json_output_path}")

    return serial_number_mapping


def create_dataset_jsons(modified_data_path, json_save_dir):
    """
    Create JSON files and save them to a specified directory.

    Parameters:
    - modified_data_path (str): Path to the modified data.
    - json_save_dir (str): Directory where JSON files will be saved.
    """

    if not os.path.exists(json_save_dir):
        os.makedirs(json_save_dir)

    # List all files in the output directory
    modified_data_path = os.path.join(modified_data_path, '1x')
    file_list = os.listdir(modified_data_path)

    # Filter out only .png files
    image_files = [file for file in file_list if file.endswith('.png')]

    # Sort the image files
    image_files.sort()

    # Extract ids from image filenames (excluding the extension)
    ids = [os.path.splitext(image_file)[0] for image_file in image_files]

    # Create the dataset dictionary
    dataset = {
        "count": len(ids),
        "ids": ids,
        "num_exemplars": len(ids),
        "train_ids": ids,
        "val_ids": []
    }

    # initialize the dictionary to store the metadata
    metadata = {}

    # Write the dataset dictionary to dataset.json
    with open(os.path.join(json_save_dir, 'dataset.json'), 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

    print("dataset.json has been successfully created in", modified_data_path)

    # initialize the lists to store camera_ids, frame_names, and time_ids
    camera_ids = []
    frame_names = []
    time_ids = []

    val_json = {
        "camera_ids": camera_ids,
        "frame_names": frame_names,
        "time_ids": time_ids
    }

    split_dir = os.path.join(json_save_dir, 'splits')
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    with open(os.path.join(split_dir, 'val.json'), 'w') as f:
        json.dump(val_json, f, indent=4)

    # iterate over the image filenames and populate the lists
    for i, filename in enumerate(ids):
        f_name = str(filename)
        camera_ids.append(int(f_name.split('_')[0]))
        
        frame_name = os.path.splitext(filename)[0]
        frame_names.append(frame_name)
        
        #!TODO check if missing time ids works or not
        time_ids.append(int(f_name.split('_')[1]))

        metadata[frame_name] = {
            "appearance_id": f_name.split('_')[1],
            "camera_id": f_name.split('_')[0],
            "warp_id": f_name.split('_')[1]
        }

    # create the dictionary to represent the json structure
    train_json = {
        "camera_ids": camera_ids,
        "frame_names": frame_names,
        "time_ids": time_ids
    }

    # write the dictionary to a json file
    with open(os.path.join(split_dir, 'train.json'), 'w') as f:
        json.dump(train_json, f, indent=4)

    with open(os.path.join(json_save_dir, 'metadata.json'), 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    return

def create_camera_jsons(calibration_file, serial_num_mapping, rgb_input_path, output_jsons_path, calib_method):
    """
    Create camera-related JSON files using calibration data and serial number mappings.

    Parameters:
    - calibration_file (str): Path to a file containing calibration data.
    - serial_num_mapping (dict): Mapping of serial numbers.
    - rgb_input_path (str): Path to input RGB data.
    - output_jsons_path (str): Directory where output JSON files will be saved.
    - calib_method (str): Calibration method used to generate the calibration data.
    """
    
    # Check if output directory exists, if not, create it
    print('should create path')
    if not os.path.exists(output_jsons_path):
        os.makedirs(output_jsons_path)

    # Load the calibration data
    with open(calibration_file, 'r') as f:
        calibration_data = json.load(f)
    rgb_input_path = os.path.join(rgb_input_path, '1x')

    if calib_method == 'MultiCamCalibTool':
        # Create a dictionary to map filenames to their calibration data
        filename_to_calibration = {}
        for frame_data in calibration_data["frames"]:
            filename = os.path.basename(frame_data["file_path"])
            filename_to_calibration[filename] = frame_data

        # Iterate over each image in the rgb_input_path
        for image_name in os.listdir(rgb_input_path):
            if image_name.endswith('.png'):
                # Get the cam_id from the image name
                cam_id = image_name.split('_')[0]

                # Map the cam_id to a serial number using the serial_num_mapping
                serial_number = None
                for key, value in serial_num_mapping.items():
                    if value == int(cam_id):
                        serial_number = key
                        break

                # If we couldn't find a serial number, skip this image
                if serial_number is None:
                    continue

                # Deduce the expected filename from the serial number and frame index
                frame_idx = image_name.split('_')[1].split('.')[0]  # Get frame index from the image name
                expected_filename = f"ts_100_{serial_number}_w1920_h1200.jpg"  # This assumes a specific naming pattern. Adjust if necessary.

                frame_data_new = None
                # Fetch the corresponding calibration data
                for frame_data in calibration_data["frames"]:
                    if expected_filename == frame_data['file_path']:
                        #frame_calibration = filename_to_calibration.get(expected_filename, None)
                        frame_data_new = {}
                        frame_data_new["fl_x"] = frame_data["fl_x"]
                        frame_data_new["fl_y"] = frame_data["fl_y"]
                        frame_data_new["w"] = frame_data["w"]
                        frame_data_new["h"] = frame_data["h"]
                        frame_data_new["cx"] = frame_data["cx"]
                        frame_data_new["cy"] = frame_data["cy"]
                        frame_data_new["k1"] = frame_data["k1"]
                        frame_data_new["k2"] = frame_data["k2"]
                        frame_data_new["k3"] = frame_data["k3"]
                        frame_data_new["k4"] = frame_data["k4"]
                        frame_data_new["p1"] = frame_data["p1"]
                        frame_data_new["p2"] = frame_data["p2"]
                        frame_data_new["transform_matrix"] = frame_data["transform_matrix"]
                        frame_data_new["camera_model"] = frame_data["camera_model"]

                # If we couldn't find calibration data, skip this image
                if frame_data_new is None:
                    continue

                transformed_calibration = {
                    "transform_matrix": frame_data_new["transform_matrix"], # 4x4 matrix
                    "fl_x": frame_data_new["fl_x"], # float
                    "fl_y": frame_data_new["fl_y"], # float
                    "c_x": frame_data_new["cx"], # float
                    "c_y": frame_data_new["cy"], # float
                    "h": frame_data_new["h"], # int
                    "w": frame_data_new["w"], # int
                    "k1": frame_data_new["k1"], # float
                    "k2": frame_data_new["k2"], # float
                    "k3": frame_data_new["k3"], # float
                    "k4": frame_data_new["k4"], # float
                    "p1": frame_data_new["p1"], # float
                    "p2": frame_data_new["p2"], # float
                    "camera_model": frame_data_new["camera_model"] # string
                }
                # Create the new JSON filename
                json_filename = f"{cam_id}_{frame_idx}.json"

                # Write the calibration data for this frame to the new file
                with open(os.path.join(output_jsons_path, json_filename), 'w') as f:
                    json.dump(transformed_calibration, f, indent=4)

    if calib_method == 'Colmap':
        # Iterate over each image in the rgb_input_path
        for image_name in os.listdir(rgb_input_path):
            if image_name.endswith('.png'):
                # Get the cam_id from the image name
                cam_id = image_name.split('_')[0]
                frame_idx = image_name.split('_')[1].split('.')[0]  # Get frame index from the image name
                # fname_json should be in format images/00001.json
                fname_json = f"images/frame_{int(cam_id)+1:05}.jpg"
                frame_data_new = None
                # Fetch the corresponding calibration data
                for frame_data in calibration_data["frames"]:
                    if fname_json == frame_data['file_path']:
                        # frame_calibration = filename_to_calibration.get(fname_json, None)
                        frame_data_new = {}
                        frame_data_new["fl_x"] = calibration_data["fl_x"]
                        frame_data_new["fl_y"] = calibration_data["fl_y"]
                        frame_data_new["w"] = calibration_data["w"]
                        frame_data_new["h"] = calibration_data["h"]
                        frame_data_new["cx"] = calibration_data["cx"]
                        frame_data_new["cy"] = calibration_data["cy"]
                        frame_data_new["k1"] = calibration_data["k1"]
                        frame_data_new["k2"] = calibration_data["k2"]
                        frame_data_new["p1"] = calibration_data["p1"]
                        frame_data_new["p2"] = calibration_data["p2"]
                        frame_data_new["transform_matrix"] = frame_data["transform_matrix"]
                        frame_data_new["camera_model"] = calibration_data["camera_model"]

                # If we couldn't find calibration data, skip this image
                if frame_data_new is None:
                    continue

                transformed_calibration = {
                    "transform_matrix": frame_data_new["transform_matrix"],  # 4x4 matrix
                    "fl_x": frame_data_new["fl_x"],  # float
                    "fl_y": frame_data_new["fl_y"],  # float
                    "c_x": frame_data_new["cx"],  # float
                    "c_y": frame_data_new["cy"],  # float
                    "h": frame_data_new["h"],  # int
                    "w": frame_data_new["w"],  # int
                    "k1": frame_data_new["k1"],  # float
                    "k2": frame_data_new["k2"],  # float
                    "p1": frame_data_new["p1"],  # float
                    "p2": frame_data_new["p2"],  # float
                    "camera_model": frame_data_new["camera_model"]  # string
                }
                # Create the new JSON filename
                json_filename = f"{cam_id}_{frame_idx}.json"

                # Write the calibration data for this frame to the new file
                with open(os.path.join(output_jsons_path, json_filename), 'w') as f:
                    json.dump(transformed_calibration, f, indent=4)


def main():
    #serial_num_map = rename_images(image_dir=DATA_DIR_IN, output_dir=DATA_DIR_OUT)

    #create_dataset_jsons(DATA_DIR_OUT, JSON_DIR_OUT)

    with open(os.path.join(DATA_DIR_OUT, 'serial_number_mapping.json'), 'r') as f:
        serial_num_map = json.load(f)

    cam_json_path = os.path.join(JSON_DIR_OUT, 'camera')
    create_camera_jsons(CALIB_JSON_DIR, serial_num_map, DATA_DIR_OUT, cam_json_path, CALIB_METHOD)


if __name__ == "__main__":
    main()
