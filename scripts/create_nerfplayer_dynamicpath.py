import json

# Load the JSON data from the file
with open('/home/hamit/Downloads/camera_path (34).json', 'r') as file:
    data = json.load(file)

num_frames = len(data['camera_path'])

# Calculate the interval between each frame
delta = 1.0 / (num_frames - 1)

for i in range(num_frames):
    data['camera_path'][i]['render_time'] = delta * i

# Save the modified data back to the file
with open('render_nerfplayer.json', 'w') as file:
    json.dump(data, file, indent=4)

print("Modified and saved render_times in render_nerfplayer.json")
