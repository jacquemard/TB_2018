import os

def create_trainval_file(input_path, ext, output_file):
     # creating the trainval text file
    trainval_file = open(output_file, "w")

    for _, _, files in os.walk(input_path):
        for f in files:
            if f.endswith(ext):
                # Adding the image to trainvals
                trainval_file.write(f.split('.')[-2] + "\n")
    trainval_file.close()
