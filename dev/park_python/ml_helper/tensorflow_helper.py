import os


def create_trainval_file(path, ext, output_file):
     # creating the trainval text file
    trainval_file = open(output_file, "w")

    for root, _, files in os.walk(dataset_path):
        for f in files:
            if f.endswith(ext):
                # Adding the image to trainvals
                trainval_file.write(file.split('.')[-2] + "\n")
    trainval_file.close()
