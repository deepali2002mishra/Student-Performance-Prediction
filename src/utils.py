import os

def create_directories():
    directories = ["outputs/visualizations"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
