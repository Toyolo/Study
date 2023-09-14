import os
import requests
import zipfile
from pathlib import Path

#setup path to data folder
data_path = Path('data/')
image_path = data_path/'pizza_steak_sushi'

#if the iumage folder doesn't exist, download and prepare the data
if image_path.is_dir():
    print(f'{image_path} already exists')
else:
    print(f'{image_path} does not exist, downloading files...')
    image_path.mkdir(parents=True, exist_ok=True)

#download the data to the image_path
with open('data/pizza_steak_sushi.zip', 'wb') as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

#unzip the data
with zipfile.ZipFile('data/pizza_steak_sushi.zip', 'r') as zip_ref:
    zip_ref.extractall(image_path)
    print("Data unzipped!")

#remove the zip file
os.remove(data_path/'pizza_steak_sushi.zip')
