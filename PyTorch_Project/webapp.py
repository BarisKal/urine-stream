import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, jsonify, request, render_template
from PIL import Image
from models.complexcnn_model import ComplexCNN
from models.basiccnn_model import BasicCNN

app = Flask(__name__)

# Modelling Task
device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
model = ComplexCNN()
model.load_state_dict(torch.load(
    './models/weights/complex_cnn_best_weights.pt', map_location=device))
model.eval()

image_size = 96
class_names = ['Bad', 'Good']


def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.31652900903431447, 0.266250765902844, 0.21060654775159457],
                             [0.1562390761865127, 0.14919033862737696, 0.14389230815030807])
    ])

    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

# Treat the web process


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':

        if 'photo' not in request.files:
            print('NOT PHOTO')
            return redirect(request.url)

        results = []
        for f in request.files.getlist('photo'):
            if f and f.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_bytes = f.read()
                tensor = transform_image(image_bytes=img_bytes)
                outputs = model.forward(tensor)
                _, prediction = torch.max(outputs, 1)

                if prediction.item() == 1:
                    results.append('Image: ' + f.filename + ' = Good')
                    print('Image: ' + f.filename + ' = Good')
                else:
                    results.append('Image: ' + f.filename + ' = Bad')
                    print('Image: ' + f.filename + ' = Bad')
            else:
                print('Wrong file extension detected!')

        return render_template('index.html', res=results)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
