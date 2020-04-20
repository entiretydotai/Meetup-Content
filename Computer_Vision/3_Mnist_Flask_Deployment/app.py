#! /usr/bin/env python
from flask import Flask, render_template, request, Response
import numpy as np
from binascii import a2b_base64
import imageio
from PIL import Image
import io
import time
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

global model_states, nb_epoch #to have access later
model_states = ['Not Trained']
nb_epoch=5


app = Flask(__name__)

model =None
@app.route('/')
def to_train():
  return render_template("Home.html")


@app.route("/loadmodel/", methods=['GET'])
def load():
  global model
  model = get_model()
  print("model loaded!")
  return render_template("Home.html")


def get_model():
  global model
  class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 3, 1)
        self.conv2 = nn.Conv2d(20, 50, 3, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2) #13
        x = F.relu(self.conv2(x)) #11
        x = F.max_pool2d(x, 2, 2) #5*5*50
        x = x.view(-1, 5*5*50)    #flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
  checkpoint = torch.load("mnist_cnn.pt")
  model = NN()
  model.load_state_dict(checkpoint)
  print("model loaded")
  return model
#page where you draw the number
@app.route('/index/', methods=['GET','POST'])
def index():
    prediction='?'
    if request.method == 'POST':

        dataURL = request.get_data()
        drawURL_clean = dataURL[22:]
        binary_data=a2b_base64(drawURL_clean)
        img = Image.open(io.BytesIO(binary_data))
        img.thumbnail((28,28))
        img.save("data_img/draw.png")
    return render_template('index.html', prediction=prediction)

#display prediction
@app.route('/result/')
def result():
    time.sleep(0.2)
    img = Image.open('data_img/draw.png')
    print(img)
    img = np.array(img)/255
    img = np.around(img, decimals = 4) 
    img = np.dot(img[...,:4],[0, 0, 0, 1])
    img = torch.Tensor(img)
    print(img)
    img = torch.unsqueeze(img , 0)
    img = torch.unsqueeze(img , 0)
    prediction = inference(img)
    print(prediction)
    return render_template("index.html",prediction=prediction)

def inference(img):
  model = get_model()
  output = model(img)
  output = torch.exp(output)
  top_prob,top_class=output.topk(1,dim=1)
  return top_class.item()

if __name__ == "__main__":
    app.run(debug=True, threaded=True)