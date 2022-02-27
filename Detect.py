
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

# importing dependencies related to image transformations
import torchvision
from torchvision import transforms
from PIL import Image

# importing dependencies related to data loading
from torchvision import datasets
from torch.utils.data import DataLoader


def detect_face(img):
    face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray)

    faces = face_clsfr.detectMultiScale(gray, 1.3, 3)
    print(f'Number of faces found = {len(faces)}')
    if len(faces) == 0:
        return None

    x, y, w, h = 0, 0, 0, 0
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), thickness=2)

    face_img = img[y:y+w, x:x+w]
    # plt.imshow(face_img)
    return face_img


def detect_mask(face_img):
    if face_img.all() == None:
        return "No mask detected because face found is 0."
    model = torch.load("model.pth")
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])])
    transform = transforms.ToTensor()
    face_img = transform(face_img)
    face_img = torch.stack([face_img])
    model.eval()
    result = model(face_img)
    label = 1
    _, predicted = torch.max(result.data, 1)
    if predicted == label:
        return "You have a Mask, You can Enter the Institute"
    else:
        return"Please wear a mask to enter the Institute"
