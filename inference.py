import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import torch.nn as nn
import segmentation_models_pytorch as smp
import tqdm
import skimage


root = Path("./")
test_path= sorted(list(root.glob("test_dataset_test/*")))
df = pd.DataFrame()
df['test_path'] = test_path


def build_model1():
    model = smp.FPN(
        encoder_name='timm-regnety_032',
        encoder_weights="imagenet",
        in_channels = 3,
        classes = 3,
        activation=None,
    )
    model.to(torch.device('cpu'))
    return model

def build_model2():
    model = smp.FPN(
        encoder_name='efficientnet-b5',
        encoder_weights="imagenet",
        in_channels = 3,
        classes = 3,
        activation=None,
    )
    model.to(torch.device('cpu'))
    return model


def load_model1(path):
    model = build_model1()
    model.load_state_dict(torch.load(path,  map_location=torch.device('cpu') ))
    model.eval()
    return model

def load_model2(path):
    model = build_model2()
    model.load_state_dict(torch.load(path,  map_location=torch.device('cpu') ))
    model.eval()
    return model


for i in tqdm.tqdm(range(0, len(df))):
    input_img = cv2.imread(test_path[i].as_posix(), cv2.IMREAD_UNCHANGED)
    input_shape = input_img.shape[0], input_img.shape[1]
    img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    img = img.astype('float32')
    img /= 255.0
    img = np.transpose(img, (2, 1, 0))
    img = np.expand_dims(img, 0)
    img = torch.tensor(img)
    img = img.to(torch.device('cpu'), dtype=torch.float)

    preds = []
    model1 = load_model1(f"regnety_032.pt") #metric 82.22
    model2 = load_model2(f"effb5.pt") #metric 81.9

    models = [model1,model2]
    for model in models:
        with torch.no_grad():
            pred = model(img)
            pred_sigmoid = (nn.Sigmoid()(pred)).double()
        preds.append(pred_sigmoid)
    preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()
    preds = (preds > 0.5).double()

    pred = preds[0,].permute((2, 1, 0)).numpy()
    pred[:, :, 0][pred[:, :, 0] == 1] = 6
    pred[:, :, 1][pred[:, :, 1] == 1] = 7
    pred[:, :, 2][pred[:, :, 2] == 1] = 10
    res = pred[:, :, 0] + pred[:, :, 1] + pred[:, :, 2]

    if 13 in np.unique(res):
        res = np.where(res == 13, 7 , res)
        print('наложение основной колеи на колею неосновную')

    if 16 in np.unique(res):
        res = np.where(res == 16, 6, res)
        print('наложение поезда на неосновную колею')

    if 17 in np.unique(res):
        res = np.where(res == 17, 7, res)
        print('наложение поезда на основную колею')

    resized_mask = skimage.transform.resize(res,
                                            input_shape,
                                            mode='edge',
                                            anti_aliasing=False,
                                            anti_aliasing_sigma=None,
                                            preserve_range=True,
                                            order=0)

    cv2.imwrite(f'./out/{str(test_path[i]).split("/")[-1]}', resized_mask)
