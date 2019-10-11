import torch
from PIL import Image
from torchvision import transforms as T
import numpy as np

def input_x(image_path):
    '''
    参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
    '''
    x = Image.open(image_path).convert('RGB')
    w, h = x.size
    if h < 512:
        x = T.Resize((512, int(512 * (512 / h))))(x)
    if w < 512:
        x = T.Resize((int(512 * (512 / w)), 512))(x)
    x = T.CenterCrop((448, 448))(x)
    x = T.ToTensor()(x)
    x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
    x=x.reshape(1,3,448,448).cuda()
    #print(x)
    return x



model=torch.load(r'F:\AI\flyai_food\model\model.pkl')
model.eval()  #必须有，否则模型输出都是100


x=input_x('baby_back_ribs.jpg')
y=model(x)
print(y.max(1)[1])