# -*- coding: utf-8 -*
'''
实现模型的调用
'''
import os
from flyai.dataset import Dataset

from model import Model

data = Dataset()
model = Model(data)
p = model.predict(image_path=os.path.join('images', 'macaroni_and_cheese/2985770.jpg'))
print(p)