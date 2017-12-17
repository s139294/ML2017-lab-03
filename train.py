#coding: utf-8
import os
from PIL import Image #Python Imaging Library
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from feature import NPDFeature
from ensemble import AdaBoostClassifier

def to_gray_resize(path,x=None,y=None):
    for file in os.listdir(path):
        file_path=os.path.join(path,file)
        label=None
        if path=='datasets/original/face/':
            label=[1]
        elif path=='datasets/original/nonface/':
            label=[-1]
        image=Image.open(file_path).convert('L').resize((24,24)) #打开图片并转换成灰度图
        if (x is None) & (y is None):#不能使用==：==是element-wise的
            x=np.array([NPDFeature(np.asarray(image)).extract()])
            y=np.array([label])
        else:
            #把是脸的和非脸的图片都拼接起来
            x=np.vstack((x,NPDFeature(np.asarray(image)).extract()))#与np.concat([x1, x2], axis=0)等价
            y=np.vstack((y,label))
    return x,y

def preprocess():
    x,y=to_gray_resize('datasets/original/face/')
    x,y=to_gray_resize('datasets/original/nonface/',x,y)

    #write binary
    with open('datasets/features/feature', 'wb') as file:
        pickle.dump(x, file)
    with open('datasets/features/label', 'wb') as file:
        pickle.dump(y, file)
    print(x.shape,y.shape)

if __name__ == "__main__":
    print('loading data...')
    # preprocess()
    with open('datasets/features/feature', 'rb') as file:
        x = pickle.load(file)
    with open('datasets/features/label', 'rb') as file:
        y = pickle.load(file)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    print('start training...')
    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 3),5)
    ada_clf.fit(X_train, y_train, X_test, y_test)
    ada_clf.plotting()

    print('Writing report...')
    ada_clf.get_report(X_test, y_test)