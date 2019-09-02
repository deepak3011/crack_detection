
import os
from keras.applications.inception_v3 import InceptionV3
from keras import layers
from keras.models import Model
from PIL import Image, ImageEnhance
import numpy as np
import random
from sklearn.model_selection import train_test_split

class DataAugmentation:
    degrees = [90, 180, 270]
    def __call__(self, img):
        img = self.Flip(img)
        img = self.Brightness(img)
        img = self.Contrast(img)
        img = self.Color(img)
        
        return img
        
    
    def check_prob(self):
        prob = random.random()
        if prob > 0.7:
            return True
        else:
            return False
        
    def Flip(self, img):
        
        if self.check_prob():
            img = img.rotate(random.choice(self.degrees))
        return img
    def Brightness(self, img):
        if self.check_prob():
            factor = random.random() + 0.5
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(factor)
        return img

    def Contrast(self, img):
        if self.check_prob():
            factor = random.random() + 0.5
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(factor)
        return img

    def Color(self, img):
        if self.check_prob():
            factor = random.random() + 0.5
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(factor)
        return img
DAobj = DataAugmentation()

            
def LoadImgIntoNumpy(fileList, label):
    X = np.zeros((0, 256, 256, 3))
    for path in fileList:
        img = Image.open(path)
        img = DAobj(img)
        img = img.resize((256, 256))
        img_np = np.array(img.getdata()).astype("float64")/255
        img_np = np.resize(img_np, (1, img.size[0], img.size[1], 3))
        X = np.concatenate((X, img_np))
        
    if label:
        Y = np.ones((X.shape[0], 1))
    else:
        Y = np.zeros((X.shape[0], 1))
    return X, Y

def Generator(positive_fileList, negative_fileList, batch_size):
    
    while 1:
        
        X_healthy, Y_healthy = LoadImgIntoNumpy(random.sample(dataset_healthy, batch_size//2), 1)
        X_defects, Y_defects = LoadImgIntoNumpy(random.sample(dataset_defects, batch_size//2), 0)
        
        X = np.concatenate((X_healthy, X_defects))
        Y = np.concatenate((Y_healthy, Y_defects))
        
        yield X,Y

        

dataset_healthy = []
directory_healthy = "YE358311_Fender_apron/YE358311_Healthy"
for fn in os.listdir(directory_healthy):
    if fn.endswith(".jpg"):
        dataset_healthy.append(os.path.join(directory_healthy, fn))

random.shuffle(dataset_healthy)

dataset_defects = []
directory_defects = "YE358311_Fender_apron/YE358311_defects"
for fn in os.listdir(directory_defects):
    if fn.endswith(".jpg"):
        dataset_defects.append(os.path.join(directory_defects, fn))
        
random.shuffle(dataset_defects)

train_dataset_healthy, val_dataset_healthy = train_test_split(dataset_healthy, test_size = 0.3)
train_dataset_defects, val_dataset_defects = train_test_split(dataset_defects, test_size = 0.3)

inceptionModel = InceptionV3(include_top = False, input_shape = (256, 256, 3))
layer_dict = dict([(layer.name, layer) for layer in inceptionModel.layers])
layer_name = "mixed3"
layer_output = layer_dict[layer_name]

x = layers.GlobalMaxPooling2D()(layer_output.output)
x = layers.Dense(1, activation = "sigmoid")(x)

model = Model(inceptionModel.input, x)
model.summary()
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

model.fit_generator(Generator(train_dataset_healthy, train_dataset_defects, 32), steps_per_epoch = 50, validation_data = Generator(val_dataset_healthy, val_dataset_defects, 32), validation_steps= 5, epochs = 50)

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")