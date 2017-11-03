import matplotlib
matplotlib.use("Agg")
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as k
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


class miniVGGnet:
	def build(width, height, depth, classes):
		model=Sequential()
		inputShape=(height, width, depth)
		chanDim=-1

		if k.image_data_format()=="channel_first":
			inputShape=(depth, height, width)
			chanDim=1

		#First layer of miniVGGNet
		model.add(Conv2D(32,(3,3), padding="same", input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(32, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))

		#2nd layer of VGGnet

		model.add(Conv2D(64, (3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64,(3,3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2,2)))
		model.add(Dropout(0.25))


        #first set of FC->relu
		model.add(Flatten())
		model.add(Dense(512)) #it will have the 512 nodes
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))
		model.add(Dense(classes))
		model.add(Activation('relu'))
		return model


#constructing the argumet parse
ap=argparse.ArgumentParser()
ap.add_argument("-o","--output", required=True, help="path to loss/accurcay plot")
args=vars(ap.parse_args())

print("[Info]: Loading datasets..")
((trainX, trainY),(testX, testY))=cifar10.load_data()
trainX=trainX.astype("float")/255.0
testX=testX.astype("float")/255.0

lb=LabelBinarizer()
trainY=lb.fit_transform(trainY)
testY=lb.transform(testY)

labelName=["airplane", "automobile", "bird", "cat", "deer","dog", "frog", "horse", "ship", "truck"]

print ("[Info] compiling models..")
opt=SGD(lr=0.01, decay=0.01/40, momentum=0.9, nesterov=True)
model=miniVGGnet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

#train the network

print ("[Info]:training network...")
H=model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=64, epochs=2, verbose=1)

#evaluating networks

print("[Info]: evaluating networks..")
predictions=model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
			predictions.argmax(axis=1), target_names=labelName))


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="training lass")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

