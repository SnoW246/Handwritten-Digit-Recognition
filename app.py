# 		Author: Adrian Golias
# Adapted from: http://flask.pocoo.org/
# 				http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# Importing flask libraries
import flask as fl
from flask import render_template, g, request, redirect, url_for
# Importing other libraries
import os
import re
import base64
import keras as kr
import numpy as np
from PIL import Image
from io import StringIO
from scipy.misc import imread, imresize

app = fl.Flask(__name__)

# Web application template function, which renders index 
# file and runs it on port: 5000 (localhost:5000)
# Root for the application
@app.route("/")
def name():
	return app.send_static_file("index.html")

# Route for image uploading
@app.route("/upload", methods=["POST"])
def uploadImage():
	data = request.get_data()
	img = re.search(b"base64,(.*)", data).group(1)
	with open("./images/canvasImg.png","wb") as fh:
		fh.write(base64.b64decode(img))
		
	convertImg = imread("./images/canvasImg.png", mode ='L')
	convertImg = np.invert(convertImg)
	convertImg = imresize(convertImg, (28,28))

	predict = np.ndarray.flatten(np.array(convertImg)).reshape(1, 28, 28).astype('float32')
	predict = predict / 255
	prediction = newPredict(predict)
	
	# returns the predicted number to the webapp
	return prediction

def newPredict(f):
	# Load previously saved model
	model = kr.models.load_model(".model.h5")
	# Make a new prediction with the model
	newPrediction = model.predict(f)
	# Return a string representation of the data in an array
	response = np.array_str(np.argmax(newPrediction))
	return response	
	
if __name__ == "__main__":
	# run our app on localhost:5000
	app.run(host='127.0.0.1', port=5000, debug=True)
	
###################################
#APP_ROOT_DIR = os.path.dirname(os.path.abspath("__file__"))
#print(APP_ROOT_DIR)

#UPLOAD_FOLDER = "./uploads"
#ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])

# Loading trained model that was generated by train.py
#model = kr.models.load_model("model.h5")

#@app.route("/hook", methods=["POST"])
#def get_image():
 #   image_b64 = request.values["imageBase64"]
 #   image_data = re.sub("^data:image/.+;base64,", "", image_b64).decode("base64")
  #  image_PIL = Image.open(cStringIO.StringIO(image_b64))
  #  image_np = np.array(image_PIL)
  #  print ("Image received: {}".format(image_np.shape))
  #  return ("")

# Web application template function, which renders index 
# file and runs it on port: 5000 (localhost:5000)
#@app.route("/")
#def name():
	#return render_template("index.html")
#	return app.send_static_file("index.html")

# Upload function, which allows uploading of files to
# rendered web application
#@app.route("/upload", methods=["POST"])
#def upload_file():
#	target = os.path.join(APP_ROOT_DIR, "uploads/")
#	print(target)
#	
#	for file in request.files.getlist("file"):
#		print(file)
#		filename = file.filename
#		destination ="/".join({target,filename})
#		print(destination)
#		file.save(destination)
#		
#		return str("File Uploaded Successfully")
#		
#@app.route("/uploadFile", methods = ["POST"])
#def uploaded():
#    init()
 #   imageParser(request.get_data())
#
 #   img = imread("./uploads/canvasImg.png", mode="L")
  #  img = np.invert(img)
  #  img = imresize(img,(28,28))
  #  newImg = np.ndarray.flatten(np.array(img)).reshape(1, 784).astype("float32")
  #  newImg /= 255

#    print(newImg)
#    prediction = model.predict(newImg.astype("int"), batch_size=512)
#    print(prediction)