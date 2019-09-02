
import flask
from keras.models import model_from_json
import numpy as np
import tensorflow as tf
from PIL import Image

#code which helps initialize our server
app = flask.Flask(__name__)
def init():
    global model, graph
    json_file = open("/Users/deepakchoudhary/Desktop/Work/Kaggle/jbm-crack/model.json",'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.summary()
    model.load_weights("/Users/deepakchoudhary/Desktop/Work/Kaggle/jbm-crack/model.h5")
    graph = tf.get_default_graph()
    
def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('imgPath'))
    return parameters
def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

# API for prediction
@app.route("/predict", methods=["GET"])
def predict():
    nameOfTheCharacter = flask.request.args.get('name')
    
    parameters = getParameters()
    img = Image.open(parameters[0])
    img = img.resize((256, 256))
    img_np = np.array(img.getdata()).astype("float64")/255
    img_np = np.resize(img_np, (1, img.size[0], img.size[1], 3))
    with graph.as_default():
        raw_prediction = model.predict(img_np)[0][0]
    if raw_prediction > 0.5:
        prediction = 'Healthy'
    else:
        prediction = 'Defected'
    
    #prediction = "alive"
    return sendResponse({nameOfTheCharacter: prediction})   

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server...""please wait until server has fully started"))
    init()
    app.run(threaded=True)
    
    
  