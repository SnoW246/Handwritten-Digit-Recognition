# 		Author: Adrian Golias
# Adapted from: http://flask.pocoo.org/
# 				http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

import flask as fl
from flask import render_template, g, request 

app = fl.Flask(__name__)

@app.route("/")

def name():
	return render_template("Index.html")
	
if __name__ == "__main__":
	app.run()