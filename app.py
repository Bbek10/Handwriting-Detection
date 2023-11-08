from flask import Flask, render_template, request
import os
from inferenceModel import ImageToWordModel




app = Flask(__name__)

BASE_PATH = os.getcwd()
UPLOAD_PATH = os.path.join(BASE_PATH, "static/upload/")


@app.route("/select", methods=["POST", "GET"])
def select():
    if request.method == "POST":
        upload_file = request.files['image']
        filename = upload_file.filename
        path_save = os.path.join(UPLOAD_PATH, filename)
        upload_file.save(path_save)
        text = ImageToWordModel.detect(path_save,filename)

        return render_template('select.html',upload=True,upload_image=filename,text=text)

    return render_template("select.html",upload=False)



@app.route("/")
def hello_world():
    # return "<p>Hello, World!</p>"
    return render_template("try.html")


@app.route("/make")
def works():
    return "<p> </p>"


@app.route("/about")
def about():
    return "<p>Hello, World!</p>"


if __name__ == "__main__":
    app.run(debug=True)
