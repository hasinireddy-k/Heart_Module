from flask import Flask, render_template, request
import os
from reconstruction.progression import compare_progression
from reconstruction.heart_3d import generate_3d_heart

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/progression", methods=["POST"])
def progression():

    # clear folders
    for file in os.listdir("uploads/t1"):
        os.remove(os.path.join("uploads/t1", file))

    for file in os.listdir("uploads/t2"):
        os.remove(os.path.join("uploads/t2", file))

    t1_files = request.files.getlist("t1_files")
    t2_files = request.files.getlist("t2_files")

    for file in t1_files:
        file.save(os.path.join("uploads/t1", file.filename))

    for file in t2_files:
        file.save(os.path.join("uploads/t2", file.filename))

    fig = compare_progression("uploads/t1", "uploads/t2")
    graph_html = fig.to_html(full_html=False)

    return render_template("index.html", graph_html=graph_html)

@app.route("/upload", methods=["POST"])
def upload():

    for file in os.listdir(UPLOAD_FOLDER):
        os.remove(os.path.join(UPLOAD_FOLDER, file))

    files = request.files.getlist("files")

    for file in files:
        if file.filename != "":
            file.save(os.path.join(UPLOAD_FOLDER, file.filename))

    fig, tumor_percentage, severity = generate_3d_heart(UPLOAD_FOLDER)

    graph_html = fig.to_html(full_html=False)

    return render_template(
        "index.html",
        graph_html=graph_html,
        tumor_percentage=round(tumor_percentage, 2),
        severity=severity
    )

if __name__ == "__main__":
    app.run(debug=True)
