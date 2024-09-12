import glob
import json

from flask import Flask, jsonify, render_template, request

import main

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chatbot.html")


@app.route("/answer", methods=["GET"])
def answer():
    query = request.args.get("query")

    if not query:
        return jsonify({"error": "The param 'query' is required."}), 400

    result = main.query(query)

    return jsonify({"response": result["result"]})


@app.route("/get-files", methods=["GET"])
def getFiles():
    folder_path = "documents/input_documents"
    input_files_path = glob.glob(f"{folder_path}/*.pdf")

    return jsonify({"response": input_files_path})


@app.route("/get-questions", methods=["GET"])
def getQuestions():
    with open("documents/questions.json", "r") as f:
        data = json.load(f)
    return jsonify(data)


if __name__ == "__main__":
    app.run(debug=True)
