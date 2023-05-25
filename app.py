from flask import Flask, render_template, request
from pegasus import pegasus_test

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "POST":
        example_text = request.form["rawtext"]
        summary_final, example_text, len_example_text, len_summary_final = pegasus_test(
            example_text)
    return render_template("summary.html", summary=summary_final, original_txt=example_text, len_original_txt=len_example_text, len_summary=len_summary_final)


if __name__ == "__main__":
    app.run(debug=True)
