from flask import Flask

app = Flask(__name__, static_folder='')

@app.route('/')
def index():
    with open('templates/index.html', encoding="UTF-8") as f:
        text = f.read()

    return text

if __name__ == "__main__":
    app.run(debug=False)
