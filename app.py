from flask import Flask, request, jsonify, send_from_directory
app = Flask(__name__)
import NMS
@app.route('/nms', methods=['GET', 'POST'])
def add_message():
    content = request.json
    return jsonify(NMS.process_json(content))

@app.route('/')
def send_js():
    return send_from_directory('webdemo', "demo.html")

if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True, port=5000)