from flask import Flask, request, jsonify
app = Flask(__name__)
import NMS
@app.route('/nms', methods=['GET', 'POST'])
def add_message():
    content = request.json
    return jsonify(NMS.process_json(content))

if __name__ == '__main__':
    app.run(host= '0.0.0.0', debug=True, port=5000)