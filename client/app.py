from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

class RESTClassifier(Resource):
    def post(self):
        json_data = request.get_json(force=True)
        return {'hello': 'world'}

api.add_resource(RESTClassifier, '/predict')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')