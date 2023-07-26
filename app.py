import sys
from flask import Flask, request, jsonify
sys.path.append('../machine-learning/')  # machine-learning 폴더의 절대 경로를 입력하세요.
import recommand  # machine-learning 폴더에 있는 recommand.py 모듈을 가져옵니다.

app = Flask(__name__)

@app.route('/learning', methods=['POST'])
def learning():
    try:
        data = request.get_json()  
        processed_data = recommand.preprocess_data(data)  
        cluster_data = recommand.get_cluster_data(processed_data)  # recommand 모듈의 get_cluster_data 함수 사용

        return jsonify(cluster_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
