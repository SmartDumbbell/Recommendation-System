import sys
import pandas as pd
from flask import Flask, request, jsonify
import csv
import json

sys.path.append('../machine-learning/') 
import machine_learning.RecommendationSys_Test2 as recommand

app = Flask(__name__)

@app.route('/learning', methods=['POST'])
def learning():
    try:

        pd.DataFrame(request.get_json()).to_csv("data.csv")

        # JSON 데이터를 CSV 파일로 변환
        csv_file = pd.read_csv("data.csv")
        

        # CSV 파일을 cluster() 함수에 전달하여 클러스터링 수행
        result_data_f, result_data_m, result_all = recommand.cluster(csv_file)

        result_all_2 = result_all[['user_id', 'gender_male', 'cluster']]
        result_all_2.rename(columns={'gender_male':'gender'}) #0이면 여자 1이면 남자

        cluster_m = result_data_m.groupby('cluster').mean()
        contents_result_m = cluster_m[['c1', 'c2', 'c3', 'c4','c5','c6']]


        cluster_f = result_data_f.groupby('cluster').mean()
        contents_result_f = cluster_f[['c1', 'c2', 'c3', 'c4','c5','c6']]

        combined_data = pd.concat([contents_result_f, contents_result_m], keys=['female', 'male'])
        combined_data.reset_index(inplace=True)
        combined_data.rename(columns={'level_0': 'gender'}, inplace=True)

        # 'user_id'를 기준으로 오름차순으로 정렬
        result_all_2_sorted = result_all_2.sort_values(by='user_id')

        # 딕셔너리로 변환
        result_all_2_sorted_dict = result_all_2_sorted.to_dict(orient='records')
        combined_data_dict = combined_data.to_dict(orient='records')        

        result_json = {
            "result_all_2": result_all_2_sorted_dict,
            "combined_data": combined_data_dict
        }

        return jsonify(result_json), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
