import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import csv

def cluster(data):

    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # One-hot encode categorical variables
    data = pd.concat([data, pd.get_dummies(data["gender"], prefix="gender")], axis=1)
    data = pd.concat([data, pd.get_dummies(data["disease"], prefix="disease")], axis=1)

    # Remove original categorical variables
    data = data.drop(columns=["gender", "disease"])

    # 클러스터링 데이터 따로 만들기
    data_clustering = data[['user_id','height', 'weight', 'age', 'fitness_level', 'gender_male', 'gender_female']]

    data_clustering_f = data_clustering[(data_clustering['gender_female'] == 1)]
    data_clustering_m = data_clustering[(data_clustering['gender_male'] == 1)]
    
    # 데이터 전처리
    def scaling(data_clustering):
        data_clustering = data_clustering.drop(['gender_female', 'gender_male'], axis=1)
        data_clustering = data_clustering.reset_index(drop=True)
        data_processing = data_clustering[['height', 'weight', 'fitness_level']]

        scaler = MinMaxScaler()
        data_scale = scaler.fit_transform(data_processing)
        scale_df = pd.DataFrame(data_scale)
        scale_df.columns = ['height', 'weight', 'fitness_level']

        k_means_optimum = KMeans(n_clusters=3, init='k-means++', random_state=42)
        y = k_means_optimum.fit_predict(scale_df)

        scale_df['cluster'] = y 

        scale_df1 = scale_df[scale_df.cluster==0]
        scale_df2 = scale_df[scale_df.cluster==1]
        scale_df3 = scale_df[scale_df.cluster==2]

        kplot = plt.axes(projection='3d')
        xline = np.linspace(0, 1, 1000)
        yline = np.linspace(0, 1, 1000)
        zline = np.linspace(0, 1, 1000)
        kplot.plot3D(xline, yline, zline, 'black')

        # Data for three-dimensional scattered points
        kplot.scatter3D(scale_df1.height, scale_df1.weight, scale_df1.fitness_level, c='red', label='Cluster 1')
        kplot.scatter3D(scale_df2.height, scale_df2.weight, scale_df2.fitness_level, c='green', label='Cluster 2')
        kplot.scatter3D(scale_df3.height, scale_df3.weight, scale_df3.fitness_level, c='blue', label='Cluster 3')
        plt.legend()
        plt.title("Kmeans")

        return y
    
    data_f_y = scaling(data_clustering_f)
    data_m_y = scaling(data_clustering_m)

    # 이제 군집된 데이터들 가지고 클러스터 넣어주기
    data2 = data[['user_id','height', 'weight', 'age', 'fitness_level', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'gender_male', 'gender_female']]
    
    data2_f = data2[(data2['gender_female'] == 1)]
    data2_f['cluster'] = data_f_y 
    data2_f = data2_f.reset_index(drop=True)
    
    data2_m = data2[(data2['gender_male'] == 1)]
    data2_m['cluster'] = data_m_y 
    data2_m = data2_m.reset_index(drop=True)
    
    dat_all = pd.concat([data2_f, data2_m])
    
    return data2_f, data2_m, dat_all
