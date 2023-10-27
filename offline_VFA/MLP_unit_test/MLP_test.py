import pickle
import joblib
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

model = MLPRegressor(hidden_layer_sizes=(50, 50),  # 隐藏层的神经元个数
                     activation='relu',
                     solver='adam',
                     alpha=0.0001,
                     max_iter=1000,
                     random_state=42,
                     # early_stopping=True,  # 是否提前停止训练
                     validation_fraction=0.1,  # 20%作为验证集
                     n_iter_no_change=300,
                     verbose=2,
                     tol=1e-20,
                     )

if __name__ == '__main__':

    nn_var_list = ['time', 'veh_load', 'des_inv']
    for i in range(1, 26):
        nn_var_list.append(f'veh_loc_{i}')
    for i in range(1, 26):
        nn_var_list.append(f'num_self_{i}')
    for i in range(1, 26):
        nn_var_list.append(f'num_oppo_{i}')
    for i in range(1, 26):
        nn_var_list.append(f'orders_till_sim_end_{i}')
    for i in range(1, 26):
        nn_var_list.append(f'bikes_s_arr_till_sim_end{i}')
    for i in range(1, 26):
        nn_var_list.append(f'bikes_c_arr_till_sim_end{i}')

    with open(f"data/nn_state_value_GLA_test.pkl", 'rb') as f:
        state_value_list = pickle.load(f)

    X = np.zeros(shape=(len(state_value_list), len(nn_var_list)))
    Y = np.zeros(shape=(len(state_value_list),))
    count = 0
    for i in range(len(state_value_list)):
        X[i, :] = np.array([state_value_list[i][0][key] for key in nn_var_list]).reshape(1, -1)
        Y[i] = state_value_list[i][1]
        count += 1
        if count % 1000 == 0:
            print(f'count: {count}')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print(
        f'X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}, Y_train.shape: {Y_train.shape}, Y_test.shape: {Y_test.shape}')

    model.fit(X_train, Y_train)

    # 可视化损失函数
    plt.figure()
    plt.plot(model.loss_curve_)
    plt.xlabel("iters")
    plt.ylabel(model.loss)
    plt.show()

    # 对测试集上进行预测
    pre_y_mlp = model.predict(X_test)
    print("mean absolute error:", metrics.mean_absolute_error(Y_test, pre_y_mlp))
    print("mean squared error:", metrics.mean_squared_error(Y_test, pre_y_mlp))

    # 输出在测试集上的R^2
    print("在训练集上的R^2:", model.score(X_train, Y_train))
    print("在测试集上的R^2:", model.score(X_test, Y_test))

    # save model
    joblib.dump(model, 'model/nn_state_value_GLA_test.pkl')
