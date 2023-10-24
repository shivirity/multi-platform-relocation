import pickle
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# parameters
model_name = 'phi_6'
train_rep = 4000

df = pd.read_csv(f'data/linear_regression_test_{model_name}_25_{train_rep}.csv', index_col=0)

x_str = ' + '.join(df.columns[1:-1])
formula_str = 'y ~ ' + x_str

model = ols(formula=formula_str, data=df).fit()
print(model.summary())
result_dict = model.params.to_dict()
result_dict['const'] = result_dict['Intercept']
del result_dict['Intercept']
with open(f'data/result_dict_{model_name}_25_{train_rep}.pkl', 'wb') as f:
    pickle.dump(result_dict, f)

y_pred, y_test = model.fittedvalues, df['y']
mse = np.mean((y_pred - y_test) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_pred - y_test))
print(f'mse: {mse}, rmse: {rmse}, mae: {mae}')
