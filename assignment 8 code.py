

import pandas as pd
import numpy as np
from scipy.stats import t

class RegressionModel(object):
    def __init__(self, x, y, create_intercept=True, regression_type="OLS"):
        self.x = pd.DataFrame(x)
        self.y = y if isinstance(y, pd.Series) else pd.Series(y)
        self.create_intercept = create_intercept
        self.regression_type = regression_type
        self.results = {}

        if self.create_intercept:
                self.add_intercept()

    def add_intercept(self):
        if "intercept" not in self.x.columns:
            self.x['intercept'] = 1
    def ols_regression(self):
        if self.regression_type == "OLS":                                #key is variable
            X = np.array(self.x)
            Y = np.array(self.y).reshape(-1,1)
            n, k = X.shape                                               #df parameters
            df = n - k
            beta_coef = np.linalg.inv(X.T @ X) @ X.T @ Y               #coefficient
            s_2 = ((Y.T @ Y) - (Y.T @ X @ np.linalg.inv(X.T @ X) @ X.T @ Y)) / df #unbiased variance matrix
            cov_B = s_2 * np.linalg.inv(X.T @ X)                         #covariance of b
            variance = np.diag(cov_B)                                    #variance
            standardError = np.sqrt(variance)                            #sqrt of variance
            t_stat = beta_coef.flatten() / standardError.flatten()       #t stat
            p_val = t.sf(t_stat, df)                                     #p value

            for num, val in enumerate(self.x.columns):
                self.results[val] = {
                    "coefficient": float(beta_coef[num].item()),
                    "standard_error": float(standardError[num].item()),
                    "t_stat": float(t_stat[num].item()),
                    "p_value": float(p_val[num].item())
                }

            return self.results
    def summary(self):
        if not self.results:
            self.ols_regression()

        var_name = []
        coef_value = []
        standardError = []
        t_stat = []
        p_value = []

        for key, value in self.results.items():
            var_name.append(key)
            coef_value.append(value['coefficient'])
            standardError.append(value['standard_error'])
            t_stat.append(value['t_stat'])
            p_value.append(value['p_value'])


        table_summary = pd.DataFrame({
            "Variable name": var_name,
            "coefficient value": coef_value,
            "standard error": standardError,
            "t-statistic": t_stat,
            "p-value": p_value
            })

        print(table_summary.to_string(index = False))
        return table_summary