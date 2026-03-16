# Import required libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error




#======================================================================
# importing dataset

input_file = "Medical Insurance cost prediction.csv"
target = "charges"

# After viewing the dataset specify which columns need type casting
dtype_cat_2_num = []
dtype_str_2_numeric = []
dfype_int_2_cate =[]
date_cols = []

# After EDA decide what columns to drop specify the list
cols_to_remove = []

#================================ upload data

# if input is .csv then
df = pd.read_csv(input_file)
#=============================================================
# rename the columns names by repalcing spaces with underscore and make it lower case
df.columns = df.columns.str.strip().str.lower().str.replace(r"[^\w]+","_", regex=True)
#=================================================================
# view the dataset
df_summary = pd.DataFrame({"variables": df.columns,
                           "data type":df.dtypes,
                           "Missing_count" : df.isnull().sum(),
                           "Missing_percent" : (df.isnull().sum()/len(df))*100,
                           "Unique" : df.nunique(),
                           "count" : df.count()
                          })

with pd.option_context(
    'display.max_columns', None,
    'display.width', None,
    'display.expand_frame_repr', False
):
    print(f"Before removing nulls in Target: \n {df_summary.sort_values("Unique").set_index("variables")}\n")


# Data handling & cleaning
#=======================================================
'''
# if target has null values drop it
df.dropna(subset = [target],inplace =True)
# view the dataset
df_summary = pd.DataFrame({"variables": df.columns,
                           "data type":df.dtypes,
                           "Missing_count" : df.isnull().sum(),
                           "Missing_percent" : (df.isnull().sum()/len(df))*100,
                           "Unique" : df.nunique(),
                           "count" : df.count()
                          })
with pd.option_context(
    'display.max_columns', None,
    'display.width', None,
    'display.expand_frame_repr', False
):
    print(f"After removing nulls in Target: \n {df_summary.sort_values("Unique").set_index("variables")}\n")
'''
#==================================================================
# data type conversion

df[dtype_cat_2_num] = df[dtype_cat_2_num].astype(float)
df[dtype_str_2_numeric] = df[dtype_str_2_numeric].apply(pd.to_numeric)
df[dfype_int_2_cate] = df[dfype_int_2_cate].astype("object")
print(df.dtypes)

#date_cols = [c for c in df.columns if "date" in c]
#df[date_cols] = df[date_cols].apply(pd.to_datetime)

# seperate the datasets as per the type and view the description
df_target = df[target]
df_num_with_target = df.select_dtypes(exclude =["object"])
df_num_wo_target = df_num_with_target.drop(columns = [target]) 

pd.reset_option('display.max_columns')
pd.reset_option('display.width')
pd.reset_option('display.expand_frame_repr')

print(f"Numeric_vaiables_with target \n {df_num_with_target.describe()}\n")

df_cat = df.select_dtypes(include = ["object"])
print(f"Category_description \n {df_cat.describe()}")



cat_cols_list = df_cat.columns
num_cols_with_target = df_num_with_target.columns
num_cols_list_wo_target = df_num_wo_target.columns
cols_to_drop = [col for col in cat_cols_list if df_cat[col].nunique() >20]
final_cats_df = df_cat.drop(columns =cols_to_drop)
final_cats_list = final_cats_df.columns



# Correlation matrix
print(df_num_with_target.corr().reset_index())

# visualize using heatmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize = (12,8))
sns.heatmap(df_num_with_target.corr(), annot =True, cmap ="coolwarm", fmt=".2f")
plt.show()




for col in num_cols_with_target:
    sns.scatterplot(x=df[col], y=df[target])
    plt.show()



# Categorical - Numeric  === box plot, voilin plot, group statistics [Insights: distribution differnce, category influence

for cat_col in final_cats_list:
     for num_col in num_cols_with_target:
           sns.violinplot(x=df[cat_col], y=df[num_col])
           plt.show()

#==============================================================
# functions
#==============================================================

def analyze_numeric_distribution(df, num_cols):

    for col in num_cols:
        
        skew_val = df[col].skew()
        
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        
        # Histogram
        sns.histplot(df[col], kde=True, ax=ax[0])
        ax[0].set_title(f"{col} Distribution\nSkewness = {round(skew_val,2)}")
        
        # Boxplot
        sns.boxplot(x=df[col], ax=ax[1])
        ax[1].set_title(f"{col} Boxplot")
        
        plt.show()
        
num_cols = df_num_with_target.select_dtypes(include='number').columns

analyze_numeric_distribution(df_num_with_target, num_cols)



print(f"Dupicates \n {df.duplicated().sum()}\n")
print(f"Missings \n {df.isnull().sum().sort_values(ascending =False)}\n")
print(f"skewness \n {df_num_wo_target.skew().sort_values(ascending =False)}")



#======================================================================================
# missing value treatment

import pandas as pd
import numpy as np

def dynamic_missing_treatment(df, drop_threshold=0.30):

    df_out = df.copy()
    report = []

    for col in df.columns:

        missing_count = df[col].isna().sum()
        missing_pct = df[col].isna().mean()

        if missing_count == 0:
            action = "No Missing"
            fill_value = None

        elif missing_pct > drop_threshold:
            df_out.drop(columns=col, inplace=True)
            action = "Column Dropped"
            fill_value = None

        else:

            if df[col].dtype in ["int64","float64"]:

                if missing_pct < 0.05:
                    fill_value = df[col].mean()
                    df_out[col].fillna(fill_value, inplace=True)
                    action = "Mean Imputation"

                else:
                    fill_value = df[col].median()
                    df_out[col].fillna(fill_value, inplace=True)
                    action = "Median Imputation"

            else:

                fill_value = df[col].mode()[0]
                df_out[col].fillna(fill_value, inplace=True)
                action = "Mode Imputation"

        report.append({
            "Column": col,
            "Missing Count": missing_count,
            "Missing %": round(missing_pct*100,2),
            "Fill Value": fill_value,
            "Action": action
        })

    report_df = pd.DataFrame(report)

    return df_out, report_df
#===================================================================================================
# Outlier Treatment
#==================================================================================================

def dynamic_outlier_treatment(df, method="iqr", cap=True):

    df_out = df.copy()
    report = []

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    for col in numeric_cols:

        if method == "iqr":

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower) | (df[col] > upper)][col]

            if len(outliers) == 0:
                action = "No Outliers"

            else:

                if cap:
                    df_out[col] = np.where(df[col] > upper, upper, df[col])
                    df_out[col] = np.where(df_out[col] < lower, lower, df_out[col])
                    action = "Outliers Capped (IQR)"

                else:
                    df_out = df_out[(df_out[col] >= lower) & (df_out[col] <= upper)]
                    action = "Outliers Removed"

            report.append({
                "Column": col,
                "Method": "IQR",
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outlier Count": len(outliers),
                "Action": action
            })

        elif method == "zscore":

            mean = df[col].mean()
            std = df[col].std()

            lower = mean - 3 * std
            upper = mean + 3 * std

            z = (df[col] - mean) / std

            outliers = df[abs(z) > 3][col]

            if len(outliers) == 0:
                action = "No Outliers"

            else:

                if cap:
                    df_out[col] = np.where(df[col] > upper, upper, df[col])
                    df_out[col] = np.where(df_out[col] < lower, lower, df_out[col])
                    action = "Outliers Capped (Z-score)"

                else:
                    df_out = df_out[abs(z) <= 3]
                    action = "Outliers Removed"

            report.append({
                "Column": col,
                "Method": "Z-score",
                "Lower Bound": lower,
                "Upper Bound": upper,
                "Outlier Count": len(outliers),
                "Action": action
            })

    report_df = pd.DataFrame(report)

    return df_out, report_df
    
#======================================================================================
# Skewness transformation
#====================================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox

def dynamic_skew_transform(df, skew_threshold=1):

    df_transformed = df.copy()
    report = []

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    for col in numeric_cols:

        skew_val = df[col].skew()

        if abs(skew_val) <= skew_threshold:
            report.append({"Column": col, "Skew": skew_val, "Transformation": "None"})
            continue

        series = df[col].replace([np.inf, -np.inf], np.nan)

        # Case 1 : negative values exist
        if (series < 0).any():

            pt = PowerTransformer(method='yeo-johnson')
            df_transformed[col] = pt.fit_transform(series.values.reshape(-1,1))

            method = "Yeo-Johnson"

        # Case 2 : zero values exist
        elif (series == 0).any():

            df_transformed[col] = np.log1p(series)

            method = "Log1p"

        # Case 3 : strictly positive
        else:

            valid_index = series.dropna().index
            transformed, _ = boxcox(series.loc[valid_index])

            df_transformed.loc[valid_index, col] = transformed

            method = "Box-Cox"

        report.append({
            "Column": col,
            "Skew": skew_val,
            "Transformation": method
        })

    report_df = pd.DataFrame(report)

    return df_transformed, report_df
    
#====================================================================================================
# Encoding
#=================================================================================================

# Encoding if cats exist
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import LabelEncoder
import pandas as pd

def dynamic_encoding(df, onehot_threshold=20):

    df_out = df.copy()
    report = []

    cat_cols = df_out.select_dtypes(include=['object','category']).columns

    for col in cat_cols:

        unique_vals = df_out[col].nunique()

        # Case 1 : Low Cardinality → One Hot Encoding
        if unique_vals < onehot_threshold:

            dummies = pd.get_dummies(
                df_out[col],
                prefix=col,
                drop_first=True
            )

            df_out = pd.concat([df_out, dummies], axis=1)
            df_out.drop(columns=col, inplace=True)

            method = "One Hot Encoding (drop first)"

        # Case 2 : High Cardinality → Label Encoding
        else:

            le = LabelEncoder()
            df_out[col] = le.fit_transform(df_out[col])

            method = "Label Encoding"

        report.append({
            "Column": col,
            "Unique Values": unique_vals,
            "Encoding": method
        })

    report_df = pd.DataFrame(report)

    return df_out, report_df
#================================================================================================

# Scaling 
#================================================================================================
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def dynamic_scaling(df, skew_threshold=1):

    df_scaled = df.copy()
    report = []

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    for col in numeric_cols:

        skew_val = df[col].skew()

        # Detect outliers using IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers_exist = ((df[col] < lower) | (df[col] > upper)).any()

        # Choose scaler dynamically
        if outliers_exist:
            scaler = RobustScaler()
            method = "RobustScaler"

        elif abs(skew_val) > skew_threshold:
            scaler = MinMaxScaler()
            method = "MinMaxScaler"

        else:
            scaler = StandardScaler()
            method = "StandardScaler"

        df_scaled[col] = scaler.fit_transform(df[[col]])

        report.append({
            "Column": col,
            "Skew": skew_val,
            "Outliers": outliers_exist,
            "Scaler": method
        })

    report_df = pd.DataFrame(report)

    return df_scaled, report_df
#===================================================================================================
# now calling and printing the outputs of missing, outlier, skewness, encoding, scaling
#========================================================================================================
X = df.drop(columns=[target])
y = df[target]

df1, missing_report = dynamic_missing_treatment(X)
df2, outlier_report = dynamic_outlier_treatment(df1)
df3, skew_report = dynamic_skew_transform(df2)
df4, encoding_report = dynamic_encoding(df3)
df5, scaling_report = dynamic_scaling(df4)

df5[target] = y.loc[df5.index]



#==================================

with pd.option_context(
    'display.max_columns', None,
    'display.width', None,
    'display.expand_frame_repr', False
):
    print(missing_report)
    print(outlier_report.sort_values("Outlier Count", ascending=False))
    print("Transformation Report")
    print(skew_report)
    print(encoding_report)
    print(scaling_report)

#======================= Outlier Before after comparision 
import matplotlib.pyplot as plt

df1.boxplot()
plt.title("Before Outlier Treatment")
plt.show()

df2.boxplot()
plt.title("After Outlier Treatment")
plt.show()

#================================= 
print("\nBefore Skew")
print(df2.select_dtypes(include='number').skew())

print("\nAfter Skew")
print(df3.select_dtypes(include='number').skew())

#=====================================================
df5.head()
df5.columns
#=====================column filtering / Feature Selection




import pandas as pd

def manual_drop_columns(df, columns_to_drop):

    df_out = df.copy()

    # keep only columns that exist in dataframe
    cols_existing = [col for col in columns_to_drop if col in df_out.columns]

    df_out.drop(columns=cols_existing, inplace=True)

    report = pd.DataFrame({
        "Dropped Columns": cols_existing
    })

    return df_out, report
    
df6, drop_report = manual_drop_columns(df5, cols_to_remove)
print(cols_to_remove)
print(drop_report)




# split independent and dependent variables
X = df6.drop(columns=[target])
y = df6[target]

# split train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.2, random_state = 42)




models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Elastic Net": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42),
    "SVR" : SVR(),
    "KNN" : KNeighborsRegressor(),
      
}

results = []

n = X_test.shape[0]     # number of observations
p = X_test.shape[1]     # number of predictors

for name, model in models.items():

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # R2
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Adjusted R2
    adj_r2 = 1 - ((1 - test_r2) * (n - 1) / (n - p - 1))

    # Error metrics
    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)

    results.append([name, train_r2, test_r2, adj_r2, mse, mae, rmse])

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Train R2",
        "Test R2",
        "Adjusted R2",
        "MSE",
        "MAE",
        "RMSE"
    ]
)

results_df = results_df.sort_values(by="Test R2", ascending=False).round(3)
with pd.option_context(
    'display.max_columns', None,
    'display.width', None,
    'display.expand_frame_repr', False
):
    print(results_df)
    


 
tree_models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    #"Extra Trees": ExtraTreesRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=42)
}

X_train = X_train.apply(lambda x: x.cat.codes if x.dtype.name == "category" else x)
X_test = X_test.apply(lambda x: x.cat.codes if x.dtype.name == "category" else x)

tree_results = []

n = X_test.shape[0]
p = X_test.shape[1]

for name, model in tree_models.items():

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    adj_r2 = 1 - ((1 - test_r2) * (n - 1) / (n - p - 1))

    mse = mean_squared_error(y_test, y_test_pred)
    mae = mean_absolute_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)

    tree_results.append([
        name, train_r2, test_r2, adj_r2, mse, mae, rmse
    ])

tree_results_df = pd.DataFrame(
    tree_results,
    columns=[
        "Model",
        "Train R2",
        "Test R2",
        "Adjusted R2",
        "MSE",
        "MAE",
        "RMSE"
    ]
)

tree_results_df = tree_results_df.sort_values(
    by="Test R2", ascending=False
).round(4)

final_results = pd.concat([results_df, tree_results_df], axis=0)

final_results = final_results.reset_index(drop=True)

final_results = final_results.sort_values(by="Test R2", ascending=False)
results_df = results_df.sort_values(by="Test R2", ascending=False).round(3)

with pd.option_context(
    'display.max_columns', None,
    'display.width', None,
    'display.expand_frame_repr', False
):
    print(final_results)



# hyoer parameter tuning
import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

best_models = {}
results = []

for name, model in models.items():

    print(f"\nRunning RandomizedSearchCV for {name}...")

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist[name],
        n_iter=30,
        cv=5,
        scoring="r2",
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Predictions
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae = mean_absolute_error(y_test, y_test_pred)

    # Overfitting / Underfitting check
    gap = train_r2 - test_r2

    if train_r2 < 0.6 and test_r2 < 0.6:
        status = "Underfitting"
    elif gap > 0.15:
        status = "Overfitting"
    else:
        status = "Normal Fit"

    best_models[name] = best_model

    results.append([
        name,
        search.best_params_,
        train_r2,
        test_r2,
        rmse,
        mae,
        status
    ])

    print("Best Params:", search.best_params_)
    print("Train R2:", train_r2)
    print("Test R2:", test_r2)

    results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Best Parameters",
        "Train R2",
        "Test R2",
        "RMSE",
        "MAE",
        "Fit Status"
    ]
)

results_df = results_df.sort_values("Test R2", ascending=False)

results_df.loc[results_df.index[0], "Fit Status"] = "Best Model"

with pd.option_context(
    'display.max_columns', None,
    'display.width', None,
    'display.expand_frame_repr', False
):
    print("\nModel Performance Leaderboard")
    print(results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = best_models[best_model_name]

print("\nBest Model:", best_model_name)

y_pred = best_model.predict(X_test)

import joblib

joblib.dump(best_model, "model.pkl")




