import matplotlib
matplotlib.use('Agg')
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load
df = pd.read_csv('housing.txt')
print(df.shape); print(df.isnull().sum())

# 2. Feature engineering
df['rooms_per_household'] = df['total_rooms']/df['households']
df['bedrooms_per_room'] = df['total_bedrooms']/df['total_rooms']
df['population_per_household'] = df['population']/df['households']
df['income_x_rooms'] = df['median_income']*df['rooms_per_household']
df['income_x_age'] = df['median_income']*df['housing_median_age']
df['log_median_income'] = np.log1p(df['median_income'])

# 3. Split + impute + heatmap
X = df.drop(columns='median_house_value'); y = df['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
num_cols = X.select_dtypes('number').columns.tolist(); cat_cols = ['ocean_proximity']

Xc = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X_train[num_cols]), columns=num_cols)
plt.figure(figsize=(11,8))
sns.heatmap(pd.concat([Xc, y_train.reset_index(drop=True)], axis=1).corr(),
            annot=True, fmt='.2f', cmap='coolwarm', annot_kws={'size':7})
plt.title('Correlation heatmap'); plt.tight_layout()
plt.savefig('plot1_heatmap.png', dpi=120); plt.close()

# 4. VIF (multicollinearity)
def vif(d):
    d = d.copy(); d.insert(0, 'const', 1.0); out = []
    for i, c in enumerate(d.columns[1:], 1):
        others = [j for j in range(d.shape[1]) if j != i]
        r2 = LinearRegression().fit(d.iloc[:, others], d.iloc[:, i]).score(d.iloc[:, others], d.iloc[:, i])
        out.append((c, 1/(1-r2) if r2 < 1 else np.inf))
    return pd.DataFrame(out, columns=['feature','VIF']).sort_values('VIF', ascending=False)
print(vif(Xc))

# 5. Preprocessing
pre = ColumnTransformer([
    ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)])

# 6. OLS, Ridge, Lasso with GridSearchCV
models = {
 'OLS':   GridSearchCV(Pipeline([('pre',pre),('m',LinearRegression())]), {}, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1),
 'Ridge': GridSearchCV(Pipeline([('pre',pre),('m',Ridge())]), {'m__alpha':[0.01,0.1,1,10,100]}, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1),
 'Lasso': GridSearchCV(Pipeline([('pre',pre),('m',Lasso(max_iter=5000))]), {'m__alpha':[0.1,1,10,100]}, cv=3, scoring='neg_root_mean_squared_error', n_jobs=-1),
}
rows, preds = [], {}
for n, g in models.items():
    print('Fitting', n, '...', flush=True)
    g.fit(X_train, y_train); p = g.predict(X_test); preds[n] = p
    rows.append({'Model':n, 'Best alpha':g.best_params_.get('m__alpha','-'),
                 'RMSE':np.sqrt(mean_squared_error(y_test,p)),
                 'MAE':mean_absolute_error(y_test,p), 'R2':r2_score(y_test,p)})
results = pd.DataFrame(rows).set_index('Model'); print(results.round(4))

# 7. Coefficients
names = models['OLS'].best_estimator_['pre'].get_feature_names_out()
coef = pd.DataFrame({n: models[n].best_estimator_['m'].coef_ for n in models}, index=names)
print('Lasso zeroed:', list(coef.index[coef['Lasso']==0]))
coef.plot(kind='barh', figsize=(10,9)); plt.title('Coefficients'); plt.tight_layout()
plt.savefig('plot2_coefficients.png', dpi=120); plt.close()

# 8. Predicted vs actual
fig, ax = plt.subplots(1, 3, figsize=(16,5), sharex=True, sharey=True)
for a, (n, p) in zip(ax, preds.items()):
    a.scatter(y_test, p, s=5, alpha=.3); a.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    a.set_title(f'{n} (R²={r2_score(y_test,p):.3f})'); a.set_xlabel('Actual')
ax[0].set_ylabel('Predicted')
plt.savefig('plot3_pred_vs_actual.png', dpi=120, bbox_inches='tight'); plt.close()

# 9. Residual analysis
fig, ax = plt.subplots(2, 3, figsize=(16,8))
for i, (n, p) in enumerate(preds.items()):
    r = y_test - p
    sns.histplot(r, kde=True, ax=ax[0,i]); ax[0,i].set_title(f'{n} residual distribution')
    ax[1,i].scatter(p, r, s=5, alpha=.3); ax[1,i].axhline(0, color='r')
    ax[1,i].set_title(f'{n} residuals vs predicted'); ax[1,i].set_xlabel('Predicted')
plt.tight_layout()
plt.savefig('plot4_residuals.png', dpi=120); plt.close()

print('Done! Plots saved in this folder.')