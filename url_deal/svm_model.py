import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

if __name__ == '__main__':
    data = pd.read_csv('../data/final_data_v1.csv')
    X = data[['len', 'url_count', 'evil_char', 'evil_word', 'shang']]
    # print(X.describe())
    y = data['label']
    # train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.4, random_state=0)
    clf = LinearSVC(dual=False)
    # clf.fit(train_x, train_y)
    # joblib.dump(clf, '../model/xss-svm-model.m')
    # y_pred = clf.predict(test_x)
    # print(classification_report(test_y, y_pred))
    #
    # print(clf.get_params())

    params = {'C': [0.1, 1, 10], 'loss': ['hinge', 'squared_hinge'],
              'tol': [1e-5, 1e-4, 1e-3]}
    grid = GridSearchCV(estimator=clf, param_grid=params, n_jobs=8, verbose=2, cv=3)
    grid.fit(X, y)
    print("best param" + str(grid.best_params_))
    print("best score" + str(grid.best_score_))
