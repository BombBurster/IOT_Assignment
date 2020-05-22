from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
from Data_prep import *
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.feature_selection import f_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

np.set_printoptions(threshold=sys.maxsize)

TIME_PERIODS = 200
STEP_DISTANCE = 200
BEST_PARAM_GRID_SVM = [
            {'C': [5],
             'tol': [0.001],
             'max_iter': [-1],
             'kernel': ['rbf'],
             'degree': [2],
             'gamma': ['auto_deprecated'],
             'decision_function_shape': ['ovr'],
             'class_weight': ['balanced']}
        ]
BEST_PARAM_GRID_RF = [
            {'bootstrap': [False],
             'max_depth': [None],
             'class_weight': ['balanced'],
             'oob_score': [False],
             'max_features': ['auto'],
             'min_samples_leaf': [1],
             'min_samples_split': [2],
             'n_estimators': [1500]}
        ]


def feature_selection_pearson(data, thresh):
    corr = data.corr()
    columns_to_remove = []
    columns = data.keys()
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if abs(corr.iloc[i,j]) >= thresh:
                columns_to_remove.append(j)
    columns_to_remove = np.array(columns_to_remove)
    columns_to_remove = np.unique(columns_to_remove)
    return columns_to_remove

def p_value(x, y, cutoff):
    columns_to_remove = []
    columns = x.keys()
    num_features = len(columns)
    raw_scores, p_values = f_regression(x, y)
    for i in range(num_features):
        if (p_values[i] > cutoff):
            columns_to_remove.append(i)
    columns_to_remove = np.array(columns_to_remove)
    columns_to_remove = np.unique(columns_to_remove)
    return columns_to_remove

def select_dataset():
    print('Dataset Options:\n1. WISDM phone data\n2. WISDM watch data\n3. Our own dataset')
    flag = 0
    flag_own = 0
    exec_mode = input('Select: ')
    if exec_mode == '1':
        flag = 1
        dataset = pd.read_csv('phone.csv')
    elif exec_mode == '2':
        flag = 1
        dataset = pd.read_csv('watch.csv')
        # print(dataset)
    elif exec_mode == '3':
        flag = 1
        flag_own = 1
        dataset = pd.read_csv('our_own.csv')
        # print(dataset)
        # dataset = dataset1.loc[np.random.choice(dataset1.index, 1500000, replace=False)]
        # del dataset1
        # print(dataset)
    else:
        flag = 0
        dataset = []
    return flag, dataset, flag_own

def select_grid_or_norm(dataset, flag_own):
    global BEST_PARAM_GRID_SVM
    global BEST_PARAM_GRID_RF
    print('Dataset Options:\n1. Grid Search SVM\n2. Grid Search Random Forest\n3. Compare different scalers\n4. Compare different feature selection\n5. Run confusion Matrix')
    flag = 1
    exec_mode = input('Select: ')
    if exec_mode == '1':
        flaq = 1
        param_grid = [
            {'C': [5, 3, 2, 1],
             'tol': [0.1, 0.01, 0.001],
             'max_iter': [-1],
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
             'degree': [2],
             'gamma': ['auto_deprecated'],
             'decision_function_shape': ['ovr'],  # one-vs-one/one-vs-rest
             'class_weight': ['balanced']}  # auto balance the weight
        ]
        column_names = dataset.keys()
        if flag_own == 0:
            test, train = split_test_train(dataset, 'user-id', 0.8)
            x_train, y_train = segment(train, TIME_PERIODS, STEP_DISTANCE, 'activity', train.columns.difference(['activity','user-id','timestamp']), 'user-id')
            x_test, y_test = segment(test, TIME_PERIODS, STEP_DISTANCE, 'activity', test.columns.difference(['activity','user-id','timestamp']), 'user-id')
            x_test_n = change_segments(x_test)
            x_train_n = change_segments(x_train)
            print(y_train)
        else:
            x, y = segment(dataset, TIME_PERIODS, STEP_DISTANCE, 'activity', dataset.columns.difference(['activity','user-id','timestamp']), 'activity')
            del dataset
            # print(x)
            x_n = change_segments_own(x)
            del x
            # print(x_n[np.isnan(x_n).any(axis=1)])
            # y = y[~np.isnan(x_n).any(axis=1)]
            # x_n = x_n[~np.isnan(x_n).any(axis=1)]
            x_train_n, x_test_n, y_train, y_test = train_test_split(x_n, y, test_size=0.2)
            del x_n
            # print(x_train_n)
        x_train_n_1 = pd.DataFrame(x_train_n)
        x_test_n_1 = pd.DataFrame(x_test_n)
        remove_columns = p_value(x_train_n_1, y_train, 0.05)
        x_train_n_1 = x_train_n_1.drop(columns=remove_columns)
        x_test_n_1 = x_test_n_1.drop(columns=remove_columns)
        transformer = StandardScaler().fit(x_train_n_1)
        x_train_n = transformer.transform(x_train_n_1)
        x_test_n = transformer.transform(x_test_n_1)
        clf = SVC()
        svm_model = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1, error_score=np.nan, verbose=1) 
        svm_model = svm_model.fit(x_train_n, y_train)
        # print the best hyper-parameters
        print('\nClass weight option:', svm_model.best_estimator_.get_params()['class_weight'])
        print('Best kernel:', svm_model.best_estimator_.get_params()['kernel'])
        print('Best polynomial degree:', svm_model.best_estimator_.get_params()['degree'])
        print('Best gamma:', svm_model.best_estimator_.get_params()['gamma'])
        print('Best C:', svm_model.best_estimator_.get_params()['C'])
        print('Best tolerance:', svm_model.best_estimator_.get_params()['tol'])
        print('Best max number of iterations', svm_model.best_estimator_.get_params()['max_iter'])
        print('Best decision function shape option', svm_model.best_estimator_.get_params()['decision_function_shape'],
              '\n')
        
        # print the accuracy scores
        y_pred = svm_model.predict(x_test_n)
        result = metrics.accuracy_score(y_test, y_pred)
        result_2 = metrics.balanced_accuracy_score(y_test, y_pred)
        f1_svm = metrics.f1_score(y_test, y_pred, average='weighted')
        prec_svm = metrics.precision_score(y_test, y_pred, average='weighted')
        rec_svm = metrics.recall_score(y_test, y_pred, average='weighted')
        print('The accuracy score is: ', result)
        print('The balanced accuracy score is: ', result_2)
        print('The f1-score is: ', f1_svm)
        print('The precision score is: ', prec_svm)
        print('The recall score is: ', rec_svm)
        # save best parameters
        BEST_PARAM_GRID_SVM = [
            {'C': [svm_model.best_estimator_.get_params()['C']],
             'tol': [svm_model.best_estimator_.get_params()['tol']],
             'max_iter': [svm_model.best_estimator_.get_params()['max_iter']],
             'kernel': [svm_model.best_estimator_.get_params()['kernel']],
             'degree': [svm_model.best_estimator_.get_params()['degree']],
             'gamma': [svm_model.best_estimator_.get_params()['gamma']],
             'decision_function_shape': [svm_model.best_estimator_.get_params()['decision_function_shape']],
             'class_weight': [svm_model.best_estimator_.get_params()['class_weight']]}
        ]
    elif exec_mode == '2':
        flaq = 1
        param_grid = [
            {'bootstrap': [False],
             'max_depth': [None],
             'class_weight': ['balanced'],
             'oob_score': [False],
             'max_features': ['auto'],
             'min_samples_leaf': [1], # , 2, 4],
             'min_samples_split': [2], #, 5, 10],
             'n_estimators': [1500, 10000, 20000]}  # 500, 1000,
        ]
        column_names = dataset.keys()
        if flag_own == 0:
            test, train = split_test_train(dataset, 'user-id', 0.8)
            x_train, y_train = segment(train, TIME_PERIODS, STEP_DISTANCE, 'activity',
                                       train.columns.difference(['activity', 'user-id', 'timestamp']), 'user-id')
            x_test, y_test = segment(test, TIME_PERIODS, STEP_DISTANCE, 'activity',
                                     test.columns.difference(['activity', 'user-id', 'timestamp']), 'user-id')
            x_test_n = change_segments(x_test)
            x_train_n = change_segments(x_train)
            print(y_train)
        else:
            x, y = segment(dataset, TIME_PERIODS, STEP_DISTANCE, 'activity',
                           dataset.columns.difference(['activity', 'user-id', 'timestamp']), 'activity')
            del dataset
            # print(x)
            x_n = change_segments_own(x)
            del x
            # print(x_n[np.isnan(x_n).any(axis=1)])
            # y = y[~np.isnan(x_n).any(axis=1)]
            # x_n = x_n[~np.isnan(x_n).any(axis=1)]
            x_train_n, x_test_n, y_train, y_test = train_test_split(x_n, y, test_size=0.2)
            del x_n
            # print(x_train_n)
        x_train_n_1 = pd.DataFrame(x_train_n)
        x_test_n_1 = pd.DataFrame(x_test_n)
        remove_columns = p_value(x_train_n_1, y_train, 0.05)
        x_train_n_1 = x_train_n_1.drop(columns=remove_columns)
        x_test_n_1 = x_test_n_1.drop(columns=remove_columns)
        transformer = StandardScaler().fit(x_train_n_1)
        x_train_n = transformer.transform(x_train_n_1)
        x_test_n = transformer.transform(x_test_n_1)
        clf = RandomForestClassifier()
        rf_model = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=3, error_score=np.nan, verbose=2)
        rf_model = rf_model.fit(x_train_n, y_train)
        # print the best hyper-parameters
        print('\nClass weight option:', rf_model.best_estimator_.get_params()['class_weight'])
        print('Bootstrap option:', rf_model.best_estimator_.get_params()['bootstrap'])
        print('Best maximum tree depth:', rf_model.best_estimator_.get_params()['max_depth'])
        print('Out-of-bag score option:', rf_model.best_estimator_.get_params()['oob_score'])
        print('Best maximum features:', rf_model.best_estimator_.get_params()['max_features'])
        print('Best minimum samples for leaf node:', rf_model.best_estimator_.get_params()['min_samples_leaf'])
        print('Best minimum samples for splitting node', rf_model.best_estimator_.get_params()['min_samples_split'])
        print('Best number of trees', rf_model.best_estimator_.get_params()['n_estimators'],
              '\n')
        
        # print the accuracy scores
        y_pred = rf_model.predict(x_test_n)
        result = metrics.accuracy_score(y_test, y_pred)
        result_2 = metrics.balanced_accuracy_score(y_test, y_pred)
        f1_rf = metrics.f1_score(y_test, y_pred, average='weighted')
        prec_rf = metrics.precision_score(y_test, y_pred, average='weighted')
        rec_rf = metrics.recall_score(y_test, y_pred, average='weighted')
        print('The accuracy score is: ', result)
        print('The balanced accuracy score is: ', result_2)
        print('The f1-score is: ', f1_rf)
        print('The precision score is: ', prec_rf)
        print('The recall score is: ', rec_rf)
        # save best parameters
        BEST_PARAM_GRID_RF = [
            {'bootstrap': [rf_model.best_estimator_.get_params()['bootstrap']],
             'max_depth': [rf_model.best_estimator_.get_params()['max_depth']],
             'class_weight': [rf_model.best_estimator_.get_params()['class_weight']],
             'oob_score': [rf_model.best_estimator_.get_params()['oob_score']],
             'max_features': [rf_model.best_estimator_.get_params()['max_features']],
             'min_samples_leaf': [rf_model.best_estimator_.get_params()['min_samples_leaf']],
             'min_samples_split': [rf_model.best_estimator_.get_params()['min_samples_split']],
             'n_estimators': [rf_model.best_estimator_.get_params()['n_estimators']]}
        ]
    elif exec_mode == '3':
        flag = 1
        print(BEST_PARAM_GRID_SVM)
        print(BEST_PARAM_GRID_RF)
        column_names = dataset.keys()
        if flag_own == 0:
            test, train = split_test_train(dataset, 'user-id', 0.8)
            x_train, y_train = segment(train, TIME_PERIODS, STEP_DISTANCE, 'activity',
                                       train.columns.difference(['activity', 'user-id', 'timestamp']), 'user-id')
            x_test, y_test = segment(test, TIME_PERIODS, STEP_DISTANCE, 'activity',
                                     test.columns.difference(['activity', 'user-id', 'timestamp']), 'user-id')
            x_test_n = change_segments(x_test)
            x_train_n = change_segments(x_train)
            print(y_train)
        else:
            x, y = segment(dataset, TIME_PERIODS, STEP_DISTANCE, 'activity',
                           dataset.columns.difference(['activity', 'user-id', 'timestamp']), 'activity')
            del dataset
            # print(x)
            x_n = change_segments_own(x)
            del x
            # print(x_n[np.isnan(x_n).any(axis=1)])
            # y = y[~np.isnan(x_n).any(axis=1)]
            # x_n = x_n[~np.isnan(x_n).any(axis=1)]
            x_train_n, x_test_n, y_train, y_test = train_test_split(x_n, y, test_size=0.2)
            del x_n
            # print(x_train_n)
        x_train_n_1 = pd.DataFrame(x_train_n)
        x_test_n_1 = pd.DataFrame(x_test_n)
        remove_columns = p_value(x_train_n_1, y_train, 0.05)
        x_train_n_1 = x_train_n_1.drop(columns=remove_columns)
        x_test_n_1 = x_test_n_1.drop(columns=remove_columns)
        array_scalers = [Normalizer(), StandardScaler(), preprocessing.MinMaxScaler()]
        results = []
        for scaler in array_scalers:
            transformer = scaler.fit(x_train_n_1) #MinMax = 0.65%
            x_train_n = transformer.transform(x_train_n_1)
            x_test_n = transformer.transform(x_test_n_1)
            clf = SVC()
            svm_model = GridSearchCV(clf, param_grid=BEST_PARAM_GRID_SVM, cv=10, n_jobs=-1, error_score=np.nan, verbose=1) 
            svm_model = svm_model.fit(x_train_n, y_train)
            y_pred = svm_model.predict(x_test_n)
            acc_svm = metrics.accuracy_score(y_test, y_pred)
            f1_svm = metrics.f1_score(y_test, y_pred, average='weighted')
            prec_svm = metrics.precision_score(y_test, y_pred, average='weighted')
            rec_svm = metrics.recall_score(y_test, y_pred, average='weighted')
            clf = RandomForestClassifier()
            rf_model = GridSearchCV(clf, param_grid=BEST_PARAM_GRID_RF, cv=10, n_jobs=-1, error_score=np.nan, verbose=1) 
            rf_model = rf_model.fit(x_train_n, y_train)
            y_pred = svm_model.predict(x_test_n)
            acc_rf = metrics.accuracy_score(y_test, y_pred)
            f1_rf = metrics.f1_score(y_test, y_pred, average='weighted')
            prec_rf = metrics.precision_score(y_test, y_pred, average='weighted')
            rec_rf = metrics.recall_score(y_test, y_pred, average='weighted')
            results.append([acc_svm, f1_svm, prec_svm, rec_svm, acc_rf, f1_rf, prec_rf, rec_rf])
        print(results)
    elif exec_mode == '4':
        flag = 1
        column_names = dataset.keys()
        print(BEST_PARAM_GRID_SVM)
        print(BEST_PARAM_GRID_RF)
        if flag_own == 0:
            test, train = split_test_train(dataset, 'user-id', 0.8)
            x_train, y_train = segment(train, TIME_PERIODS, STEP_DISTANCE, 'activity',
                                       train.columns.difference(['activity', 'user-id', 'timestamp']), 'user-id')
            x_test, y_test = segment(test, TIME_PERIODS, STEP_DISTANCE, 'activity',
                                     test.columns.difference(['activity', 'user-id', 'timestamp']), 'user-id')
            x_test_n = change_segments(x_test)
            x_train_n = change_segments(x_train)
            print(y_train)
        else:
            x, y = segment(dataset, TIME_PERIODS, STEP_DISTANCE, 'activity',
                           dataset.columns.difference(['activity', 'user-id', 'timestamp']), 'activity')
            del dataset
            # print(x)
            x_n = change_segments_own(x)
            del x
            # print(x_n[np.isnan(x_n).any(axis=1)])
            # y = y[~np.isnan(x_n).any(axis=1)]
            # x_n = x_n[~np.isnan(x_n).any(axis=1)]
            x_train_n, x_test_n, y_train, y_test = train_test_split(x_n, y, test_size=0.2)
            del x_n
            # print(x_train_n)
        array_correlation = ['pearson', 'p-value']
        results = []
        for feature_select in array_correlation:
            if feature_select == 'p-value':
                x_train_n_1 = pd.DataFrame(x_train_n)
                x_test_n_1 = pd.DataFrame(x_test_n)
                remove_columns = p_value(x_train_n_1, y_train, 0.05)
                x_train_n_1 = x_train_n_1.drop(columns=remove_columns)
                x_test_n_1 = x_test_n_1.drop(columns=remove_columns)
            elif feature_select == 'pearson':
                x_train_n_1 = pd.DataFrame(x_train_n)
                x_test_n_1 = pd.DataFrame(x_test_n)
                remove_columns = feature_selection_pearson(x_train_n_1, 0.9)
                x_train_n_1 = x_train_n_1.drop(columns=remove_columns)
                x_test_n_1 = x_test_n_1.drop(columns=remove_columns)
            transformer = StandardScaler().fit(x_train_n_1)
            x_train_n_1 = transformer.transform(x_train_n_1)
            x_test_n_1 = transformer.transform(x_test_n_1)
            clf = SVC()
            svm_model = GridSearchCV(clf, param_grid=BEST_PARAM_GRID_SVM, cv=10, n_jobs=-1, error_score=np.nan, verbose=1) 
            svm_model = svm_model.fit(x_train_n_1, y_train)
            y_pred = svm_model.predict(x_test_n_1)
            acc_svm = metrics.accuracy_score(y_test, y_pred)
            f1_svm = metrics.f1_score(y_test, y_pred, average='weighted')
            prec_svm = metrics.precision_score(y_test, y_pred, average='weighted')
            rec_svm = metrics.recall_score(y_test, y_pred, average='weighted')
            clf = RandomForestClassifier()
            rf_model = GridSearchCV(clf, param_grid=BEST_PARAM_GRID_RF, cv=10, n_jobs=-1, error_score=np.nan, verbose=1) 
            rf_model = rf_model.fit(x_train_n_1, y_train)
            y_pred = svm_model.predict(x_test_n_1)
            acc_rf = metrics.accuracy_score(y_test, y_pred)
            f1_rf = metrics.f1_score(y_test, y_pred, average='weighted')
            prec_rf = metrics.precision_score(y_test, y_pred, average='weighted')
            rec_rf = metrics.recall_score(y_test, y_pred, average='weighted')
            results.append([acc_svm, f1_svm, prec_svm, rec_svm, acc_rf, f1_rf, prec_rf, rec_rf, x_train_n_1.shape[1]])
        print(results)
    elif exec_mode == '5':
        flag = 1
        param_grid = BEST_PARAM_GRID_SVM
        column_names = dataset.keys()
        if flag_own == 0:
            test, train = split_test_train(dataset, 'user-id', 0.8)
            x_train, y_train = segment(train, TIME_PERIODS, STEP_DISTANCE, 'activity',
                                       train.columns.difference(['activity', 'user-id', 'timestamp']), 'user-id')
            x_test, y_test = segment(test, TIME_PERIODS, STEP_DISTANCE, 'activity',
                                     test.columns.difference(['activity', 'user-id', 'timestamp']), 'user-id')
            x_test_n = change_segments(x_test)
            x_train_n = change_segments(x_train)
            print(y_train)
        else:
            x, y = segment(dataset, TIME_PERIODS, STEP_DISTANCE, 'activity',
                           dataset.columns.difference(['activity', 'user-id', 'timestamp']), 'activity')
            # del dataset
            # print(x)
            x_n = change_segments_own(x)
            del x
            # print(x_n[np.isnan(x_n).any(axis=1)])
            # y = y[~np.isnan(x_n).any(axis=1)]
            # x_n = x_n[~np.isnan(x_n).any(axis=1)]
            x_train_n, x_test_n, y_train, y_test = train_test_split(x_n, y, test_size=0.2)
            del x_n
            # print(x_train_n)
        x_train_n_1 = pd.DataFrame(x_train_n)
        x_test_n_1 = pd.DataFrame(x_test_n)
        remove_columns = p_value(x_train_n_1, y_train, 0.05)
        x_train_n_1 = x_train_n_1.drop(columns=remove_columns)
        x_test_n_1 = x_test_n_1.drop(columns=remove_columns)
        transformer = StandardScaler().fit(x_train_n_1)
        x_train_n = transformer.transform(x_train_n_1)
        x_test_n = transformer.transform(x_test_n_1)
        clf = SVC()
        svm_model = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=-1, error_score=np.nan, verbose=1)
        svm_model = svm_model.fit(x_train_n, y_train)
        # print the best hyper-parameters
        print('\nClass weight option:', svm_model.best_estimator_.get_params()['class_weight'])
        print('Best kernel:', svm_model.best_estimator_.get_params()['kernel'])
        print('Best polynomial degree:', svm_model.best_estimator_.get_params()['degree'])
        print('Best gamma:', svm_model.best_estimator_.get_params()['gamma'])
        print('Best C:', svm_model.best_estimator_.get_params()['C'])
        print('Best tolerance:', svm_model.best_estimator_.get_params()['tol'])
        print('Best max number of iterations', svm_model.best_estimator_.get_params()['max_iter'])
        print('Best decision function shape option', svm_model.best_estimator_.get_params()['decision_function_shape'],
              '\n')

        # print the accuracy scores
        y_pred = svm_model.predict(x_test_n)
        result = metrics.accuracy_score(y_test, y_pred)
        result_2 = metrics.balanced_accuracy_score(y_test, y_pred)
        f1_svm = metrics.f1_score(y_test, y_pred, average='weighted')
        prec_svm = metrics.precision_score(y_test, y_pred, average='weighted')
        rec_svm = metrics.recall_score(y_test, y_pred, average='weighted')
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        print('The accuracy score is: ', result)
        print('The balanced accuracy score is: ', result_2)
        print('The f1-score is: ', f1_svm)
        print('The precision score is: ', prec_svm)
        print('The recall score is: ', rec_svm)
        print('Confusion Matrix: ', conf_matrix)
        param_grid = BEST_PARAM_GRID_RF
        column_names = dataset.keys()
        if flag_own == 0:
            test, train = split_test_train(dataset, 'user-id', 0.8)
            x_train, y_train = segment(train, TIME_PERIODS, STEP_DISTANCE, 'activity',
                                       train.columns.difference(['activity', 'user-id', 'timestamp']), 'user-id')
            x_test, y_test = segment(test, TIME_PERIODS, STEP_DISTANCE, 'activity',
                                     test.columns.difference(['activity', 'user-id', 'timestamp']), 'user-id')
            x_test_n = change_segments(x_test)
            x_train_n = change_segments(x_train)
            print(y_train)
        else:
            x, y = segment(dataset, TIME_PERIODS, STEP_DISTANCE, 'activity',
                           dataset.columns.difference(['activity', 'user-id', 'timestamp']), 'activity')
            del dataset
            # print(x)
            x_n = change_segments_own(x)
            del x
            # print(x_n[np.isnan(x_n).any(axis=1)])
            # y = y[~np.isnan(x_n).any(axis=1)]
            # x_n = x_n[~np.isnan(x_n).any(axis=1)]
            x_train_n, x_test_n, y_train, y_test = train_test_split(x_n, y, test_size=0.2)
            del x_n
            # print(x_train_n)
        x_train_n_1 = pd.DataFrame(x_train_n)
        x_test_n_1 = pd.DataFrame(x_test_n)
        remove_columns = p_value(x_train_n_1, y_train, 0.05)
        x_train_n_1 = x_train_n_1.drop(columns=remove_columns)
        x_test_n_1 = x_test_n_1.drop(columns=remove_columns)
        transformer = StandardScaler().fit(x_train_n_1)
        x_train_n = transformer.transform(x_train_n_1)
        x_test_n = transformer.transform(x_test_n_1)
        clf = RandomForestClassifier()
        rf_model = GridSearchCV(clf, param_grid=param_grid, cv=5, n_jobs=3, error_score=np.nan, verbose=1)
        rf_model = rf_model.fit(x_train_n, y_train)
        # print the best hyper-parameters
        print('\nClass weight option:', rf_model.best_estimator_.get_params()['class_weight'])
        print('Bootstrap option:', rf_model.best_estimator_.get_params()['bootstrap'])
        print('Best maximum tree depth:', rf_model.best_estimator_.get_params()['max_depth'])
        print('Out-of-bag score option:', rf_model.best_estimator_.get_params()['oob_score'])
        print('Best maximum features:', rf_model.best_estimator_.get_params()['max_features'])
        print('Best minimum samples for leaf node:', rf_model.best_estimator_.get_params()['min_samples_leaf'])
        print('Best minimum samples for splitting node', rf_model.best_estimator_.get_params()['min_samples_split'])
        print('Best number of trees', rf_model.best_estimator_.get_params()['n_estimators'],
              '\n')

        # print the accuracy scores
        y_pred = rf_model.predict(x_test_n)
        result = metrics.accuracy_score(y_test, y_pred)
        result_2 = metrics.balanced_accuracy_score(y_test, y_pred)
        f1_rf = metrics.f1_score(y_test, y_pred, average='weighted')
        prec_rf = metrics.precision_score(y_test, y_pred, average='weighted')
        rec_rf = metrics.recall_score(y_test, y_pred, average='weighted')
        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        print('The accuracy score is: ', result)
        print('The balanced accuracy score is: ', result_2)
        print('The f1-score is: ', f1_rf)
        print('The precision score is: ', prec_rf)
        print('The recall score is: ', rec_rf)
        print('Confusion Matrix: ', conf_matrix)
    else:
        flag = 0
    return flag

flag, dataset, flag_own = select_dataset()
if flag == 1:
    flag = select_grid_or_norm(dataset, flag_own)
    if flag == 0:
        print('Quitting')
else:
    print('Quitting')




