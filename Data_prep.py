import pandas as pd
import numpy as np
import random
import scipy.stats as stats
import scipy.signal as sig
import matplotlib.pyplot as plt


def split_test_train(data, id_column, threshold):
    user_ids = data[id_column].unique()
    user_ids = user_ids.astype(int)
    shape = user_ids.shape
    no_of_train_ids = np.rint(shape[0] * threshold)
    user_train = []
    train_shape = len(user_train)
    while train_shape < no_of_train_ids:
        random_id = random.randint(user_ids.min(), user_ids.max())
        if random_id in user_train:
            continue
        elif random_id in user_ids:
            user_train.append(random_id)
            train_shape = len(user_train)
    user_test = np.setdiff1d(user_ids,user_train)
    column_names = data.keys()
    train = pd.DataFrame(columns = column_names)
    test = pd.DataFrame(columns = column_names)
    grouped = data.groupby(data[id_column])
    for id_train in user_train:
        user_data = grouped.get_group(id_train)
        train = train.append(user_data)
    
    for id_test in user_test:
        user_data = grouped.get_group(id_test)
        test = test.append(user_data)
    
    return test, train


# def segment(data, period, step, label_column, columns, id_column):
#     user_ids = data[id_column].unique()
#     N_FEATURES = columns.shape[0]
#     segments = []
#     labels = []
#     grouped = data.groupby(data[id_column])
#     for id_i in user_ids:
#         user_data = grouped.get_group(id_i)
# #         user_data = user_data.to_numpy()
#         for i in range(0, len(user_data) - period, step):
#             features = user_data[columns]
#             features1 = features[i: i + period]
#             # del features
#             features2 = features1.to_numpy()
#             # del features1
#             # Retrieve the most often used label in this segment
#             label = stats.mode(user_data[label_column][i: i + period])[0][0]
#             segments.append(features2)
#             # del features2
#             labels.append(label)
#
#     # Bring the segments into a better shape
#     reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, period, N_FEATURES)
#     labels = np.asarray(labels)
#
#     return reshaped_segments, labels

def segment(data, period, step, label_column, columns, id_column):
    user_ids = data[id_column].unique()
    N_FEATURES = columns.shape[0]
    segments = []
    labels = []
    grouped = data.groupby(data[id_column])
    for id_i in user_ids:
        user_data = grouped.get_group(id_i)
        for i in range(0, len(user_data) - period, step):
            features = user_data[columns].values[i: i + period]
            # Retrieve the most often used label in this segment
            label = stats.mode(user_data[label_column][i: i + period])[0][0]
            segments.append(features)
            labels.append(label)

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, period, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def segment_own(data, period, step, label_column, columns, id_column):
    # user_ids = data[id_column].unique()
    N_FEATURES = columns.shape[0]
    segments = []
    labels = []
    # grouped = data.groupby(data[id_column])
    # del data
    # for id_i in user_ids:
    #     user_data = grouped.get_group(id_i)
    #     print(id_i)
        # user_data = user_data.to_numpy()
    for i in range(0, len(data) - period, step):
        # print(i)
        features = data[columns].values[0: period]
        # print(i)
        # Retrieve the most often used label in this segment
        label = stats.mode(data[label_column][0: period])[0][0]
        data = data.drop(data.index[0: period])
        # print(data)
        segments.append(features)
        print(i)
        features = None
        labels.append(label)
        # del user_data
    data = None

    # Bring the segments into a better shape
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, period, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels


def change_segments(data):
    out_data = np.empty((0,84))
    print(data.shape)
    for j in range(0,data.shape[0]):
        features = []
        features = np.array(features)
        for i in range(0,data.shape[2]):
            hist, bins = np.histogram(data[j,:,i], bins=10)
#             hist = hist.T
            features = np.concatenate((features, hist))
            avg = np.average(data[j,:,i])
            features = np.concatenate((features, [avg]))
            peaks, properties = sig.find_peaks(data[j,:,i])
            peaks = np.array(peaks)
            differences = np.diff(peaks)
            difference = np.average(differences)
            features = np.concatenate((features, [difference]))
            standard_dev = np.std(data[j,:,i])
            features = np.concatenate((features, [standard_dev]))
            mad = np.mean(np.absolute(data[j,:,i] - np.mean(data[j,:,i])))
            features = np.concatenate((features, [mad]))
            
        features = np.array(features)
        features = np.reshape(features, (1, features.shape[0]))
        out_data = np.vstack((out_data,features))
        
    return out_data
            
            
def plot(values, peaks):
    y = range(0,200)
    plt.plot(values)
    plt.scatter(peaks,values[peaks])
    plt.ylabel(y)
    plt.show()




