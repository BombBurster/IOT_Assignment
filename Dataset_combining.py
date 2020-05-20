import pandas as pd
import glob

PATH_acc_mobile = "./raw/phone/accel/"
PATH_gyro_mobile = "./raw/phone/gyro/"
PATH_acc_watch = "./raw/watch/accel/"
PATH_gyro_watch = "./raw/watch/gyro/"
PATH_acc_luca = "./merging_self_generated_data/LucaAcelerometer/"
PATH_gyro_luca = "./merging_self_generated_data/LucaGyro/"
PATH_acc_caroline = "./merging_self_generated_data/CarolineAcelerometer/"
PATH_gyro_caroline = "./merging_self_generated_data/CarolineGyro/"
column_names_acc_mobile = ['user-id','activity', 'timestamp', 'x-axis_acc_mobile', 'y-axis_acc_mobile', 'z-axis_acc_mobile']
column_names_gyro_mobile = ['user-id','activity', 'timestamp', 'x-axis_gyro_mobile', 'y-axis_gyro_mobile', 'z-axis_gyro_mobile']
column_names_acc_watch = ['user-id','activity', 'timestamp', 'x-axis_acc_watch', 'y-axis_acc_watch', 'z-axis_acc_watch']
column_names_gyro_watch = ['user-id','activity', 'timestamp', 'x-axis_gyro_watch', 'y-axis_gyro_watch', 'z-axis_gyro_watch']
column_names_acc_luca = ['user-id','activity', 'time', 'x-axis_acc', 'y-axis_acc', 'z-axis_acc', 'timestamp']
column_names_gyro_luca = ['user-id','activity', 'time', 'x-axis_gyro', 'y-axis_gyro', 'z-axis_gyro', 'timestamp']
column_names_acc_caroline = ['user-id','activity', 'time', 'x-axis_acc', 'y-axis_acc', 'z-axis_acc', 'timestamp']
column_names_gyro_caroline = ['user-id','activity', 'time', 'x-axis_gyro', 'y-axis_gyro', 'z-axis_gyro', 'timestamp']


def loadfromFilePath(path, column_names, own):
    files = glob.glob(path)
    dataset = pd.DataFrame(columns = column_names)
    for file in files:
        if own == 0:
            df = pd.read_csv(file, header=None, names=column_names, skiprows=[30], skipfooter=30, engine='python')
        else:
#             print(file)
            df = pd.read_csv(file, header=0, names=column_names, skiprows=[30], skipfooter=30, engine='python')
#             print(df)
        dataset = dataset.append(df)
    if own == 0:
        dataset[column_names[5]].replace(regex=True, inplace=True, to_replace=r';', value=r'')
        dataset[column_names[5]] = dataset[column_names[5]].astype(float)
    dataset.dropna(axis=0, how='any', inplace=True)
    return dataset


acc_mobile = loadfromFilePath(PATH_acc_mobile+"*.txt", column_names_acc_mobile,0)
gyro_mobile = loadfromFilePath(PATH_gyro_mobile+"*.txt", column_names_gyro_mobile,0)
acc_watch = loadfromFilePath(PATH_acc_watch+"*.txt", column_names_acc_watch,0)
gyro_watch = loadfromFilePath(PATH_gyro_watch+"*.txt", column_names_gyro_watch,0)
acc_luca = loadfromFilePath(PATH_acc_luca+"*.csv", column_names_acc_luca,1)
gyro_luca = loadfromFilePath(PATH_gyro_luca+"*.csv", column_names_gyro_luca,1)
acc_caroline = loadfromFilePath(PATH_acc_caroline+"*.csv", column_names_acc_caroline,1)
gyro_caroline = loadfromFilePath(PATH_gyro_caroline+"*.csv", column_names_gyro_caroline,1)
acc_luca = acc_luca.drop(columns=['time'])
gyro_luca = gyro_luca.drop(columns=['time'])
acc_caroline = acc_caroline.drop(columns=['time'])
gyro_caroline = gyro_caroline.drop(columns=['time'])
# print(acc_mobile)
# print(gyro_mobile)
# print(acc_watch)
# print(gyro_watch)
# print(acc_luca)
# print(gyro_luca)


dataset_merge = pd.merge(acc_mobile, gyro_mobile, on=['user-id', 'activity', 'timestamp'])
dataset_merge_2 = pd.merge(acc_watch, gyro_watch, on=['user-id', 'activity', 'timestamp'])
dataset_merge_3 = pd.merge(acc_luca, gyro_luca, on=['user-id', 'activity', 'timestamp'])
dataset_merge_4 = pd.merge(acc_caroline, gyro_caroline, on=['user-id', 'activity', 'timestamp'])
dataset_merge_3 = pd.concat([dataset_merge_3, dataset_merge_4], axis=0)
# print(dataset_merge_3)
# print(dataset_merge_2)


# showing that merging the watch data and the phone data is not compatible - therefore train seperately
merge_1 = pd.merge(acc_mobile, acc_watch, on=['user-id', 'activity', 'timestamp'])
# print(merge_1)


dataset_merge.activity = pd.Categorical(dataset_merge.activity)
dataset_merge_2.activity = pd.Categorical(dataset_merge_2.activity)
dataset_merge_3.activity = pd.Categorical(dataset_merge_3.activity)
dataset_merge_3['user-id'] = pd.Categorical(dataset_merge_3['user-id'])
dataset_merge.activity = dataset_merge.activity.cat.codes
dataset_merge_3.activity = dataset_merge_3.activity.cat.codes
dataset_merge_3['user-id'] = dataset_merge_3['user-id'].cat.codes
# print(dataset_merge_3)


dataset_merge_3 = dataset_merge_3.iloc[::20, :]
print(dataset_merge_3)

dataset_merge.to_csv('phone.csv', index=False)
dataset_merge_2.to_csv('watch.csv', index=False)
dataset_merge_3.to_csv('our_own.csv', index=False)






