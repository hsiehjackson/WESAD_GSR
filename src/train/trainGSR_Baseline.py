import csv
import random
from sys import argv
import warnings  
from xgb_utils_base import *
import scipy.stats as stats
from multiprocessing import set_start_method
from sklearn.decomposition import PCA
import pickle as pk
"======setting======" 
threshold = 1 #different feature number
participant_list = np.array([2,3,4,5,6,7,8,9,10,11,13,14,15,16,17])
fold = len(participant_list)
sensor = 'chest'
#sensor = 'wrist'

"======setting======"

def data_shuffle(X, Y):
	pair = list(zip(X,Y))
	random.shuffle(pair)
	X, Y = zip(*pair)
	return np.array(X), np.array(Y)

def feature_clean(data, feat_name, part_name):
	for column in range(data.shape[1],0,-1):
		for n, element in enumerate(data[:,column-1]):
			if str(element) == 'nan' or str(element)=='inf':
				data = np.delete(data,column-1,1)
				feat_name = np.delete(feat_name,column-1)
				break
	data = data.astype('float')
	for column in range(data.shape[1],0,-1):
		for participant in participant_list:
			where = np.where(part_name==participant)[0]
			if np.std(data[where[0]:where[-1]+1, column-1])<= 1e-15:
				data = np.delete(data,column-1,1)
				feat_name = np.delete(feat_name,column-1)
				break
	return data, feat_name

def readfile(filename,task):
	file = open(filename,'r')
	name_all = []
	label_all_list = []
	data_all_list = []
	data_all_dict = {}
	label_all_dict = {}
	alllist = list(csv.reader(file))
	for n, row in enumerate(alllist[1:]):
		name_all.append(int(row[0].split('_')[0]))
		label_all_list.append(float(row[-1]))
		data_all_list.append(row[1:-1])
	file.close()
	feat_name = np.array(alllist[0][1:-1])
	label_all_list = np.array(label_all_list).astype('int')
	data_all_list = np.array(data_all_list).astype('float')
	data_all_list, feat_name = feature_clean(data_all_list[:,task[0]:task[1]], feat_name[task[0]:task[1]], name_all)
	for i, name in enumerate(name_all):
		if name not in data_all_dict:
			data_all_dict[name] = [data_all_list[i]]
			label_all_dict[name] = [label_all_list[i]]
		else:
			data_all_dict[name].append(data_all_list[i])
			label_all_dict[name].append(label_all_list[i])
		
	for i in sorted(list(data_all_dict.keys())):
		mean = np.mean(np.array(data_all_dict[i]),axis=0)
		std = np.std(np.array(data_all_dict[i]),axis=0)
		data_all_dict[i] = (data_all_dict[i] - mean)/std

	return feat_name, label_all_dict ,data_all_dict


def writefile(filename, accuracy,write):
	writefile = open(filename,write)
	writefile.write('feature_size, train_acc, train_f1, val_acc, val_f1\n')
	for i in range(len(accuracy)):
		writefile.write(','.join(repr(accuracy[i][j]) for j in range(len(accuracy[i]))))
		writefile.write('\n')
	writefile.close()
 
def writefeat(filename,feat,vote,write):
	writefile = open(filename,write)
	writefile.write(','.join(str(feat[i]) for i in range(len(feat))))
	writefile.write('\n')
	writefile.write(','.join(str(vote[i]) for i in range(len(vote))))
	writefile.write('\n')
	writefile.close()

def main():
	os.makedirs("gsr_result_{}/3_class_baseline/acc/".format(sensor), exist_ok=True)
	os.makedirs("gsr_result_{}/3_class_baseline/val/".format(sensor), exist_ok=True)
	trainTask([0,14],'ada.csv',classifier='AD')
	trainTask([0,14],'svm.csv',classifier='SVM')
	trainTask([0,14],'kNN.csv',classifier='kNN')
	trainTask([0,14],'DT.csv',classifier='DT')
	trainTask([0,14],'RF.csv',classifier='RF')


def trainTask(task,outfile,classifier):
	print(outfile,'...')
	feature = {}
	name, label_all, data_all = readfile(infile,task)
	train_acc = np.zeros((threshold,fold))
	train_f1 = np.zeros((threshold,fold))
	val_f1 = np.zeros((threshold,fold))
	val_acc = np.zeros((threshold,fold))
	feature_size = np.zeros((threshold,fold))
	target_all = []
	predict_all = []
	"======15 fold validation======"
	for num, val in enumerate(participant_list):
		print('val subject ', val)
		"===========train and val==========="
		X_val = np.array(data_all[val])
		Y_val = np.array(label_all[val])
		for n, part in enumerate(np.delete(participant_list,num)):
			if n == 0:
				X_train = data_all[part].tolist()
				Y_train = label_all[part]
			else: 
				X_train += data_all[part].tolist()
				Y_train += label_all[part]
		X_train = np.array(X_train)
		Y_train = np.array(Y_train)
		print(X_train.shape, X_val.shape)
		"===========normalize==========="
		mean = np.mean(X_train,axis=0)
		std = np.std(X_train,axis=0)
		X_train = (X_train-mean)/std
		X_val = (X_val-mean)/std
		"==========shuffle=========="
		
		X_train, Y_train = data_shuffle(X_train,Y_train)
		X_val, Y_val = data_shuffle(X_val, Y_val)
		"===========train==========="
		tn_acc, tn_f1, v_acc, v_f1, feat_size, feature, predict, target = train(X_train, Y_train, X_val, Y_val, 
																		  val, threshold , name, outfile[:-4], feature,
																		  classifier)
		predict_all.append(predict)
		target_all.append(target)
		train_acc[:,num] = tn_acc
		train_f1[:,num] = tn_f1
		val_acc[:,num] = v_acc
		val_f1[:,num] = v_f1
		feature_size[:,num] = feat_size
	"===========getresult==========="
	folder = 'gsr_result_{}/3_class_baseline/val/'.format(sensor)+outfile[:-4]+'/'
	if not os.path.exists(folder):
		os.mkdir(folder)
	pk.dump(target_all,open(folder+'target.pkl','wb'))
	pk.dump(predict_all,open(folder+'predict.pkl','wb'))
	train_acc = np.sum(train_acc,axis=1)/(fold)
	train_f1 = np.sum(train_f1,axis=1)/(fold)
	val_acc = np.sum(val_acc,axis=1)/(fold)
	val_f1 = np.sum(val_f1,axis=1)/(fold)
	feature_size = np.sum(feature_size,axis=1)/(fold)
	print("===============================")
	print('feature_size = '+str(feature_size))
	print('train accuracy = '+str(train_acc))
	print('train f1 score = '+str(train_f1))
	print('test accuracy = '+str(val_acc))
	print('test f1 score = '+str(val_f1))
	accuracy = np.array([feature_size, train_acc, train_f1, val_acc, val_f1]).T
	writefile('gsr_result_{}/3_class_baseline/acc/'.format(sensor)+outfile,accuracy,'w')
	
if __name__ == '__main__':
	warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)  
	infile = argv[1]
	main()

