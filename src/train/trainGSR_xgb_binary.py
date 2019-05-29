import csv
import random
from sys import argv
import warnings  
from xgb_utils_binary import *
import scipy.stats as stats
from multiprocessing import set_start_method
from sklearn.decomposition import PCA
import pickle as pk
from fs_module.feature_selector import FeatureSelector
import pandas as pd
"======setting======" 
threshold = 28 #different feature number
participant_list = np.array([2,3,4,5,6,7,8,9,10,11,13,14,15,16,17])
fold = len(participant_list)
uselabel = ['baseline','stress','amusement']
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
		if float(row[-1])==2:
			label_all_list.append(float(row[-1])-2)
		else:
			label_all_list.append(float(row[-1]))
		data_all_list.append(row[1:-1])
		#print(float(row[-1]),label_all_list[-1])
		#input()
	file.close()
	feat_name = np.array(alllist[0][1:-1])
	label_all_list = np.array(label_all_list).astype('int')
	data_all_list = np.array(data_all_list).astype('float')
	data_all_list1, feat_name1 = feature_clean(data_all_list[:,task[0][0]:task[0][1]], feat_name[task[0][0]:task[0][1]], name_all)
	data_all_list2, feat_name2 = feature_clean(data_all_list[:,task[1][0]:task[1][1]], feat_name[task[1][0]:task[1][1]], name_all)
	#data_all_list3, feat_name3 = feature_clean(data_all_list[:,task[2][0]:task[2][1]], feat_name[task[2][0]:task[2][1]], name_all)
	data_all_list = np.concatenate((data_all_list1,data_all_list2),axis=1)
	feat_name = np.concatenate((feat_name1,feat_name2),axis=0)
	print(feat_name1[0], feat_name1[-1], feat_name1.shape)
	print(feat_name2[0], feat_name2[-1], feat_name2.shape)
	#print(feat_name3[0], feat_name3[-1], feat_name3.shape)
	print(data_all_list.shape, feat_name.shape, np.unique(label_all_list))
	input()
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

	for num, part in enumerate(participant_list):
		if num == 0:
			X_train = data_all_dict[part].tolist()
			Y_train = label_all_dict[part]
		else: 
			X_train = X_train + data_all_dict[part].tolist()
			Y_train = Y_train + label_all_dict[part]
	X_train = np.array(X_train)
	Y_train = np.array(Y_train)
	#X_train = (X_train-np.mean(X_train,axis=0))/np.std(X_train,axis=0)

	pd_data = pd.DataFrame(data=X_train,columns=feat_name)
	pd_label = pd.DataFrame(data=Y_train,columns=['label'])
	fs = FeatureSelector(data = pd_data, labels = pd_label)
	fs.identify_missing(missing_threshold=0.6)
	fs.identify_single_unique()
	fs.identify_collinear(correlation_threshold=0.9)
	correlated_features = fs.ops['collinear']
	#fs.plot_collinear(plot_all=True)
	#plt.show()
	pd_data_new = fs.remove(methods=['collinear'])
	new_data = pd_data_new.values
	new_name = pd_data_new.columns.values
	#print(new_data.shape, new_name.shape)
	data_all_dict = {}
	for i, name in enumerate(name_all):
		if name not in data_all_dict:
			data_all_dict[name] = [new_data[i]]
		else:
			data_all_dict[name].append(new_data[i])
	return new_name, label_all_dict ,data_all_dict


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
	os.makedirs("gsr_result_{}/2_class/acc/".format(sensor), exist_ok=True)
	os.makedirs("gsr_result_{}/2_class/val/".format(sensor), exist_ok=True)
	os.makedirs("gsr_result_{}/2_class/ft/".format(sensor), exist_ok=True)
	os.makedirs("gsr_result_{}/2_class/importance/".format(sensor),exist_ok=True)
	trainTask([[0,48],[111,148]],'time.csv')
	trainTask([[48,78],[148,163]],'frequency.csv')
	trainTask([[78,100],[163,184]],'wavelet.csv')
	trainTask([[100,111],[184,196]],'entropy.csv')
	trainTask([[196,210]],'WESAD_baseline.csv')
	trainTask([[0,210]],'fusion.csv')


def trainTask(task,outfile):
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
				X_train = data_all[part]
				Y_train = label_all[part]
			else: 
				X_train = X_train + data_all[part]
				Y_train = Y_train + label_all[part]
		X_train = np.array(X_train)
		Y_train = np.array(Y_train)
		print(X_train.shape,X_val.shape)

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
																		  val, threshold , name, outfile[:-4], feature, sensor)
		predict_all.append(predict)
		target_all.append(target)
		train_acc[:,num] = tn_acc
		train_f1[:,num] = tn_f1
		val_acc[:,num] = v_acc
		val_f1[:,num] = v_f1
		feature_size[:,num] = feat_size
	"===========getresult==========="
	folder = 'gsr_result_{}/2_class/val/'.format(sensor)+outfile[:-4]+'/'
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
	vote, best_feature = getbestfeature(feature)
	accuracy = np.array([feature_size, train_acc, train_f1, val_acc, val_f1]).T
	writefile('gsr_result_{}/2_class/acc/'.format(sensor)+outfile,accuracy,'w')
	writefeat('gsr_result_{}/2_class/ft/'.format(sensor)+outfile,best_feature,vote,'w')
	
if __name__ == '__main__':
	warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)  
	infile = argv[1]
	main()

