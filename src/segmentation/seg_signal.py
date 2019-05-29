import _pickle as pk
from sys import argv
import numpy as np
import os
import matplotlib.transforms as mtransforms

datafolder = './data/WESAD/'
os.makedirs(datafolder, exist_ok=True)
os.makedirs('./data/seg_data', exist_ok=True)

labelname = ['trasient','baseline','stress','amusement','meditation','None','None','None']
uselabel = ['baseline','stress','amusement']
uselabel_count = {1:0,2:0,3:0}
fs_chest = 700
fs_wrist = 4
all_seg_w = 0
all_seg_c = 0
# 132593
# fs > 2 * fc

def readfile(sub,filename, plot=False, normalize=False):
	raw_data = pk.load(open(filename, 'rb'),encoding='latin1')
	c_eda = np.array(raw_data['signal']['chest']['EDA']).flatten()
	w_eda = np.array(raw_data['signal']['wrist']['EDA']).flatten()
	label = np.array(raw_data['label'])
	if normalize:
		c_eda = (c_eda-np.mean(c_eda))/np.std(c_eda)
		w_eda = (w_eda-np.mean(w_eda))/np.std(w_eda)
		#print(np.mean(c_eda), np.mean(w_eda))
	if plot:
		label_down = label[::175]
		c_eda_down = downsample(c_eda,fs=700,nfs=4)
		print(label_down.shape, c_eda_down.shape)
		input()
		fig = plt.figure()
		color = ['w','grey','lightblue','pink','tan','ivory','ivory','ivory']
		
		ax1 = fig.add_subplot(211)
		trans = mtransforms.blended_transform_factory(ax1.transData, ax1.transAxes)
		ax1.plot(c_eda_down)
		for i in range(len(np.unique(label_down))):
			ax1.fill_between(range(len(label_down)), 0, 1, where=label_down==np.unique(label_down)[i], facecolor=color[i], transform=trans,label=labelname[i])
		ax1.legend(loc='upper right', fontsize = 'small')
		ax1.set_xlabel('chest',fontsize='medium')

		ax2 = fig.add_subplot(212)
		trans = mtransforms.blended_transform_factory(ax2.transData, ax2.transAxes)
		ax2.plot(w_eda)
		for i in range(len(np.unique(label_down))):
			ax2.fill_between(range(len(label_down)), 0, 1, where=label_down==np.unique(label_down)[i], facecolor=color[i], transform=trans,label=labelname[i])
		ax2.legend(loc='upper right', fontsize = 'small')
		ax2.set_xlabel('wrist',fontsize='medium')

		plt.show()
		#plt.savefig('image/'+str(sub)+'.png')
		plt.close()

	return c_eda, w_eda, label

def seg_label(c_eda, w_eda, label):
	data = {'w':{1:[],2:[],3:[]},'c':{1:[],2:[],3:[]}}
	for i in range(len(w_eda)):
		if int(label[i*175]) in data['w']:
			data['w'][int(label[i*175])].append(w_eda[i])
	for i in range(len(c_eda)):
		if int(label[i]) in data['c']:
			data['c'][int(label[i])].append(c_eda[i])

	for i in [1,2,3]:
		if len(data['w'][i])*175>len(data['c'][i]):
			for j in range(len(data['w'][i])*175 - len(data['c'][i])):
				data['c'][i].append(data['c'][i][-1])
		elif len(data['w'][i])*175<len(data['c'][i]):
			data['c'][i]= data['c'][i][:len(data['w'][i])*175]

	print('minutes: {}'.format((len(data['w'][1])+len(data['w'][2])+len(data['w'][3]))/(4*60)))
	#print(len(data['w'][1])*175, len(data['c'][1]), len(data['w'][2])*175, len(data['c'][2]),len(data['w'][3])*175, len(data['c'][3]))
	return data

def seg_window(data,win_size,stride,plot=False):
	global all_seg_c, all_seg_w, uselabel_count
	seg_data = {'w':{1:[],2:[],3:[]},'c':{1:[],2:[],3:[]}}
	for i in [1,2,3]:
		for j in range(0,len(data['w'][i]),int(stride*fs_wrist)):
			if j+win_size*fs_wrist >= len(data['w'][i]):
				break
			seg_data['w'][i].append([data['w'][i][j:j+win_size*fs_wrist]])
			all_seg_w += 1
			uselabel_count[i] += 1
			if plot:
				print(i)
				plotsignal(seg_data['w'][i][-1])

	for i in [1,2,3]:
		for j in range(0,len(data['c'][i]),int(stride*fs_chest)):
			if j+win_size*fs_chest >= len(data['c'][i]):
				break
			seg_data['c'][i].append([data['c'][i][j:j+win_size*fs_chest]])
			all_seg_c += 1
			if plot:
				print(i)
				plotsignal(seg_data['c'][i][-1])
	#print(len(seg_data['w'][1]), len(seg_data['c'][1]), len(seg_data['w'][2]), len(seg_data['c'][2]),len(seg_data['w'][3]), len(seg_data['c'][3]))
	return seg_data

def readlabel():
	target = pk.load(open(targetfolder+'target.pkl','rb'))
	predict = pk.load(open(targetfolder+'predict.pkl','rb'))
	for participant in range(15):
		if participant == 8:
			predict_all = predict[participant][0].tolist()
			target_all = target[participant][0].tolist()
		#else:
		#	predict_all += predict[participant][thr].tolist()
		#	target_all += target[participant][thr].tolist()
	predict_all = np.array(predict_all)
	target_all = np.array(target_all)
	print(predict_all.shape)
	print(target_all.shape)
	return predict_all, target_all

def main():
	participant_list = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17]
	participant_data = {}
	#predict, target = readlabel()
	for participant in participant_list:
		sub = 'S'+str(participant)
		print(sub)
		filename = datafolder+sub+'/'+sub+'.pkl' 
		c_eda, w_eda, label = readfile(sub,filename, plot=False, normalize=False)
		all_data = seg_label(c_eda, w_eda, label)
		all_data = seg_window(all_data, win_size=60, stride=30, plot=False)
		participant_data[participant] = all_data
	print('all seg count: {}'.format(all_seg_c))
	for i in list(uselabel_count.keys()):
		print('{}: {}'.format(uselabel[i-1], uselabel_count[i]/(all_seg_c)))
	pk.dump(participant_data,open('./data/seg_data/'+outputname+'.pkl', 'wb'))
	
if __name__ == '__main__':
	outputname = argv[1]
	#targetfolder = argv[2]
	main()