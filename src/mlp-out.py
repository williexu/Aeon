import pickle

res=[]
with open('../data/mlp_results-100.txt','r') as f:
	res.append(pickle.load(f))

with open('../data/mlp_results-200.txt','r') as f:
	res.append(pickle.load(f))

with open('../data/mlp_results-500.txt','r') as f:
	res.append(pickle.load(f))

with open('../data/mlp_results-1000.txt','r') as f:
	res.append(pickle.load(f))

with open('../data/mlp_results-2000.txt','r') as f:
	res.append(pickle.load(f))

for r in res:
	for i in r:
		print "Hidden Layers: "+str(i[0])+ " MSE: "+str(i[1])
	print
	print	
