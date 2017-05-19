import pickle

with open('../data/individual_words.txt', 'rb') as word_file:
	word_set = pickle.load(word_file)
	print type(word_set)
	print len(word_set)
	set200000 = set()
	set500000 = set()
	set1000000 = set()
	set5000000 = set()
	for word in word_set:
		hashed = hash(word)
		set200000.add(hashed % 200000)
		set500000.add(hashed % 500000)
		set1000000.add(hashed % 1000000)
		set5000000.add(hashed % 5000000)
	print "200,000: ", len(set200000)
	print "500,000: ", len(set500000)
	print "1,000,000: ", len(set1000000)
	print "5,000,000: ", len(set5000000)
		

# 109182
# 200,000:  81383
# 500,000:  94131
# 1,000,000:  98913
# 5,000,000:  107375