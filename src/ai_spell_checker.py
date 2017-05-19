import pickle
from collections import Counter
import re
from nltk.stem import WordNetLemmatizer

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]

    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def subwords(word):
	s = [(word[:i+1], word[i+1:]) for i in range(len(word)-1)]
	return set([tup[i] for i in range(2) for tup in s])

def known(words):
    return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or known(subwords(word)) or [word]
    return max(candidates, key=NWORDS.get)


def words(text):
    return re.findall('[a-z]+', text.lower())

NWORDS = Counter(words(file('big.txt').read())) # use big.txt for spell checker

# with open('../data/individual_words_lemmatized.txt', 'rb') as word_file:
# 	my_words = pickle.load(word_file) # use own word file
# 	NWORDS = Counter(words(file('big.txt').read())) # use big.txt for spell checker
# 	lemmatizer = WordNetLemmatizer()
# 	def check_reduce(word):
# 		# stemmed = stemmer.stem(word)
# 		# return lemmatizer.lemmatize(stemmed)
# 		return lemmatizer.lemmatize(correct(word))
# 		# return stemmer.stem(word)


# 	# print type(NWORDS)
# 	# print correct("jsdhfjhd")
# 	# print correct("hattt")
# 	# print correct("haters")
# 	# print correct("spell")
# 	# print correct("spelling")
# 	# print correct("speling")
# 	# print NWORDS["g"]
# 	# print NWORDS.most_common(100)
# 	# word_counter = Counter([correct(word) for word in NWORDS])
# 	word_counter = Counter()
# 	counter = 0
# 	for word in my_words:
# 		corrected = check_reduce(word)
# 		if corrected in word_counter:
# 			word_counter[corrected] += my_words[word]
# 		else:
# 			word_counter[corrected] = my_words[word]
# 		counter += 1
# 		# if counter % 100 == 0:
# 		print counter

# 	print len(NWORDS)
# 	print len(word_counter)


# with open('../data/words_spell_checked.txt', 'wb') as word_file:
# 	pickle.dump(word_counter, word_file)

	# print len(word_counter)
	# print word_counter 




	# set200000 = set()
	# set500000 = set()
	# set1000000 = set()
	# set5000000 = set()
	# for word in word_set:
	# 	hashed = hash(word)
	# 	set200000.add(hashed % 200000)
	# 	set500000.add(hashed % 500000)
	# 	set1000000.add(hashed % 1000000)
	# 	set5000000.add(hashed % 5000000)
	# print "200,000: ", len(set200000)
	# print "500,000: ", len(set500000)
	# print "1,000,000: ", len(set1000000)
	# print "5,000,000: ", len(set5000000)
		

# 109182
# 200,000:  81383
# 500,000:  94131
# 1,000,000:  98913
# 5,000,000:  107375