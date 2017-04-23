import json
import gzip


def parse(path):
	g = gzip.open(path, 'r')
	for l in g:
		yield eval(l)

# reviews_file = open("../data/review_Sport_and_Outdoors_5.json")
# reviews_file = open("../data/review_Sport_and_Outdoors_5.json")

counter = 0

for l in parse("../data/reviews_Sports_and_Outdoors_5.json.gz"):
	print l
	counter += 1
	if counter > 9:
		break
