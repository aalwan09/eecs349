import csv
import sys


def main():

	if ((len(sys.argv) < 5) | (len(sys.argv) > 5)):
		sys.exit( "Wrong number of command line arguments. \n Usage: python decisiontree.py <inputFileName> <trainingSetSize> <numberOfTrials> <verbose> ")
	else:
		try:
			inputFileName = str(sys.argv[1])
			trainingSetSize = int(sys.argv[2])
			numberOfTrials = int(sys.argv[3])
			verbose = bool(sys.argv[4])
		except:
			sys.exit("The provided arguments are malformed")
	reader = csv.reader(open(inputFileName, 'rb'), delimiter='\t')
	for row in reader:
		print row
	



if __name__ == "__main__":
    main()