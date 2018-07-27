from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import sys
import kmeans as km
import pandas as pd

#K-Means clustering implementation
#Some hints on how to start have been added to this file.
#You will have to add more code that just the hints provided here for the full implementation.

# ====
# Define a function that computes the distance between two data points
def dist(a, b, ax=1):
	return np.linalg.norm(a - b, axis=ax)
# ====
# Define a function that reads data in from the csv files  HINT: http://docs.python.org/2/library/csv.html
def readFile(name):
	nm = name[4:-4]
	print nm
	if nm is "Both":
		colnames = ['Countries', 'BirthRate', 'LifeExpectancy']
	else:
		colnames = ['Countries', 'BirthRate', 'LifeExpectancy']

	data = pd.read_csv(name, names=colnames)
	countries = data.Countries.tolist()
	birthRate = data.BirthRate.tolist()
	lifeExpec = data.LifeExpectancy.tolist()	
	return data, countries, birthRate, lifeExpec
# ====
# Write the initialisation procedure
def main(args):
	print args

if __name__ == '__main__':
	main(sys.argv)
	name = raw_input("Please enter the name of the file you wish to analize: ")
	clust = raw_input("Please enter the number of clusters to create: ")
	data, countries, birthRate, lifeExpec = km.readFile(name)
# ====
# Implement the k-means algorithm, using appropriate looping
	birthRate = birthRate[1:]
	countries = countries[1:]
	lifeExpec = lifeExpec[1:]
	birthRate = [float(i) for i in birthRate]
	lifeExpec = [float(i) for i in lifeExpec]

	X = np.array(list(zip(birthRate, lifeExpec))).reshape(len(birthRate), 2)
	colors = ['b', 'g', 'r', 'k']
	markers = ['o']

	K = int(clust)

	itr = raw_input("How many iterations would you like to run?\n")
	itr = int(itr)
	kmeans_model = KMeans(n_clusters=K, max_iter = itr).fit(X)
	print "inertia: "+str(kmeans_model.inertia_)
	#print {i: np.where(kmeans_model.labels_ == i)[0] for i in range(kmeans_model.n_clusters)}
	

	plt.plot()
	lstClust = []
	lstCountries = []
	for i, l in enumerate(kmeans_model.labels_):
	    plt.plot(birthRate[i], lifeExpec[i], color=colors[l], marker='o',ls='None')
	    lstCountries.append(i)
	    lstClust.append(l)
	    plt.xlim([0, 100])
	    plt.ylim([0, 100])
	#print lstCountries
	#print lstClust
#	plt.show()
	
	zero = 0
	one = 0
	two = 0
	three = 0
	four = 0

	for i in lstClust:
		if i == 0:
			zero = zero+1
		elif i == 1 :	
			one = one+1
		elif i == 2:
			two = two+1
		elif i == 3:
			three = three+1
			
# ====
# Print out the results
	if K==1:
		print "Number of blue countries: "+str(zero)

	elif K==2:
		print "Number of blue countries: "+str(zero)
		print "Number of green countries: "+str(one)

	elif K==3:
		print "Number of blue countries: "+str(zero)
		print "Number of green countries: "+str(one)
		print "Number of red countries: "+str(two)
	elif K==4:
		print "Number of blue countries: "+str(zero)
		print "Number of green countries: "+str(one)
		print "Number of red countries: "+str(two)
		print "Number of black countries: "+str(three)

	count = 0


	ob = raw_input("Please enter to continue: ")
	while (count < K):
		cntryList = []
		for cntry, clust in zip(countries, lstClust):
		#	print "COUNTRY: "+str(cntry)
		#	print "CLUSTER: "+str(clust)
			if clust == 0 and count == 0:
				cntryList.append(cntry)
			if clust == 1 and count == 1:
				cntryList.append(cntry)
			if clust == 2 and count == 2:
				cntryList.append(cntry)
			if clust == 3 and count == 3:
				cntryList.append(cntry)
		if count == 0:
			print "Countries in Blue:\n"
			for i in cntryList:
				print i
			print "\n"

		if count == 1:
			print "Countries in Green:\n"
			for i in cntryList:
				print i
			print "\n"
		if count == 2:
			print "Countries in Red:\n"
			for i in cntryList:
				print i
			print "\n"
		if count == 3:
			print "Countries in Black:\n"
			for i in cntryList:
				print i
			print "\n"
		count = count+1
	cntr = 0

	bi = raw_input("Press enter when you're ready to continue")
	while (cntr < K):
		meanLifeEx = []
		meanBirthR = []
		for le, br, cntry, clust in zip(lifeExpec, birthRate, countries, lstClust):
			if clust == cntr:
				meanLifeEx.append(le)
				meanBirthR.append(br)
		if cntr == 0:
			print "Average Birth Rate for Blue = "+str(sum(meanBirthR)/len(meanBirthR))
			print "Average Life Expectancy rate for Blue = "+str(sum(meanLifeEx)/len(meanLifeEx))+"\n"		
		elif cntr == 1:
			print "Average Birth Rate for Green = "+str(sum(meanBirthR)/len(meanBirthR))
			print "Average Life Expectancy rate for Green = "+str(sum(meanLifeEx)/len(meanLifeEx))+"\n"

		elif cntr == 2:
			print "Average Birth Rate for Red = "+str(sum(meanBirthR)/len(meanBirthR))
			print "Average Life Expectancy rate for Red = "+str(sum(meanLifeEx)/len(meanLifeEx))+"\n"
		elif cntr == 3:
			print "Average Birth Rate for Black = "+str(sum(meanBirthR)/len(meanBirthR))
			print "Average Life Expectancy rate for Black = "+str(sum(meanLifeEx)/len(meanLifeEx))+"\n"





		cntr =  cntr+1
	
	bz = raw_input("Press Enter when you're ready to plot <<<<<<")
	plt.title(name, fontsize=20)
	plt.xlabel('Birth Rate', fontsize=18)
	plt.ylabel('Ave Life Expectancy', fontsize=18)
	plt.show()

