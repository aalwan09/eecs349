import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist



###PROBLEM 1A
images, labels = load_mnist(digits=range(0,10), path = '.')
print "total number of images: " + str (len(images))

for i in range(0,10):
    images, labels = load_mnist(digits=[i], path='.')

    print str(i) + " number of images: " + str(len(images))


