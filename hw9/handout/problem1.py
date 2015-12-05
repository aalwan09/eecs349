import matplotlib.pyplot as plt
import numpy as np
from mnist import load_mnist




images, labels = load_mnist(digits=[2], path='.')

print "number of images: " + str(len(images))

