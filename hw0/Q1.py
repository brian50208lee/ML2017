import sys, numpy

# read
matrix1 = numpy.loadtxt(fname=sys.argv[1], dtype='int', delimiter=',', ndmin=2)
matrix2 = numpy.loadtxt(fname=sys.argv[2], dtype='int', delimiter=',', ndmin=2)

# dot
matrix3 = numpy.dot(matrix1, matrix2)

# sort
array = matrix3[0]
array.sort()

# output
out = open('ans_one.txt', 'w')
for value in array:
	out.write("%d\n" % value)
out.close()

