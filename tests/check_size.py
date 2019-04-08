import sys
x0 = int(sys.argv[1])

#        				ksize		pad		stride
x1 			= 1+ (x0 -	3 	+ 2 *	0) / 	2  		# conv0
x2 			= 1+ (x1 - 	3 	+ 2 *	0) /	1		# conv1
x3 			= 1+ (x2 - 	3 	+ 2 *	1) / 	1		# conv2
x4 			= 1+ (x3 - 	3 	+ 2 *	0) / 	2  		# pool2
x5 			= 1+ (x4 - 	1 	+ 2 *	0) / 	1  		# conv3
x6 			= 1+ (x5 - 	3 	+ 2 *	0) / 	1  		# conv4
head_out	= 1+ (x6 - 	3 	+ 2 *	0) / 	2  		# pool4

print("Head:\t", end="")
print(*[x0, x1, x2, x3, x4, x5, x6, head_out], sep=" -> ")

x0 = head_out
#        				ksize		pad		stride
x1 			= 1+ (x0 - 	3 	+ 2 *	1) / 	1		# 3 x inception1 block
x2 			= 1+ (x1 - 	3 	+ 2 *	0) / 	2		# 1 x inception2 block
x3 			= 1+ (x2 - 	3 	+ 2 *	1) / 	1		# 4 x inception3 block
x4 			= 1+ (x3 - 	3 	+ 2 *	0) / 	2		# 1 x inception4 block
x5 			= 1+ (x4 - 	3 	+ 2 *	1) / 	1		# 2 x inception5 block

print("Blocks:\t", end="")
print(*[x0, x1, x2, x3, x4, x5], sep=" -> ")


##### "GOOD" sizes:
# 299 427 491 523 555
