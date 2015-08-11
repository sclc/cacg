#!/usr/bin/python

import os;
import sys;
import numpy as np;


def output_all(argv):
	row_num = int(sys.argv[1]);
	col_num = int(sys.argv[2]);

	if row_num * col_num * 8 / (1024*1024*1024) > 4:
		print "Too big a matrix > 4GB"
		exit()

	#mrhs = np.random.rand(row_num, col_num)
	mrhs = np.random.beta(1.5,1.5 ,(row_num, col_num) )
	print str(float(row_num)*float(col_num)*8.0/(1024.0**3))+"GB matrix generated"

	np.savetxt(sys.argv[3], mrhs, fmt='%.18e', delimiter=',', newline='\n');
	return

#argv[1]: row num; arg[2]: col num; arg[3]: output from 0 to argv[3]
#argv[4]: csv file name prefix
def output_specified_columns(argv):
	row_num = int(sys.argv[1]);
	col_num = int(sys.argv[2]);

	if row_num * col_num * 8 / (1024*1024*1024) > 4:
		print "Too big a matrix > 4GB"
		exit()

	#mrhs = np.random.rand(row_num, col_num)
	mrhs = np.random.beta(1.5,1.5 , (row_num, col_num) )
	print str(float(row_num)*float(col_num)*8.0/(1024.0**3))+"GB matrix generated"
	
	for fileid in range(1,int(argv[3])+1):
		np.savetxt(argv[4]+"_"+str(fileid)+".csv", mrhs[:,0:fileid], fmt='%.18e', delimiter=',', newline='\n');

	return

	
#print mrhs[1,:]
#output_all(sys.argv)
output_specified_columns(sys.argv);
