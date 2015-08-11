#!/usr/bin/python

import os
import fnmatch
import shutil

#for root,dirs,files in os.walk("."):
	#print root 
	#print dirs 
#	print files 

matches = []
for root,dirs,files in os.walk('./'):
	for filename in fnmatch.filter(files,'*.mtx'):
		matches.append(os.path.join(root,filename))
		print os.path.join(root,filename), "added"
for src_files in matches:
	shutil.move(src_files, "./");	
	print "move", src_files, "done";
