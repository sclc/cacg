#!/usr/bin/python

import os
import glob
import tarfile

for files in glob.glob("*.tar.gz"):
	print "extract", files;
	tfile = tarfile.open(files, 'r:gz');
	tfile.extractall('.');
	print "extract", files, "done";

