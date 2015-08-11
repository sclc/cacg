#!/usr/bin/python3

import os;
import sqlite3;
import csv;
import re;

def DB_General_Query (csvItemList, DBCur):
	for item in csvItemList:
        	tempStr = item[0].replace('.mtx', '',1);
        	datalist = DBCur.execute ("select * from FMM where name ='%s'" %tempStr);
        	for data in datalist:
                	print (data);
	return

def DB_Diagonal_Matrix_Query (csvItemList, DBCur):
	for item in csvItemList:
        	tempStr = item[0].replace('.mtx', '',1);
        	datalist = DBCur.execute ("select * from FMM where name ='%s' and patternsymmetry==1 and nrows==nnz" %tempStr);
        	for data in datalist:
                	print (data);
	return


itemList = [];

with open("MatrixNameList.txt") as csvfile:
	matrixReader = csv.reader(csvfile, delimiter=',');
	for row in matrixReader:
		print(row[0]);
		itemList.append(row);

DBConnector = sqlite3.connect("/home/scl/MStore/FMM_DB.db");

DBCursor = DBConnector.cursor();

#Norm2CondColUpdate(itemList, DBCursor);
DB_General_Query(itemList, DBCursor);
#DB_Diagonal_Matrix_Query(itemList, DBCursor);

DBConnector.close;
