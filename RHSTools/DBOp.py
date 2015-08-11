#!/usr/bin/python3

import os;
import sqlite3;
import csv;
import re;

def Norm2CondColUpdate (csvItemList, DBCur):
	for item in csvItemList:
        	tempStr = item[0].replace('.mtx', '',1);
        	DBCur.execute ( "update FMM set Norm2Cond = ? where name = ?", (item[1], tempStr) ); 

	return	

def UpdatedDBCheck (csvItemList, DBCur):
	for item in csvItemList:
        	tempStr = item[0].replace('.mtx', '',1);
        	datalist = DBCur.execute ("select * from FMM where name ='%s'" %tempStr);
        	for data in datalist:
                	print (data);
	return


itemList = [];

with open("ConditionNumberFile.txt") as csvfile:
	matrixReader = csv.reader(csvfile, delimiter=',');
	for row in matrixReader:
		print(row[0],row[1]);
		itemList.append(row);

DBConnector = sqlite3.connect("./FMM_DB.db");

DBCursor = DBConnector.cursor();

#Norm2CondColUpdate(itemList, DBCursor);
UpdatedDBCheck(itemList, DBCursor);

DBConnector.commit();
DBConnector.close;
