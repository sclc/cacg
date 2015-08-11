#python3 code

import os;
import sqlite3;
import csv;
import urllib.request;


def GetUrlByName (csvItemList, DBCur, generatedUrlList):
	for item in csvItemList:
		#tempStr = item.replace('.mtx', '',1);
		tempStr = item;
		datalist = DBCur.execute ("select groupid from FMM where name ='%s'" %tempStr);
		for data in datalist:
			generatedUrlList.append( (data[0], tempStr) ); # in case of tongming matrix
	return


itemList = [];
urlList = [];
baseUrl = "http://www.cise.ufl.edu/research/sparse/MM/";
filePostfix = ".tar.gz";

with open("MatrixNameListToDownload.txt") as csvfile:
	matrixReader = csv.reader(csvfile, delimiter=',');
	for row in matrixReader:
		#print(row[0]);
		itemList.append(row[0]);

DBConnector = sqlite3.connect("../FMM_DB.db");

DBCursor = DBConnector.cursor();

GetUrlByName(itemList, DBCursor, urlList);

DBConnector.close;

# Still no way to handle isonym matrix, file downloaded previously will be over-written
for matrixInfo in urlList:
	tempStr = baseUrl + matrixInfo[0] + "/" + matrixInfo[1] + filePostfix;  
	matrixFile = urllib.request.urlopen(tempStr);
	savefile = open(matrixInfo[1]+filePostfix, 'wb');
	savefile.write( matrixFile.read() );
	savefile.close();
	print (tempStr, " downloaded");

