import os
import fnmatch
import shutil

os.chdir("./");

matches = []
for root,dirs,files in os.walk('./'):
        for filename in fnmatch.filter(files,'*.mtx'):
                matches.append(os.path.join(root,filename))
                print os.path.join(root,filename), "added"
for files in matches:
        os.remove(files);
        print files,"removed";

matches = []
for root,dirs,files in os.walk('./'):
        for subdirname in dirs:
                matches.append(os.path.join(root,subdirname));
                print os.path.join(root,subdirname), "added";
for dirpath in matches:
        shutil.rmtree(dirpath);
        print dirpath, "removed";


