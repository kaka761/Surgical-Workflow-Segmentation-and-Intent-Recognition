# -*- coding: utf-8 -*-

import os
   
files=os.listdir('F:/Feature/MISAW/Procedural decription/')#annotation position
#print(files)
#print(len(files)-1)
total = []
for i in range(len(files)-1):
    with open('F:/Feature/MISAW/Procedural decription/' + files[i],'r') as f:
        data = f.readlines()
        total.append(data[1:])
total = sum(total,[])
fi = open('F:/Feature/MISAW/Procedural decription/' + 'annotation.txt','w')#final txt position
for t in total:
    fi.write(t)
fi.close()

