# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 12:33:15 2017

@author: debbie
"""

from PIL import Image
import math
import sys

img = Image.open(sys.argv[1])
#Image.open('westbrook.jpg')
pix = img.load()
#out = img.point(lambda i : math.floor(i / 2))
#out.save('Q2.jpg')



for i in range(img.size[0]): # for every pixel:
    for j in range(img.size[1]):        
        r, g, b = pix[i, j]
        
        r = (int)(r / 2)
        g = (int)(g / 2)
        b = (int)(b / 2)
        pix[i, j] = r, g, b
        #img.putpixel((i, j), (r, g, b))
        
img.save('Q2.png')

#rgb_im = img.convert('RGB')
#r, g, b = rgb_im.getpixel((1, 1))
#print(r, g, b)
#r2, g2, b2 = rgb_im.split()
#print(r2)


#rgb_out = img.convert('RGB')
#r1, g1, b1 = rgb_out.getpixel((1, 1))
#print(r1, g1, b1)

