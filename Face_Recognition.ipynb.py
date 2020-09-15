#!/usr/bin/env python
# coding: utf-8

# In[54]:


import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data,exposure
import cv2


# In[55]:


image=cv2.imread('elle.jpg')
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


# In[56]:


fd,hog_image=hog(image,orientations=8,pixels_per_cell=(16,16),
                cells_per_block=(1,1),visualize=True,multichannel=True)


# In[57]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(8,4), sharex=True, sharey=True)
ax1.axis('off')
ax1.imshow(image,cmap=plt.cm.gray)
ax1.set_title('Input Image')

hog_image_rescaled=exposure.rescale_intensity(hog_image,in_range=(0,10))
ax2.axis('off')
ax2.imshow(hog_image_rescaled,cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients')
plt.show()


# In[58]:


len(fd)


# In[59]:


image.shape


# In[60]:


import face_recognition

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import numpy as np
import cv2

get_ipython().run_line_magic('matplotlib', 'inline')


# In[61]:


image=cv2.imread('leeandelle.jpg')
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image)


# In[62]:


face_locations=face_recognition.face_locations(image)
number_of_faces=len(face_locations)
print("Found {} face(s) in the Image".format(number_of_faces))


# In[63]:


plt.imshow(image)
ax=plt.gca()

for face_location in face_locations:
    
    #location of each face in this image.list of co-ordinates in (top,right,bottom,left)
    top,right,bottom,left=face_location
    x,y,w,h=left,top,right,bottom
    print("A face is located at pixel location Top: {},Left: {},Bottom: {},Right: {}".format(x,y,w,h))
    
    
    
    #draw a box around the face
    rect=Rectangle((x,y),w-x,h-y,fill=False,color='red')
    ax.add_patch(rect)
    
#output
plt.show()    


# In[64]:


import face_recognition

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
import numpy as np
import cv2

get_ipython().run_line_magic('matplotlib', 'inline')


# In[80]:


#Load the Known Faces(Database creation)
#elle=face_recognition.load_image_file("person.jpg")
image=cv2.imread('elle.jpg')
elle=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

image=cv2.imread('lee.jpg')
lee=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

image=cv2.imread('kat.jpg')
kat=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


# In[81]:


elle_encoding=face_recognition.face_encodings(elle)[0]
lee_encoding=face_recognition.face_encodings(lee)[0]
kat_encoding=face_recognition.face_encodings(kat)[0]


# In[82]:


known_face_encodings=[
    elle_encoding,
    lee_encoding,
    kat_encoding
]


# In[93]:


image=cv2.imread('elle2.jpg')
unknown_image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(unknown_image)

unknown_face_encodings=face_recognition.face_encodings(unknown_image)


# In[94]:


from scipy.spatial import distance

#looping to find out number of faces
for unknown_face_encoding in unknown_face_encodings:
    
    results=[]
    for known_face_encoding in known_face_encodings:
        d=distance.euclidean(known_face_encoding,unknown_face_encoding)
        #print("Euclidean distance: ",d)
        results.append(d)
    threshold=0.6
    results=np.array(results) <= threshold
    
    name="Unknown"
    
    if results[0]:
        name='Elle'
    elif results[1]:
        name='Lee'
    elif results[2]:
        name='Kat'
    print(f"Found {name} in the Photo")
    
    


# In[95]:


#Finding Facial Features
face_landmarks_list=face_recognition.face_landmarks(image)


# In[98]:


import matplotlib.lines as mlines
from matplotlib.patches import Polygon
 
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
plt.imshow(image)
ax=plt.gca()

for face_landmarks in face_landmarks_list:
    left_eyebrow_pts=face_landmarks['left_eyebrow']
    pre_x,pre_y=left_eyebrow_pts[0]
    for (x,y) in left_eyebrow_pts[1:]:
        l=mlines.Line2D([pre_x,x],[pre_y,y],color="red")
        ax.add_line(l)
        pre_x,pre_y=x,y
    
    
    right_eyebrow_pts=face_landmarks['right_eyebrow']
    pre_x,pre_y=right_eyebrow_pts[0]
    for (x,y) in right_eyebrow_pts[1:]:
        l=mlines.Line2D([pre_x,x],[pre_y,y],color="red")
        ax.add_line(l)
        pre_x,pre_y=x,y
    
    p=Polygon(face_landmarks['top_lip'],facecolor='lightsalmon',edgecolor='orangered')
    ax.add_patch(p)
    p=Polygon(face_landmarks['bottom_lip'],facecolor='lightsalmon',edgecolor='orangered')
    ax.add_patch(p)

plt.show()
        
        
    


# In[ ]:




