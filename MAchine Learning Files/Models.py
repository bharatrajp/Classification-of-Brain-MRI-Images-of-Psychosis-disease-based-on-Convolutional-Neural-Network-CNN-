#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import math
from functools import partial
from torch.autograd import Variable


# In[ ]:


#Brain Inf. (2021) 8:23 https://doi.org/10.1186/s40708-021-00144-2
#3D convolutional neural networks‑based multiclass classification of Alzheimer’s and Parkinson’s 
#diseases using PET and SPECT neuroimaging modalities
#Ahsan Bin Tufail, Yong‑Kui Ma, Qiu‑Na Zhang, Adil Khan, Lei Zhao, Qiang Yang, Muhammad Adeel, Rahim Khan and Inam Ullah

class Tufail_CNN(nn.Module):

    def __init__(self, num_classes):

        
        super(Tufail_CNN, self).__init__()

        self.conv_layer1 = self._make_conv_layer(1, 64)
        self.conv_layer2 = self._make_conv_layer(64, 128)
        self.conv_layer3 = self._make_conv_layer(128, 256)
        self.conv_layer4 = self._make_conv_layer(256, 128)
        self.conv_layer5 = self._make_conv_layer(128, 64)
        
        self.fc1 = nn.Linear(512, 300)    
        self.fc2 = nn.Linear(300, 100)
        self.relu = nn.LeakyReLU()
        
        self.drop=nn.Dropout(p=0.1)
        self.fc3 = nn.Linear(100, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=5, padding=2),
        nn.BatchNorm3d(out_c),
        nn.ELU(),
        #nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
        #nn.LeakyReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2),
        )
        return conv_layer

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        x=self.conv_layer5(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print(x.size())
        x = self.fc1(x)
        x = self.relu(x)
        #print(x.size())
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)

        return x#,x1


# In[ ]:


#Front Psychiatry 2020 Feb 3;11:16. doi: 10.3389/fpsyt.2020.00016. eCollection 2020.
#Identifying Schizophrenia Using Structural MRI With a Deep Learning Algorithm
#Jihoon Oh 1, Baek-Lok Oh 2, Kyong-Uk Lee 3, Jeong-Ho Chae 1, Kyongsik Yun 4 5

class Oh_CNN(nn.Module):

    def __init__(self, num_classes):

        
        super(Oh_CNN, self).__init__()

        self.conv_layer1 = self._make_conv_layer(1, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer3 = self._make_conv_layer(64, 64)
        
        self.fc1 = nn.Linear(768, 512)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(512)
        self.drop=nn.Dropout(p=0.5)        
       
        self.fc2 = nn.Linear(512, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
        nn.LeakyReLU(),
        nn.MaxPool3d(kernel_size=3, stride=3),
        nn.Dropout(p=0.25)
        )
        return conv_layer

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        
        #print(x.size())
        x = self.fc1(x)
        x = self.relu(x)
        #print(x.size())
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc2(x)
       
        return x#,x1


# In[ ]:


#International Workshop on PRedictive Intelligence In MEdicine - PRIME 2020: 
#Lecture Notes in Computer Science book series (LNIP,volume 12329), pp 156–168
#Uniformizing Techniques to Process CT Scans with 3D CNNs for Tuberculosis Prediction
#Hasib Zunair, Aimon Rahman, Nabeel Mohammed & Joseph Paul Cohen 

class Zunair_CNN(nn.Module):

    def __init__(self, num_classes):

        
        super(Zunair_CNN, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            )


        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(128),
            )
        
        self.conv_layer4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(256),
            )
        
        self.fc1 = nn.Linear(9216, 512)
        self.relu = nn.LeakyReLU()
        self.drop=nn.Dropout(p=0.4)        
       
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        
        x = x.view(x.size(0), -1)
        #print(x.size())
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        
        return x#,x1


# In[ ]:


#Computerized Medical Imaging and Graphics 89 (2021) 101882
#Deep learning based automatic diagnosis of first-episode psychosis, bipolar disorder and healthy controls
#Zhuangzhuang Li, Wenmei Li, Yan Wei, Guan Gui, Rongrong Zhang, Haiyan Liu, Yuchen Chen, Yiqiu Jiang 

#Modified, they use 2D
class Li_CNN(nn.Module):

    def __init__(self, num_classes):

        
        super(Li_CNN, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=7, stride=1),
            nn.LeakyReLU(),
            )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=5, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=3, stride=3), # added from original model to keep the fature vector reasonable
            )


        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=3, stride=3),
            )
        
        self.classification = nn.Sequential(
            #nn.Linear(9216, 128),
            nn.Linear(12544, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
            )
        
    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
       
        
        x = x.view(x.size(0), -1)
        #print(x.size())
        
        x = self.classification(x)
        
        return x#,x1


# In[ ]:


#Front. Neurol., 08 April 2020 https://www.frontiersin.org/articles/10.3389/fneur.2020.00244/full
#Brain Morphometry Estimation: From Hours to Seconds Using Deep Learning
#Rebsamen Michael, Suter Yannick, Wiest Roland, Reyes Mauricio, Rummel Christian

class AlexNet_3D(nn.Module):

    def __init__(self, num_classes):

        
        super(AlexNet_3D, self).__init__()

        
        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 96, kernel_size=11, stride=1),
            #nn.Conv3d(1, 144, kernel_size=5, stride=4),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(96, 256, kernel_size=5, stride=1),
            #nn.Conv3d(144, 192, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2),
            )


        self.conv_layer3 = nn.Sequential(
            #nn.Conv3d(192, 192, kernel_size=5, stride=1),
            nn.Conv3d(256, 384, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv3d(384, 384, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv3d(384, 256, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Dropout(p=0.4)
            )
        
        self.fc1 = nn.Linear(11520, 4096)
        self.relu1 = nn.LeakyReLU()
        self.dropout=nn.Dropout(p=0.4)
       
       
        self.fc2 = nn.Linear(4096, 4096)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(4096, num_classes)

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        
        x = x.view(x.size(0), -1)
        #print(x.size())
        
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x#,x1
    

# In[ ]:


#ISBI2017, pp 835–838
#RESIDUAL AND PLAIN CONVOLUTIONAL NEURAL NETWORKS FOR 3D BRAIN MRI CLASSIFICATION
#Sergey Korolev, Amir Safiullin, Mikhail Belyaev, Yulia Dodonova 

class VoxCNN(nn.Module):

    def __init__(self, num_classes):

        
        super(VoxCNN, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(8, 8, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            #nn.BatchNorm3d(16),
            )
        
        self.conv_layer2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(16, 16, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            #nn.BatchNorm3d(32),
            )


        self.conv_layer3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(32, 32, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            #nn.BatchNorm3d(32),
            )
        
        self.conv_layer4 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
            )
        
        self.class_layer = nn.Sequential(
            nn.Linear(5120, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.7),
            nn.Linear(128, 64),
            nn.Linear(64, num_classes),
            )
        

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        
        x = x.view(x.size(0), -1)
        #print(x.size())
        
        x = self.class_layer(x)
        
        return x#,x1


# In[ ]:


#ISBI2017, pp 835–838
#RESIDUAL AND PLAIN CONVOLUTIONAL NEURAL NETWORKS FOR 3D BRAIN MRI CLASSIFICATION
#Sergey Korolev, Amir Safiullin, Mikhail Belyaev, Yulia Dodonova 

class VoxResNet21(nn.Module):

    def __init__(self, num_classes):

        
        super(VoxResNet21, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 32, kernel_size=3, stride=1,padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=3, stride=2,padding=1),
            )
        
        self.res_layer2 = self._make_res_layer(64,1)
        self.res_layer3 = self._make_res_layer(64,1)
        self.bridge_layer4 = self._make_bridge_layer(64,64,1)
        
        self.res_layer5 = self._make_res_layer(64,1)
        self.res_layer6 = self._make_res_layer(64,1)
        self.bridge_layer7 = self._make_bridge_layer(64,128,1)
        
        self.res_layer8 = self._make_res_layer(128,1)
        self.res_layer9 = self._make_res_layer(128,1)
        
        #self.pool = nn.MaxPool3d(kernel_size=7, stride=7)
        self.pool = nn.MaxPool3d(kernel_size=7, stride=3)
        
        self.class_layer = nn.Sequential(
            #nn.MaxPool3d(kernel_size=7, stride=7),
            #nn.flatten(),
            nn.Linear(1024, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_classes),
            )
    
    def _make_res_layer(self, in_c, pad):
        res_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.BatchNorm3d(in_c),
            nn.Conv3d(in_c, in_c, kernel_size=3, stride=1,padding=pad),
            nn.LeakyReLU(),
            nn.BatchNorm3d(in_c),
            nn.Conv3d(in_c, in_c, kernel_size=3, stride=1,padding=pad),
        )
        return res_layer
    
    def _make_bridge_layer(self, in_c, out_c, pad):
        bridge_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.BatchNorm3d(in_c),
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=2,padding=pad),
        )
        return bridge_layer

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        
        x_res = self.res_layer2(x)
        x=x+x_res
        
        x_res = self.res_layer3(x)
        x=x+x_res
        
        x=self.bridge_layer4(x)
        
        x_res = self.res_layer5(x)
        x=x+x_res
        
        x_res = self.res_layer6(x)
        x=x+x_res
        
        x=self.bridge_layer7(x)
        
        x_res = self.res_layer8(x)
        x=x+x_res
        
        x_res = self.res_layer8(x)
        x=x+x_res
                
        x=self.pool(x) 
        x = x.view(x.size(0), -1)
        #print(x.size())
        
        x = self.class_layer(x)
        
        return x#,x1

