import torch
import torch.nn.functional as F
from torch import nn


class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.layer1 = nn.Sequential( nn.Conv2d(3, 32, kernel_size=3) ,
            nn.MaxPool2d(2,2), nn.ReLU()) 
        
        self.layer2 = nn.Sequential( nn.Conv2d(32, 64, kernel_size=3) ,
            nn.MaxPool2d(2,2), nn.ReLU())  
        
        self.layer3 = nn.Sequential( nn.Conv2d(64, 64, kernel_size=3) ,
            nn.MaxPool2d(2,2), nn.ReLU()) 
        
        self.layer4 = nn.Sequential( nn.Conv2d(64, 128, kernel_size=2) ,
            nn.ReLU())
        
        self.fc1 = nn.Linear(1152, 256) 
        self.fc2 = nn.Linear(256, 136) #68Х2
        
    def forward(self, x):
        out = self.layer1(x) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1) 
        out = self.fc1(out)
        out = F.relu(out) 
        out = self.fc2(out) 
        return out
    

# https://www.researchgate.net/publication/319277818_Deep_Alignment_Network_A_Convolutional_Neural_Network_for_Robust_Face_Alignment
class DAN(nn.Module):
    def __init__(self):
        super(DAN, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1) ,
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.Conv2d(64, 64, kernel_size=3, padding=1) ,
                                    nn.ReLU(),
                                    nn.BatchNorm2d(64),
                                    nn.MaxPool2d(2,2)) 
        
        self.layer2 = nn.Sequential( nn.Conv2d(64, 128, kernel_size=3, padding=1) ,
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),
                                    nn.Conv2d(128, 128, kernel_size=3, padding=1) ,                       
                                    nn.ReLU(),
                                    nn.BatchNorm2d(128),
                                    nn.MaxPool2d(2,2))  
        
        self.layer3 = nn.Sequential( nn.Conv2d(128, 256, kernel_size=3, padding=1) ,
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.Conv2d(256, 256, kernel_size=3, padding=1) ,                       
                                    nn.ReLU(),
                                    nn.BatchNorm2d(256),
                                    nn.MaxPool2d(2,2)
                                    ) 
        
        self.layer4 = nn.Sequential( nn.Conv2d(256, 512, kernel_size=3, padding=1) ,
                                    nn.ReLU(),
                                    nn.BatchNorm2d(512),
                                    nn.Conv2d(512, 512, kernel_size=3, padding=1) ,                       
                                    nn.ReLU(),
                                    nn.BatchNorm2d(512),
                                    nn.MaxPool2d(2,2))  
        
        self.fc1 = nn.Linear(25088, 256) 
        self.fc2 = nn.Linear(256, 136) #68Х2
        
    def forward(self, x):
        out = self.layer1(x) 
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = torch.flatten(out, 1) 
        out = self.fc1(out)
        out = F.relu(out) 
        out = self.fc2(out) 
        return out