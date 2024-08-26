from models.gcn.stgc import *
from models.gcn.graph import Graph
from models.gcn.create_graph import GraphBuilder
import torch
import torch.nn as nn
import numpy as np
from models.mha import MultiHeadedAttention

class IdentityEncoderGCN(nn.Module):
    def __init__(self, cfg = None):
        super(IdentityEncoderGCN, self).__init__()
        self.cfg = cfg

        builder = GraphBuilder()
        edges = builder.create_face_graph()
        self.graph64 = Graph(64, edges, 57, max_hop = 3, k = 3)
        self.ca64 = torch.tensor(self.graph64.A, dtype=torch.float32, requires_grad=False)
        self.fc = nn.Linear(128, 1536)
        self.conv2d = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (2, 1))
        self.lrelu = nn.LeakyReLU()
        self.norm = nn.BatchNorm2d(256)

    def forward(self, x):
        x = x.view(-1, 128)
        x = self.fc(x).view(-1, 512, 3, 1)
        x = self.lrelu(self.norm(self.conv2d(x)))
        return x

class Decoder(nn.Module):

    def __init__(self, device = None, num_class = 12, dropout = 0.1, train_phase=True, num_joints=64):
        super(Decoder, self).__init__()

        ##############################
        ####GRAPHS INITIALIZATIONS####
        ##############################

        cols0 = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 20 ,21, 22, 23, 24, 25, 28, 30, 32, 33, 34, 48, 50, 62, 52, 54, 56, 66, 58, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 68]
        cols1 = [1, 3, 5, 7, 10, 11, 14, 15, 20, 22, 25, 28, 31, 36, 40, 42]
        cols2 = [9, 10, 11, 12, 13, 14, 15]
        cols3 = [6]

        edges = np.load("models/edges.npy")
        edges = [tuple(e) for e in edges]
        edges += [(68, 49), (68, 50), (68, 51), (68, 52), (68, 53), (68, 58), (68, 57), (68, 56), (68, 55), (68, 59), (68, 18), (68, 19), (68, 20), (68, 23), (68, 24), (68, 25), (18, 49), (19, 50), (20, 51), (23, 51), (24, 52), (25, 53), (33, 51), (33, 68), (32, 50), (34, 52), (37, 49), (38, 50), (43, 52), (44, 53), (68, 61), (68, 62), (68, 63), (68, 65), (68, 66), (68, 67)]

        self.graph69 = Graph(69, edges, 68, max_hop = 1, k = 2)
        self.ca69 = torch.tensor(self.graph69.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a69 = torch.tensor(self.graph69.getA(cols0), dtype=torch.float32, requires_grad=False).to(device)
        _, l0 = self.graph69.getLowAjd(cols0)

        self.graph43 = Graph(43, l0, 42, max_hop = 1, k = 2)
        self.ca43 = torch.tensor(self.graph43.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a43 = torch.tensor(self.graph43.getA(cols1), dtype=torch.float32, requires_grad=False).to(device)
        _, l1 = self.graph43.getLowAjd(cols1)

        self.graph16 = Graph(16, l1, 15, max_hop = 1, k = 2)
        self.ca16 = torch.tensor(self.graph16.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a16 = torch.tensor(self.graph16.getA(cols2), dtype=torch.float32, requires_grad=False).to(device)
        _, l2 = self.graph16.getLowAjd(cols2)

        self.graph7 = Graph(7, l2, 6, max_hop = 1, k = 2)
        self.ca7 = torch.tensor(self.graph7.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a7 = torch.tensor(self.graph7.getA(cols3), dtype=torch.float32, requires_grad=False).to(device)
        _, l3 = self.graph7.getLowAjd(cols3)

        self.graph1 = Graph(1, l3, 0)
        self.ca1 = torch.tensor(self.graph1.A, dtype=torch.float32, requires_grad=False).to(device)
        ##############################
        #############END##############
        ##############################
        self.num_class = num_class
        self.num_joints = num_joints
        self.device = device
        self.train_phase = train_phase

        self.embed = nn.Embedding(self.num_class,512)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
        self.act = self.lrelu

        self.normx0 = nn.BatchNorm2d(512)
        self.norm0 = nn.BatchNorm2d(256)
        self.norm1 = nn.BatchNorm2d(512)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(64)
        self.norm5 = nn.BatchNorm2d(32)
        self.norm6 = nn.BatchNorm2d(16)
        self.norm7 = nn.BatchNorm2d(8)
        self.norm8 = nn.BatchNorm2d(4)
        self.norm9 = nn.BatchNorm2d(2)

        ########STGCN#######
        self.gcnx0 = st_gcn(768, 512, (1, self.ca1.size(0)))
        self.gcn0 = st_gcn(512, 256, (1, self.ca1.size(0)))
        #self.gcn1 = st_gcn(640, 512, (1,self.ca7.size(0)))
        self.gcn2 = st_gcn(256, 128, (1,self.ca7.size(0)))
        #self.gcn3 = st_gcn(256, 128, (3,self.ca16.size(0)))
        self.gcn4 = st_gcn(128, 64, (3,self.ca16.size(0)))
        #self.gcn5 = st_gcn(64, 32, (7,self.ca43.size(0)))
        self.gcn6 = st_gcn(64, 16, (7,self.ca43.size(0)))
        #self.gcn7 = st_gcn(16, 8, (15,self.ca69.size(0)))
        self.gcn8 = st_gcn(16, 4, (15,self.ca69.size(0)))
        self.gcn9 = st_gcn(4, 2, (15,self.ca69.size(0)))
        #########END##########

        #mid-gcns
        self.gcnm1 = st_gcn(256, 256, (1,self.ca7.size(0)))
        self.gcnm2 = st_gcn(64, 64, (3,self.ca16.size(0)))
        self.gcnm3 = st_gcn(16, 16, (7,self.ca43.size(0)))
        self.gcnm4 = st_gcn(4, 4, (15, self.ca69.size(0)))

        #######GRAPH-UPSAMPLING########
        self.ups1 = UpSampling(1,7,self.a7,1024)
        self.ups2 = UpSampling(7,16,self.a16,256)
        self.ups3 = UpSampling(16,43,self.a43,64)
        self.ups4 = UpSampling(43,69,self.a69,16)
        ###############END##############

        #######TEMPORAL-UPSAMPLING########
        #self.uptx00 = nn.ConvTranspose2d(768,768,(2,1),stride=(2,1))
        self.uptx0 = nn.ConvTranspose2d(768,768,(2,1),stride=(2,1))
        self.upt0 = nn.ConvTranspose2d(512,512,(2,1),stride=(2,1))
        self.upt1 = nn.ConvTranspose2d(256,256,(2,1),stride=(2,1))
        self.upt2 = nn.ConvTranspose2d(128,128,(2,1),stride=(2,1))
        self.upt3 = nn.ConvTranspose2d(64,64,(2,1),stride=(2,1))
        self.upt4 = nn.ConvTranspose2d(32,32,(2,1),stride=(2,1))
        self.upt5 = nn.ConvTranspose2d(16,16,(2,1),stride=(2,1))
        self.upt6 = nn.ConvTranspose2d(8,8,(2,1),stride=(2,1))
        ###############END##############
        
        self.resx0 = nn.Conv2d(768, 512, kernel_size = (1, 1), stride = 1)
        self.resx = nn.Conv2d(512, 256, kernel_size = (1, 1), stride = 1)
        self.res1 = nn.Conv2d(256, 128, kernel_size = (1, 1), stride = 1)
        self.res2 = nn.Conv2d(128, 64, kernel_size = (1, 1), stride = 1)
        self.res3 = nn.Conv2d(64, 16, kernel_size = (1, 1), stride = 1)
        self.res4 = nn.Conv2d(16, 4, kernel_size = (1, 1), stride = 1)

        self.projection_layer = torch.nn.Linear(38, 8).to(device)

        self.normm1 = nn.BatchNorm2d(256)
        self.normm2 = nn.BatchNorm2d(128)
        self.normm3 = nn.BatchNorm2d(64)
        self.normm4 = nn.BatchNorm2d(16)

    def forward(self, encoded_features, s_feature, z):

        z = z.unsqueeze(-1)

        #LR x BN x GCNST x UPT x GCNRES
        #768 -> 512
        resx0 = self.resx0(z)
        resx0 = torch.cat([resx0, resx0], dim = 2)
        out0 = self.act(self.normx0(self.gcnx0(self.uptx0(z),self.ca1)))
        out0 += resx0

        #512 -> 256
        resx = self.resx(out0)
        resx = torch.cat([resx, resx], dim = 2)
        out1 = self.act(self.norm0(self.gcn0(self.upt0(out0),self.ca1)))
        out1 += resx

        #256 -> 128
#        out2 = self.act(self.norm1(self.gcn1(self.ups1(out1),self.ca7)))
        out2 = self.normm1(self.ups1(out1))
        res1 = self.res1(out2)
        res1 = torch.cat([res1, res1], dim = 2)
        out2 = self.dropout(self.act(self.norm2(self.gcn2(self.upt1(out2),self.ca7))))
        out2 += res1

        #128 -> 64
#        out3 = self.act(self.norm3(self.gcn3(self.ups2(out2),self.ca16)))
        out3 = self.normm2(self.ups2(out2))
        res2 = self.res2(out3)
        res2 = torch.cat([res2, res2], dim = 2)
        out3 = self.dropout(self.act(self.norm4(self.gcn4(self.upt2(out3),self.ca16))))
        out3 += res2

        #64 -> 16
#        out4 = self.act(self.norm5(self.gcn5(self.ups3(out3),self.ca43)))
        out4 = self.normm3(self.ups3(out3))
        res3 = self.res3(out4)
        res3 = torch.cat([res3, res3], dim = 2)
        out4 = self.dropout(self.act(self.norm6(self.gcn6(self.upt3(out4),self.ca43))))
        out4 += res3

        #16 -> 4
#        out5 = self.act(self.norm7(self.gcn7(self.ups4(out4), self.ca69)))
        out5 = self.normm4(self.ups4(out4))
        res4 = self.res4(out5)
        out5 = self.dropout(self.act(self.norm8(self.gcn8(out5, self.ca69))))
        out5 += res4

        #4 -> 2
        out6 = self.gcn9(out5, self.ca69)
        return out6
    
class Discriminator(nn.Module):

    def __init__(self,device,num_class = 12,size_sample = 64,num_joints=25):
        super(Discriminator,self).__init__()

        ##############################
        ####GRAPHS INITIALIZATIONS####
        ##############################

        cols1 = [1, 3, 5, 7, 10, 11, 14, 15, 20, 22, 25, 28, 31, 36, 40, 42]
        cols2 = [9, 10, 11, 12, 13, 14, 15]
        cols3 = [6]

        edges = np.load("edges.npy")
        edges = [tuple(e) for e in edges]
        edges += [(42, 36), (42, 40), (42, 37), (42, 35), (42, 20), (42, 34), (42, 38), (42, 41), (42, 39), (12, 13), (25, 28), (10, 42), (11, 42), (22, 42), (25, 42), (28, 42), (31, 42), (14, 42)]

        self.graph43 = Graph(43, edges, 42, max_hop = 1, k = 2)
        self.ca43 = torch.tensor(self.graph43.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a43 = torch.tensor(self.graph43.getA(cols1), dtype=torch.float32, requires_grad=False).to(device)
        _, l1 = self.graph43.getLowAjd(cols1)

        self.graph16 = Graph(16, l1, 15, max_hop = 1, k = 2)
        self.ca16 = torch.tensor(self.graph16.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a16 = torch.tensor(self.graph16.getA(cols2), dtype=torch.float32, requires_grad=False).to(device)
        _, l2 = self.graph16.getLowAjd(cols2)

        self.graph7 = Graph(7, l2, 6, max_hop = 1, k = 2)
        self.ca7 = torch.tensor(self.graph7.A, dtype=torch.float32, requires_grad=False).to(device)
        self.a7 = torch.tensor(self.graph7.getA(cols3), dtype=torch.float32, requires_grad=False).to(device)
        _, l3 = self.graph7.getLowAjd(cols3)

        self.graph1 = Graph(1, l3, 0)
        self.ca1 = torch.tensor(self.graph1.A, dtype=torch.float32, requires_grad=False).to(device)

        ##############################
        #############END##############
        ##############################

        self.size_sample = size_sample
        self.num_joints = num_joints
        self.device = device
        self.num_class = num_class

        self.embed = nn.Embedding(self.num_class,self.num_joints)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout()

        self.act = nn.Tanh()

        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(256)
        self.norm4 = nn.BatchNorm2d(512)

        ########STGCN#######
        self.gcn0 = st_gcn(3,2,(7,self.ca43.size(0)))
        self.gcn1 = st_gcn(2,32,(7,self.ca43.size(0)))
        self.gcn2 = st_gcn(32,64,(3,self.ca16.size(0)))
        self.gcn3 = st_gcn(64,128,(3,self.ca16.size(0)))
        self.gcn4 = st_gcn(128,256,(1,self.ca7.size(0)))
        self.gcn5 = st_gcn(256,512,(1,self.ca7.size(0)))
        self.gcn6 = st_gcn(512, 1,(1,self.ca1.size(0)))
        #########END##########

        #######GRAPH-DOWNSAMPLING########
        self.dws1 = DownSampling(43,16,self.a43,64)
        self.dws2 = DownSampling(16,7,self.a16,256)
        self.dws3 = DownSampling(7,1,self.a7,1)
        ###############END##############

        #######TEMPORAL-DOWNSAMPLING########
        self.dwt1 = nn.Conv2d(32,32,(int(self.size_sample/2)+1,1))
        self.dwt2 = nn.Conv2d(64,64,(int(self.size_sample/4)+1,1))
        self.dwt3 = nn.Conv2d(128,128,(int(self.size_sample/8)+1,1))
        self.dwt4 = nn.Conv2d(256,256,(int(self.size_sample/16)+1,1))
        self.dwt5 = nn.Conv2d(512,512,(int(self.size_sample/32)+1,1))
        self.dwt6 = nn.Conv2d(1,1,(int(self.size_sample/32),1))      
        ###############END##############


    def forward(self,x):

        aux = self.lrelu(self.dwt1(self.gcn1(x,self.ca43)))
        aux = self.lrelu(self.norm1(self.dws1(self.dwt2(self.gcn2(aux,self.ca43)))))
        aux = self.lrelu(self.norm2(self.dwt3(self.gcn3(aux,self.ca16))))
        aux = self.lrelu(self.norm3(self.dws2(self.dwt4(self.gcn4(aux,self.ca16)))))
        aux = self.lrelu(self.norm4(self.dws3(self.dwt5(self.gcn5(aux,self.ca7)))))
        aux = self.dwt6(self.gcn6(aux,self.ca1))
    
        return self.sigmoid(aux)


def main():

    device = torch.device("cpu")
    decoder = Discriminator(device = device)
    x = torch.randn(1, 2, 64, 43)
    decoder(x)

if __name__ == "__main__":
    main()
