import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

t = torch.arange(6,dtype = torch.float32)

data_loader = DataLoader(t)

for item in data_loader:
    print(item)
    
data_loader = DataLoader(t,batch_size=3,drop_last=False)
for i,batch in enumerate(data_loader):
    print(f'batch {i}:', batch)

torch.manual_seed(1)
t_x = torch.rand([4,3],dtype = torch.float32)
t_y = torch.arange(4)

class JoinDataset(Dataset) :
    def __init__(self,x,y) :
        self.x = x
        self.y = y
        
    def __len__(self) :
        return len(self.x)
    
    def __getitem__(self,idx) :
        return self.x[idx],self.y[idx]

joint_dataset = TensorDataset(t_x,t_y)
for example in joint_dataset:
    print('x: ', example[0], 'y: ',example[1])
    
torch.manual_seed(1)
data_loader = DataLoader(dataset=joint_dataset,batch_size=2,shuffle=True)
for i, batch in enumerate(data_loader,1):
    print(f'batch {i}:', 'x:',batch[0],'\n y:',batch[1])
