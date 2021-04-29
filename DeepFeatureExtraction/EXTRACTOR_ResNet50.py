import torch
import torchvision
#import skimage.io as io
import numpy as np
import torchvision.transforms as t
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import torchvision.models as model
#from sklearn.metrics import accuracy_score
torch.cuda.set_device(0)
#device=torch.device(#"cuda" if torch.cuda.is_available() else "cpu")
#print("device= ",device)

train_path="/content/drive/My Drive/SIPaKMeD/Train"
test_path="/content/drive/My Drive/SIPaKMeD/Test"
val_path="/content/drive/My Drive/SIPaKMeD/Validation"
plot_path="/content/drive/My Drive/SIPaKMeD"
snapshot_path="/content/drive/My Drive/SIPaKMeD"

#set_proper name
model_name='ResNet50'
batch_s = 100
transform=t.Compose([t.Resize((224,224)),
                     #t.RandomCrop((224,224)),
                     #t.RandomHorizontalFlip(),
                     #t.RandomVerticalFlip(),
                     #t.RandomAffine(degrees=(-180,180), translate=(0.1,0.1), scale=(0.9,1.1), shear=(-5,5)),
                     t.ToTensor()])
dset_train=torchvision.datasets.ImageFolder(root=train_path,transform=transform)

test_trans=t.Compose([t.Resize((224,224)),t.ToTensor()])
dset_test=torchvision.datasets.ImageFolder(root=test_path,transform=test_trans)
dset_val=torchvision.datasets.ImageFolder(root=val_path,transform=test_trans)

train_loader=torch.utils.data.DataLoader(dset_train,batch_size=batch_s,shuffle=True,num_workers=16)#,drop_last=True)
val_loader=torch.utils.data.DataLoader(dset_val,batch_size=batch_s,shuffle=False,num_workers=16)#,drop_last=True)
test_loader=torch.utils.data.DataLoader(dset_test,batch_size=batch_s,num_workers=16)#, drop_last=True)

num_classes = 7
#net=model.googlenet()

############################## MODEL : GOOGLENET ########################################








models = torchvision.models.resnet50(pretrained=True)
#net.fc = nn.Linear(net.fc.in_features,num_classes)




class MyModel(nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    img_modules = list(models.children())[:-1]
    self.ModelA = nn.Sequential(*img_modules)
    self.Linear1 = nn.Linear(1024, 256)
    self.relu = nn.ReLU()
    self.Linear2 = nn.Linear(256, 2)
    #self.softmax = nn.Softmax(dim = 1)
    self.Linear3 = nn.Linear(2048, 5, bias = True)
    #self.Linear4 = nn.Linear(1024, 2, bias = True)

  def forward(self, x):
    x = self.ModelA(x)
    x1 = torch.flatten(x, 1)
    x2 = self.Linear3(x1)
    #x2 = self.relu(x1)
    #x2 = self.Linear4(x2) 
    #x2 = self.softmax(x2)

    return  x1, x2







#net.fc=nn.Linear(1024,2,True)
net = MyModel()
net=net.cuda()
criterion=nn.CrossEntropyLoss()
params = net.parameters()
optimizer=torch.optim.Adam(net.parameters())
#optimizer = torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#opyimizer = torch.optim.Adagrad(net.parameters(), lr=0.005, lr_decay=0.01, weight_decay=0.005, initial_accumulator_value=0, eps=1e-10)
model_name1 = 'ResNet50'

load_model=snapshot_path+'/model_'+model_name+'.pth'
loaded_flag=False
if os.path.exists(load_model):
    checkpoint=torch.load(load_model)
    net.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    print("model loaded successfully")
    print('starting training after epoch: ',checkpoint['epoch'])
    loaded_flag=True
    

def plot(val_loss,train_loss):
    plt.title("Loss after epoch: {}".format(len(train_loss)))
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(list(range(len(train_loss))),train_loss,color="r",label="Train_loss")
    plt.plot(list(range(len(val_loss))),val_loss,color="b",label="Validation_loss")
    plt.legend()
    plt.savefig(os.path.join(plot_path,"loss_"+model_name+".png"))
#     plt.figure()
    plt.close()



val_interval=1
min_loss=99999
val_loss_gph=[]
train_loss_gph=[]

if loaded_flag:
    min_loss=checkpoint['loss']
    val_loss_gph=checkpoint["val_graph"]
    train_loss_gph=checkpoint["train_graph"]


########################## TRIAN ##################

def train(epoch=5):
  i=0
  global min_loss
  flag=True
  while i+1<=epoch and flag:
    print("Epoch {}".format(i+1 if not loaded_flag else i+1+checkpoint['epoch']))
    train_loss=0.0
    i+=1
    data1 = []
    correct=total=0
    #net = net.train()
    for (image,label) in train_loader:
      net.train()
      optimizer.zero_grad()
      outputs1, outputs2=net(image.cuda())
      #data1.append(outputs1)
      loss=criterion(outputs2 ,label.cuda())
      loss.backward()
      optimizer.step()
      train_loss+=loss.item()*image.size(0)
      _, predicted = torch.max(outputs2.data, 1)
      total += label.size(0)
      correct += (predicted == label.cuda()).sum().item()
    print("Train accuracy", (100*correct/total))
    train_loss_gph.append(train_loss/len(dset_train))
    #net = net.eval()
    
    if (i+1)%val_interval==0 or (i+1)==epoch:
        net.eval()
        with torch.no_grad():
          val_loss=0
          correct=total=0
          for (img_v,lab_v ) in val_loader:
            output_v1, output_v2=net(img_v.cuda())
            #data1.append(output_v1)
            #val_loss+=criterion(output_v2,lab_v.cuda())
            val_loss+=criterion(output_v2,lab_v.cuda())*img_v.size(0)
            _, predicted = torch.max(output_v2.data, 1)
            total += lab_v.size(0)
            correct += (predicted == lab_v.cuda()).sum().item()
          print("Val accuracy", (100*correct/total))
          val_loss_gph.append(val_loss/len(dset_val))
        
          if val_loss<min_loss:
            state={
                "epoch":i if not loaded_flag else i+checkpoint['epoch'],
                "model_state":net.cpu().state_dict(),
                "optimizer_state":optimizer.state_dict(),
                "loss":min_loss,
                "train_graph":train_loss_gph,
                "val_graph":val_loss_gph,
            }
            
            min_loss=val_loss
            torch.save(state,os.path.join(snapshot_path,"model_"+model_name+'.pth'))
            net.cuda()
          print("validation loss : {:.6f} ".format(val_loss/len(dset_val)))
    plot(val_loss_gph,train_loss_gph)
    print("Train loss : {:.6f}".format(train_loss/len(dset_train)))
    if i==epoch:
      flag=False
      break
  
train(50)


print("validation MIN loss obtained: {:.6f}".format(min_loss))
net=net.eval()
correct = 0
total = 0
data1 = []
with torch.no_grad():
      for data in train_loader:
          images, labels = data
          labels=labels.cuda()
          outputs1, outputs2 = net(images.cuda())
          data1.append(outputs1)
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
      100 * correct / total))

print("validation MIN loss obtained: {:.6f}".format(min_loss))
net=net.eval()
correct = 0
total = 0
data2 = []
with torch.no_grad():
      for data in val_loader:
          images, labels = data
          labels=labels.cuda()
          outputs1, outputs2 = net(images.cuda())
          data2.append(outputs1)
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
      100 * correct / total))


print("validation MIN loss obtained: {:.6f}".format(min_loss))
net=net.eval()
correct = 0
total = 0
data3 = []
with torch.no_grad():
      for data in test_loader:
          images, labels = data
          labels=labels.cuda()
          outputs1, outputs2 = net(images.cuda())
          data3.append(outputs1)
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
      100 * correct / total))

########### SAVING THE MODEL OF LAST EPOCH ############## 
state1 = {
    "Model_State": net.cpu().state_dict(),
    "Optimiser_State": optimizer.state_dict(),

}
torch.save(state1,os.path.join(snapshot_path,"model_"+model_name1+'.pth'))
'''
p = train(100)

temp=p
import csv
labels=[]
for i in range(len(temp[0].tolist()[0])):
  labels.append("Plain"+str(i+1))
with open ("/content/drive/My Drive/ND Sir's Project/HErlev_PlainNet.csv",'w+',newline='') as file:
  writer=csv.writer(file)
  writer.writerow(labels)
  for i in range(len(temp)):
    row=temp[i].tolist()[0]
    writer.writerow(row)
'''




############### LOADING THE MODEL SAVED AT LAST EPOCH ###########3

load_model1 = '/content/drive/My Drive/SIPaKMeD/model_ResNet50_last.pth'

if os.path.exists(load_model1):
    checkpoint=torch.load(load_model1)
    net.load_state_dict(checkpoint['Model_State'])
    optimizer.load_state_dict(checkpoint['Optimiser_State'])
    print("model loaded successfully")
    #print('starting training after epoch: ',checkpoint['epoch'])
    #loaded_flag=True


###### LOADING DATA WITH BATCH SIZE 1 ################

test_trans=t.Compose([t.Resize((224,224)),t.ToTensor()])
dset_train=torchvision.datasets.ImageFolder(root=train_path,transform=test_trans)


dset_test=torchvision.datasets.ImageFolder(root=test_path,transform=test_trans)
dset_val=torchvision.datasets.ImageFolder(root=val_path,transform=test_trans)


train_loader=torch.utils.data.DataLoader(dset_train,batch_size=1,shuffle=False,num_workers=16)#,drop_last=True)
val_loader=torch.utils.data.DataLoader(dset_val,batch_size=1,shuffle=False,num_workers=16)#,drop_last=True)
test_loader=torch.utils.data.DataLoader(dset_test,batch_size=1,num_workers=16)#, drop_last=True)

############### EXTRACTION OF FEATURES ############

net = net.cuda()

print("validation MIN loss obtained: {:.6f}".format(min_loss))
net=net.eval()
correct = 0
total = 0
data1 = []
with torch.no_grad():
      for data in train_loader:
          images, labels = data
          labels=labels.cuda()
          outputs1, outputs2 = net(images.cuda())
          data1.append(outputs1)
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
      100 * correct / total))

print("validation MIN loss obtained: {:.6f}".format(min_loss))
net=net.eval()
correct = 0
total = 0
data2 = []
with torch.no_grad():
      for data in val_loader:
          images, labels = data
          labels=labels.cuda()
          outputs1, outputs2 = net(images.cuda())
          data2.append(outputs1)
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
      100 * correct / total))


print("validation MIN loss obtained: {:.6f}".format(min_loss))
net=net.eval()
correct = 0
total = 0
data3 = []
with torch.no_grad():
      for data in test_loader:
          images, labels = data
          labels=labels.cuda()
          outputs1, outputs2 = net(images.cuda())
          data3.append(outputs1)
          _, predicted = torch.max(outputs2.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (
      100 * correct / total))



data_all = data1+data2+data3

####### LOADING THE CSV #############3

temp=data_all
import csv
labels=[]
for i in range(len(temp[0].tolist()[0])):
  labels.append("ResNet18"+str(i+1))
with open ("/content/drive/My Drive/SIPaKMeD/SIPaKMeD_ResNet18_b_original_.csv",'w+',newline='') as file:
  writer=csv.writer(file)
  writer.writerow(labels)
  for i in range(len(temp)):
    row=temp[i].tolist()[0]
    writer.writerow(row)
