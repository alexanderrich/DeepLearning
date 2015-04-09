require("io")
require("os")
require("torch")

torch.manualSeed(123)
data = torch.load("/scratch/courses/DSGA1008/A3/data/train.t7b")

classes = (#data.index)[1]
trialsperclass = (#data.index)[2]
perm = torch.randperm(trialsperclass)

traintrials = torch.floor(trialsperclass * .9)

traindata = {}

traindata.content = data.content

valdata = {}
valdata.content = data.content


index = {}
for i = 1, classes do
   index[i] = torch.Tensor(traintrials, 1)
   for j = 1, traintrials do
      index[i][j][1] = data.index[{i,perm[j]}]
   end
end
traindata.index = index
length = {}
for i = 1, classes do
   length[i] = torch.Tensor(traintrials, 1)
   for j = 1, traintrials do
      length[i][j][1] = data.length[{i,perm[j]}]
   end
end
traindata.length = length


index = {}
for i = 1, classes do
   index[i] = torch.Tensor(trialsperclass - traintrials, 1)
   for j = traintrials + 1, trialsperclass do
      index[i][j - traintrials][1] = data.index[{i,perm[j]}]
   end
end
valdata.index = index
length = {}
for i = 1, classes do
   length[i] = torch.Tensor(trialsperclass - traintrials, 1)
   for j = traintrials + 1, trialsperclass do
      length[i][j - traintrials][1] = data.length[{i,perm[j]}]
   end
end
valdata.length = length
torch.save("traindata.t7b", traindata)
torch.save("valdata.t7b", valdata)
