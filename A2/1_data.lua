-- loading data
if opt.dataSource == 'mat' then
   matio = require 'matio'
end

-- if opt.size == 'full' then
--    print '==> using regular, full training data'
--    lsize = 4500
--    valsize = 500
-- elseif opt.size == 'small' then
--    print '==> using reduced training data, for fast experiments'
--    lsize = 500
-- end
lsize = 4500
valsize = 500
Xsize = lsize + valsize
testsize = 8000
unlsize = 100000
trsize = lsize + unlsize
numcats = 10

if opt.dataSource == 'mat' then
   mat = matio.load(opt.dataDir ..'/matlab/train.mat')
   mat.X = mat.X:double()
   mat.X = nn.Reshape(3,96,96):forward(mat.X)
   mat.y = mat.y[{{},1}]
   unl = matio.load(opt.dataDir ..'/matlab/unlabled_small.mat')
   mat.U = unl.X:double()
   mat.U = nn.Reshape(3,96,96):forward(mat.U)

else
   mat = {}
   data_fd = torch.DiskFile(opt.dataDir .. '/binary/train_X.bin', "r", true)
   data_fd:binary():littleEndianEncoding()
   mat.X = torch.ByteTensor(Xsize,3,96,96)
   data_fd:readByte(mat.X:storage())
   mat.X = mat.X:float()

   unl_fd = torch.DiskFile(opt.dataDir .. '/binary/unlabeled_X.bin', "r", true)
   unl_fd:binary():littleEndianEncoding()
   mat.U = torch.ByteTensor(unlsize,3,96,96)
   unl_fd:readByte(mat.U:storage())
   mat.U = mat.U:float()

   labels_fd = torch.DiskFile(opt.dataDir .. '/binary/train_y.bin', "r", true)
   labels_fd:binary():littleEndianEncoding()
   mat.y = torch.ByteTensor(Xsize)
   labels_fd:readByte(mat.y:storage())
   mat.y = mat.y:float()
end
mat.X = mat.X:transpose(3,4)
mat.U = mat.U:transpose(3,4)

-- use a random index to select validation set from training data
idtrain = torch.randperm(5000)
trainX = torch.Tensor(lsize, 3,96,96)
trainY = torch.Tensor(lsize)
for i=1,lsize do
   trainX[{i,{},{},{}}] = mat.X[{idtrain[i],{},{},{}}]
   trainY[i] = mat.y[idtrain[i]]
end
valdataX = torch.Tensor(valsize, 3,96,96)
valdataY = torch.Tensor(valsize)
for i=1,valsize do
   valdataX[{i,{},{},{}}] = mat.X[{idtrain[lsize+i],{},{},{}}]
   valdataY[i] = mat.y[idtrain[lsize+i]]
end
trainX = torch.cat(trainX, mat.U, 1)
traindataX = torch.Tensor(trsize, 3,96,96)
traindataY = torch.Tensor(trsize, numcats)
idX = torch.randperm(trsize)
for i=1,trsize do
  traindataX[{i,{},{},{}}] = trainX[{idX[i],{},{},{}}]
  if idX[i] <= lsize then  
    traindataY[{i, trainY[idX[i]]}] = 1
  else
    traindataY[{i,{}}] = torch.ones(numcats) * .1
  end
end

trainData = {
   data = traindataX,
   labels = traindataY,
   size = function() return trsize end
}

valData = {
   data = valdataX,
   labels = valdataY,
   size = function() return valsize end
}

-- loading test data
if opt.dataSource == 'mat' then
   mat = matio.load(opt.dataDir .. '/matlab/test.mat')
   mat.X = mat.X:double()
   mat.X = nn.Reshape(3,96,96):forward(mat.X)
   mat.y = mat.y[{{},1}]
else
   mat = {}
   data_fd = torch.DiskFile(opt.dataDir .. '/binary/test_X.bin', "r", true)
   data_fd:binary():littleEndianEncoding()
   mat.X = torch.ByteTensor(8000,3,96,96)
   data_fd:readByte(mat.X:storage())
   mat.X = mat.X:float()

   labels_fd = torch.DiskFile(opt.dataDir .. '/binary/test_y.bin', "r", true)
   labels_fd:binary():littleEndianEncoding()
   mat.y = torch.ByteTensor(8000)
   labels_fd:readByte(mat.y:storage())
   mat.y = mat.y:float()
end

mat.X = mat.X:transpose(3,4)

testData = {
   data = mat.X,
   labels = mat.y, -- tensor becomes number (for ClassNLL)
   size = function() return testsize end
}



-- Convert all images to YUV
for i = 1,trainData:size() do
   image.rgb2yuv(trainData.data[i], trainData.data[i]);
end
for i = 1,valData:size() do
   image.rgb2yuv(valData.data[i], valData.data[i]);
end
for i = 1,testData:size() do
   image.rgb2yuv(testData.data[i], testData.data[i]);
end



-- per channel mean substraction
mean = {} -- save for later
std = {}
pixelmeans = torch.Tensor(3,96,96)
for i = 1,3 do
    mean[i] = trainData.data[{ {},i,{},{} }]:mean()
    trainData.data[{ {},i,{},{} }]:add(-mean[i])
    -- normalizing standard deviation as well, can't imagine this hurts...
    std[i] = trainData.data[{ {},i,{},{} }]:std()
    trainData.data[{ {},i,{},{} }]:div(std[i])

    for j = 1,96 do
       for k = 1,96 do
          pixmean = trainData.data[{{},i,j,k}]:mean()
          trainData.data[{{},i,j,k}]:add(-pixmean)
          pixelmeans[{i,j,k}] = pixmean
       end
    end
end

-- mean subtract and normalize for validation and test sets
for i = 1,3 do
    valData.data[{ {},i,{},{} }]:add(-mean[i])
    valData.data[{ {},i,{},{} }]:div(std[i])
    for j = 1,96 do
       for k = 1,96 do
          valData.data[{{},i,j,k}]:add(-pixelmeans[{i,j,k}])
       end
    end
end


for i = 1,3 do
    testData.data[{ {},i,{},{} }]:add(-mean[i])
    testData.data[{ {},i,{},{} }]:div(std[i])
    for j = 1,96 do
       for k = 1,96 do
          testData.data[{{},i,j,k}]:add(-pixelmeans[{i,j,k}])
       end
    end
end

