-- loading data
if opt.dataSource == 'mat' then
   matio = require 'matio'
end

if opt.dataSource == 'mat' then
   mat = matio.load(opt.dataDir ..'/matlab/train.mat')
   mat.X = mat.X:double()
   mat.X = nn.Reshape(3,96,96):forward(mat.X)
   mat.y = mat.y[{{},1}]

else
   mat = {}
   data_fd = torch.DiskFile(opt.dataDir .. '/binary/train_X.bin', "r", true)
   data_fd:binary():littleEndianEncoding()
   mat.X = torch.ByteTensor(5000,3,96,96)
   data_fd:readByte(mat.X:storage())
   mat.X = mat.X:float()

   labels_fd = torch.DiskFile(opt.dataDir .. '/binary/train_y.bin', "r", true)
   labels_fd:binary():littleEndianEncoding()
   mat.y = torch.ByteTensor(5000)
   labels_fd:readByte(mat.y:storage())
   mat.y = mat.y:float()
end
mat.X = mat.X:transpose(3,4)

-- if opt.size == 'full' then
--    print '==> using regular, full training data'
--    trsize = 4500
--    valsize = 500
-- elseif opt.size == 'small' then
--    print '==> using reduced training data, for fast experiments'
--    trsize = 500
-- end
trsize = 4500
valsize = 500
testsize = 8000


-- use a random index to select validation set from training data
idx = torch.randperm(5000)
traindataX = torch.Tensor(trsize, 3,96,96)
traindataY = torch.Tensor(trsize)
for i=1,trsize do
   traindataX[{i,{},{},{}}] = mat.X[{idx[i],{},{},{}}]
   traindataY[i] = mat.y[idx[i]]
end
valdataX = torch.Tensor(valsize, 3,96,96)
valdataY = torch.Tensor(valsize)
for i=1,valsize do
   valdataX[{i,{},{},{}}] = mat.X[{idx[trsize+i],{},{},{}}]
   valdataY[i] = mat.y[idx[trsize+i]]
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
mean = torch.Tensor(3) -- save for later
std = torch.Tensor(3)
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

torch.save('mean.dat', mean)
torch.save('std.dat', std)
torch.save('pixelmeans.dat', pixelmeans)

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

