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



 -- Convert all images to HSV ---------------------
for i = 1,trainData:size() do
    image.rgb2hsv(trainData.data[i], trainData.data[i]);
end
for i = 1,valData:size() do
    image.rgb2hsv(valData.data[i], valData.data[i]);
end
for i = 1,trainData:size() do
    image.rgb2hsv(testData.data[i], testData.data[i]);
end




-- per channel normalization ---------------------
if opt.colorNormalize == 'channel' then  
  print '==> per channel normalization'
  mean = {}
  std = {}
  for i = 2,3 do -- only S and V channel
      -- use mean and std from training data    
      mean[i] = trainData.data[{ {},i,{},{} }]:mean()
      std[i] = trainData.data[{ {},i,{},{} }]:std()
      
      trainData.data[{ {},i,{},{} }]:add(-mean[i])
      trainData.data[{ {},i,{},{} }]:div(std[i])

      valData.data[{ {},i,{},{} }]:add(-mean[i])
      valData.data[{ {},i,{},{} }]:div(std[i])

      testData.data[{ {},i,{},{} }]:add(-mean[i])
      testData.data[{ {},i,{},{} }]:div(std[i])
  end
end

-- per pixel normalization ---------------------
if opt.colorNormalize == 'pixel' then  
  print '==> per pixel normalization'
  pixel_mean = torch.Tensor(3,96,96)
  pixel_std = torch.Tensor(3,96,96)
  for i = 2,3 do -- only S and V channel
      for j = 1,96 do
          for k = 1,96 do
            -- use mean and std from training data
            pixel_mean[{ i,j,k }] = trainData.data[{ {},i,j,k }]:mean()
            pixel_std[{ i,j,k }] = trainData.data[{ {},i,j,k }]:std()

            trainData.data[{ {},i,j,k }]:add(-pixel_mean[{ i,j,k }])
            trainData.data[{ {},i,j,k }]:div(pixel_std[{ i,j,k }])

            valData.data[{ {},i,j,k }]:add(-pixel_mean[{ i,j,k }])
            valData.data[{ {},i,j,k }]:div(pixel_std[{ i,j,k }])

            testData.data[{ {},i,j,k }]:add(-pixel_mean[{ i,j,k }])
            testData.data[{ {},i,j,k }]:div(pixel_std[{ i,j,k }])
          end
      end
  end
end

