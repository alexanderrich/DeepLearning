require 'torch'
require 'image'
require 'cunn'

-- load test data and transpose rows/columns to take advantage of
-- row-major data storage

mat = {}
data_fd = torch.DiskFile('/scratch/courses/DSGA1008/A2/binary/test_X.bin', "r", true)
data_fd:binary():littleEndianEncoding()
mat.X = torch.ByteTensor(8000,3,96,96)
data_fd:readByte(mat.X:storage())
mat.X = mat.X:float()

mat.X = mat.X:transpose(3,4)

testData = {
   data = mat.X,
   size = function() return 8000 end
}

for i = 1,testData:size() do
   image.rgb2yuv(testData.data[i], testData.data[i]);
end

-- LOAD MEANS/STD'S/PIXEL MEANS
mean = torch.load('mean.dat')
std = torch.load('std.dat')
pixelmeans = torch.load('pixelmeans.dat')

for i = 1,3 do
    testData.data[{ {},i,{},{} }]:add(-mean[i])
    testData.data[{ {},i,{},{} }]:div(std[i])
    for j = 1,96 do
       for k = 1,96 do
          testData.data[{{},i,j,k}]:add(-pixelmeans[{i,j,k}])
       end
    end
end



modelpredictions = torch.Tensor(3,testData:size())
-- load trained model

for m = 1,5 do
   model = torch.load('/home/asr443/DeepLearning/A2/model' .. m .. '.net')
   model:cuda()
   model:evaluate()
   -- for each test image, get model prediction and add to predictions
   for t = 1, testData:size() do
      local input = testData.data[t]
      input = input:cuda()
      local pred = model:forward(input)
      -- use max function to get index of most-activated output node and print to datastring
      garbage, argmax = torch.max(pred,1)
      modelpredictions[{m,t}] = argmax[1]
   end
end
datastring = 'Id,Category\n'
voteCounts = torch.Tensor(10)
for t = 1, testData:size() do
   voteCounts:fill(0)
   
   for i = 1,5 do
      voteCounts[modelpredictions[{i,t}]] =  voteCounts[modelpredictions[{i,t}]] + 1
   end

   garbage, argmax = torch.max(voteCounts, 1)
   

   datastring = datastring .. t .. ', ' .. argmax[1] .. '\n'
end
   
-- save test results
file = io.open("predictions.csv", "w")
io.output(file)
io.write(datastring)
io.close(file)

   
