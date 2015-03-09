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



-- load trained model
model = torch.load('/home/asr443/DeepLearning/A2/model.net')
print(model)
model:cuda()
model:evaluate()

-- for each test image, get model prediction and add to predictions
datastring = 'Id,Category\n'
for t = 1, testData:size() do
   local input = testData.data[t]
   input = input:cuda()
   local pred = model:forward(input)
   -- use max function to get index of most-activated output node and print to datastring
   garbage, argmax = torch.max(pred,1)
   datastring = datastring .. t .. ', ' .. argmax[1] .. '\n'
end

-- save test results
file = io.open("predictions.csv", "w")
io.output(file)
io.write(datastring)
io.close(file)

   
