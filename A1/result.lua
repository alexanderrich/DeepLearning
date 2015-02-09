require 'torch'
require 'image'
require 'nn'
require 'csvigo'

-- load test data and transpose rows/columns to take advantage of
-- row-major data storage
loaded = torch.load('test_32x32.t7', 'ascii')
testData = {
   data = loaded.X:transpose(3,4),
   labels = loaded.y[1],
   size = function() return 26032 end
}
testData.data = testData.data:float()

-- convert data from rgb to yuv spectrum
for i = 1,testData:size() do
   testData.data[i] = image.rgb2yuv(testData.data[i])
end


-- means and std's of 3 channels from the "extra" size training set,
-- used to normalize test data.
mean = {110.231, 1.76616, -0.30603}
std = {49.4773, 9.54416, 10.64495}
for i = 1,3 do
   testData.data[{ {},i,{},{} }]:add(-mean[i])
   testData.data[{ {},i,{},{} }]:div(std[i])
end

-- length-13 gaussian kernel used for local normalization of channels:
neighborhood = image.gaussian1D(13)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- Normalize all channels locally:
for c = 1,3 do
   for i = 1,testData:size() do
      testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
   end
end


-- load trained model
model = torch.load('model.net')
print(model)
model:evaluate()

-- for each test image, get model prediction and add to predictions
predictions = {Id={},Prediction={}}
for t = 1, testData:size() do
   local input = testData.data[t]
   input = input:double()
   local pred = model:forward(input)
   predictions.Id[t] = t
   -- use max function to get index of most-activated output node
   trash_var, predictions.Prediction[t] = torch.max(pred, 1)
   predictions.Prediction[t] = predictions.Prediction[t][1]
end

-- save test results
csvigo.save({data=predictions, path = "predictions.csv"})
   
