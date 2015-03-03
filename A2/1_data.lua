-- loading data
mat = matio.load('stl10_matlab/train.mat')
mat.X = mat.X:double()
mat.X = nn.Reshape(3,96,96):forward(mat.X)
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
   traindataY[i] = mat.y[{idx[i],1}]
end
valdataX = torch.Tensor(valsize, 3,96,96)
valdataY = torch.Tensor(valsize)
for i=1,valsize do
   valdataX[{i,{},{},{}}] = mat.X[{idx[trsize+i],{},{},{}}]
   valdataY[i] = mat.y[{idx[trsize+i],1}]
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
mat = matio.load('stl10_matlab/test.mat')
mat.X = mat.X:double()
mat.X = nn.Reshape(3,96,96):forward(mat.X)
mat.X = mat.X:transpose(3,4)

testData = {
   data = mat.X,
   labels = mat.y[{{},1}], -- tensor becomes number (for ClassNLL)
   size = function() return testsize end
}



-- Convert all images to YUV
for i = 1,trainData:size() do
   image.rgb2yuv(trainData.data[i], trainData.data[i]);
end
for i = 1,valData:size() do
   image.rgb2yuv(valData.data[i], valData.data[i]);
end


-- per channel mean substraction
mean = {} -- save for later
std = {}
for i = 1,3 do
    mean[i] = trainData.data[{ {},i,{},{} }]:mean()
    trainData.data[{ {},i,{},{} }]:add(-mean[i])
    -- normalizing standard deviation as well, can't imagine this hurts...
    std[i] = trainData.data[{ {},i,{},{} }]:std()
    trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- mean subtract and normalize for validation and test sets
for i = 1,3 do
    valData.data[{ {},i,{},{} }]:add(-mean[i])
    valData.data[{ {},i,{},{} }]:div(std[i])
end


for i = 1,3 do
    testData.data[{ {},i,{},{} }]:add(-mean[i])
    testData.data[{ {},i,{},{} }]:div(std[i])
end
