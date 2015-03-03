-- loading data
mat = matio.load('stl10_matlab/train.mat')
mat.X = mat.X:double()
mat.X = nn.Reshape(3,96,96):forward(mat.X)
mat.X = mat.X:transpose(3,4)

if opt.size == 'full' then
   print '==> using regular, full training data'
   trsize = 5000
elseif opt.size == 'small' then
   print '==> using reduced training data, for fast experiments'
   trsize = 500
end


trainData = {
    data = mat.X,
    labels = mat.y[{{},1}], -- tensor becomes number (for ClassNLL)
    size = function() return trsize end
}


-- Convert all images to YUV
for i = 1,trainData:size() do
    image.rgb2yuv(trainData.data[i], trainData.data[i]);
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
