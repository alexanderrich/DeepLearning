-- loading training data
print '==> loading training data'
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
   block = 1
   mat.X = torch.Tensor(5000*3*96*96)
   mat.y = torch.Tensor(5000)
   file = io.open(opt.dataDir .. '/binary/train_X.bin')
   i = 0
   while true do
      i = i+1
      byte = file:read(block)
      if not byte then break end
      mat.X[{i}] = tonumber(string.byte(byte))
   end
   mat.X = mat.X:double()
   mat.X = nn.Reshape(5000,3,96,96):forward(mat.X)

   file = io.open(opt.dataDir .. '/binary/train_y.bin')
   i = 0
   while true do
      i = i+1
      byte = file:read(block)
      if not byte then break end
      mat.y[{i}] = tonumber(string.byte(byte))
   end
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
print '==> shuffle, then split into training and validation set'
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
print '==> loading test data'
if opt.dataSource == 'mat' then
   mat = matio.load(opt.dataDir .. '/matlab/test.mat')
   mat.X = mat.X:double()
   mat.X = nn.Reshape(3,96,96):forward(mat.X)
   mat.y = mat.y[{{},1}]
else
   mat = {}
   block = 1
   mat.X = torch.Tensor(8000*3*96*96)
   mat.y = torch.Tensor(8000)
   file = io.open( opt.dataDir .. '/binary/test_X.bin')
   i = 0
   while true do
      i = i+1
      byte = file:read(block)
      if not byte then break end
      mat.X[{i}] = tonumber(string.byte(byte))
   end
   mat.X = mat.X:double()
   mat.X = nn.Reshape(8000,3,96,96):forward(mat.X)

   file = io.open( opt.dataDir .. '/binary/test_y.bin')
   i = 0
   while true do
      i = i+1
      byte = file:read(block)
      if not byte then break end
      mat.y[{i}] = tonumber(string.byte(byte))
   end
end

mat.X = mat.X:transpose(3,4)

testData = {
   data = mat.X,
   labels = mat.y, -- tensor becomes number (for ClassNLL)
   size = function() return testsize end
}



------------------------------------------
-- data augmenting
------------------------------------------
inflate_factor = opt.dataAugmentation

if inflate_factor == 1 then
  print '==> no data augmentation'
else 
  print ('==> augment train data set by ' .. inflate_factor)
  
  -- assign storage
  augmentData = {}
  augmentData.data = torch.Tensor(inflate_factor * trainData:size(), 3, 96, 96)
  augmentData.labels = torch.Tensor(inflate_factor * trainData:size())
  function augmentData:size() return augmentData.data:size()[1] end

  -- fill with original images
  for i = 1, augmentData:size() do
      -- if (math.fmod(i,5000)==0) then print("fill with original images: "..i) end
      original_i = math.fmod((i-1),trainData:size())+1
      augmentData.data[i] = augmentData.data[i]:copy(trainData.data[original_i])
      augmentData.labels[i] = trainData.labels[original_i]
  end

  -- transformation functions ---------------------

  -- return 1 or -1
  function rand_plus_minus() return( (math.random(0,1)-0.5)*2 ) end

  -- shift + crop + re-scale to original resolution
  function aug_translate(img)
      local right = rand_plus_minus() -- right=-1 for leftwards shift
      local x_shift = math.random(2,10) -- 2 to 10 pixels
      
      local down = rand_plus_minus() -- down=-1 for upwards shift
      local y_shift = math.random(2,10) -- 2 to 10 pixels
      
      image.crop(img, img,
          math.max(0, 0+(right*x_shift)),   math.max(0, 0+(down*y_shift)), 
          math.min(96, 96+(right*x_shift)), math.min(96, 96+(down*y_shift)))
  end

  -- horizontal flip
  function aug_hflip(img)
      image.hflip(img, img)
  end

  -- rotate + crop + re-scale to original resolution
  function aug_rotate(img, garbage)
      local theta = rand_plus_minus()*math.random(10, 34)/100 -- 6 to 20 degrees

      image.rotate(garbage, img, theta)

      theta = math.abs(theta)
      local r = 96*( (math.cos(theta) + math.sin(theta))-1 )/2 

      image.crop(img, garbage, r, r, 96-r, 96-r)
  end

  -- color and contrast2 from paper
  function aug_hsv_color_contrast2(img)
      -- h
      img[1]:add(rand_plus_minus()/10) -- "add between −0.1 and 0.1"
      -- s
      img[2]:pow(math.random(25,400)/100) -- "to a power between 0.25 and 4"
      img[2]:mul(math.random(70,140)/100) -- "multiply by a factor between 0.7 and 1.4"
      img[2]:add(rand_plus_minus()/10) -- "add between −0.1 and 0.1"
      -- v
      img[3]:pow(math.random(25,400)/100) -- "to a power between 0.25 and 4"
      img[3]:mul(math.random(70,140)/100) -- "multiply by a factor between 0.7 and 1.4"
      img[3]:add(rand_plus_minus()/10) -- "add between −0.1 and 0.1"
  end

  -- Convert all images to HSV ---------------------
  for i = 1,augmentData:size() do
      image.rgb2hsv(augmentData.data[i], augmentData.data[i]);
  end

  -- apply transformations ---------------------
  garbage = torch.Tensor(3,96,96)
  begin = trainData:size()+1 -- don't perform transformations on first block 
  for i = begin, augmentData:size() do
      -- if (math.fmod(i,5000)==0) then print(i) end
      if math.random(0,1)==1 then aug_translate(augmentData.data[i]) end -- down-right or up-left shift
      if math.random(0,1)==1 then aug_hflip(augmentData.data[i]) end -- horizontal flip
      if true                then aug_rotate(augmentData.data[i], garbage) end -- rotation
      if math.random(0,1)==1 then aug_hsv_color_contrast2(augmentData.data[i]) end -- color and contrast
  end

  -- overwrite trainData
  trainData = augmentData
  print '==> augmentation finished'
end

print ('==> trainData:size() = ' .. trainData:size())


-- OLD ----------

-- -- Convert all images to YUV
-- for i = 1,trainData:size() do
--    image.rgb2yuv(trainData.data[i], trainData.data[i]);
-- end
-- for i = 1,valData:size() do
--    image.rgb2yuv(valData.data[i], valData.data[i]);
-- end
-- for i = 1,testData:size() do
--    image.rgb2yuv(testData.data[i], testData.data[i]);
-- end



-- -- per channel mean substraction
-- mean = {} -- save for later
-- std = {}
-- for i = 1,3 do
--     mean[i] = trainData.data[{ {},i,{},{} }]:mean()
--     trainData.data[{ {},i,{},{} }]:add(-mean[i])
--     -- normalizing standard deviation as well, can't imagine this hurts...
--     std[i] = trainData.data[{ {},i,{},{} }]:std()
--     trainData.data[{ {},i,{},{} }]:div(std[i])
-- end

-- -- mean subtract and normalize for validation and test sets
-- for i = 1,3 do
--     valData.data[{ {},i,{},{} }]:add(-mean[i])
--     valData.data[{ {},i,{},{} }]:div(std[i])
-- end


-- for i = 1,3 do
--     testData.data[{ {},i,{},{} }]:add(-mean[i])
--     testData.data[{ {},i,{},{} }]:div(std[i])
-- end

