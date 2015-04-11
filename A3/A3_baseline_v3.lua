require 'torch'
require 'nn'
require 'optim'

ffi = require('ffi')



-- The self made module
local TemporalLogExpPooling, parent = torch.class('nn.TemporalLogExpPooling', 'nn.Module')

function TemporalLogExpPooling:__init(kW, dW, beta)
   parent.__init(self)

   self.kW = kW
   self.dW = dW
   self.beta = beta

   self.indices = torch.Tensor()
end

function TemporalLogExpPooling:updateOutput(input)
   kW = self.kW
   dW = self.dW
   
   if (input:nDimension()==2) then -- no batches
       nInputFrame = input:size(1)
       inputFrameSize = input:size(2)
       nOutputFrame = math.floor((nInputFrame - kW) / dW + 1)

       output = torch.Tensor(nOutputFrame, inputFrameSize)
       for f = 1, inputFrameSize do -- per feature column
           i = 1
           for j = 1, nOutputFrame do -- per output element
               vec = input[{{},{f}}]
               subset = vec:narrow(1,i,kW)
               output[j][f] = temporalLogExpPooling(subset)
               i = i + dW
           end
       end
   end

   if (input:nDimension()==3) then -- with batches
       minibatchSize = input:size(1)
       nInputFrame = input:size(2)
       inputFrameSize = input:size(3)
       nOutputFrame = math.floor((nInputFrame - kW) / dW + 1)

       output = torch.Tensor(minibatchSize, nOutputFrame, inputFrameSize)
       for b = 1, minibatchSize do -- per batch element
           for f = 1, inputFrameSize do -- per feature column
               i = 1
               for j = 1, nOutputFrame do -- per output element
                   vec = input[b][{{},{f}}]
                   subset = vec:narrow(1,i,kW)
                   output[b][j][f] = temporalLogExpPooling(subset)
                   i = i + dW
               end
           end
       end
   end
    
    self.output = output
    
    return self.output
end

function TemporalLogExpPooling:updateGradInput(input, gradOutput)
    kW = self.kW
    dW = self.dW
    if (input:nDimension()==2) then -- no batches
        nInputFrame = input:size(1)
        inputFrameSize = input:size(2)
        nOutputFrame = gradOutput:size(1)

        gradInput = torch.Tensor(nInputFrame, inputFrameSize)
        gradInput = gradInput:fill(0)
        for f = 1, inputFrameSize do -- per feature column
            i = 1
            vec = input[{{},{f}}]
            subset = vec:narrow(1,i,kW)
            gradI = temporalLogExpPoolingGradient(subset)
            for s = 1, kW do -- per subset element
                ii = i+s-1
                gradInput[ii][f] = gradInput[ii][f] + gradI[s] -- update gradInput
            end
            i = i + dW
        end
    end
    if (input:nDimension()==3) then -- with batches
        minibatchSize = input:size(1)
        nInputFrame = input:size(2)
        inputFrameSize = input:size(3)
        nOutputFrame = gradOutput:size(2)

        gradInput = torch.Tensor(minibatchSize, nInputFrame, inputFrameSize)
        gradInput = gradInput:fill(0)
        for b = 1, minibatchSize do -- per batch element
            for f = 1, inputFrameSize do -- per feature column
                i = 1
                vec = input[b][{{},{f}}]
                subset = vec:narrow(1,i,kW)
                for s = 1, kW do -- per subset element
                    ii = i+s-1
                    if (subset[s] == temporalLogExpPooling(subset)) then
                         gradInput[b][ii][f] = gradInput[b][ii][f] + gradI[s] -- update gradInput
                    end
                end
                i = i + dW
            end
        end
    end
    
    self.gradInput = gradInput
    
    return self.gradInput
end

function TemporalLogExpPooling:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
-- End self made module

beta = 1
-- Does the temporal pooling operation
function temporalLogExpPooling(input)
    local n = input:size()
    local parens = input:exp():sum():div(n[1])
    tlep = math.log(parens)/beta
    return tlep
end

--Computes gradient for above function
function temporalLogExpPoolingGradient(input)
    sumInp = input:apply(expb):sum()
    grad = input:apply(expb):div(sumInp)
    return grad
end

--Subfunction for gradient
function expb(x)
    bx = - beta * x
    return math.exp(bx)
end

--- Parses and loads the GloVe word vectors into a hash table:
-- glove_table['word'] = vector
function load_glove(path, inputDim)
    
    local glove_file = io.open(path)
    local glove_table = {}

    local line = glove_file:read("*l")
    while line do
        -- read the GloVe text file one line at a time, break at EOF
        local i = 1
        local word = ""
        for entry in line:gmatch("%S+") do -- split the line at each space
            if i == 1 then
                -- word comes first in each line, so grab it and create new table entry
                word = entry:gsub("%p+", ""):lower() -- remove all punctuation and change to lower case
                if string.len(word) > 0 then
                    glove_table[word] = torch.zeros(inputDim, 1) -- padded with an extra dimension for convolution
                else
                    break
                end
            else
                -- read off and store each word vector element
                glove_table[word][i-1] = tonumber(entry)
            end
            i = i+1
        end
        line = glove_file:read("*l")
    end
    
    return glove_table
end

--- Here we simply encode each document as a fixed-length vector 
-- by computing the unweighted average of its word vectors.
-- A slightly better approach would be to weight each word by its tf-idf value
-- before computing the bag-of-words average; this limits the effects of words like "the".
-- Still better would be to concatenate the word vectors into a variable-length
-- 2D tensor and train a more powerful convolutional or recurrent model on this directly.
function preprocess_data(raw_data, wordvector_table, opt)
    
    local data = torch.zeros(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs), opt.inputDim, 1)
    local labels = torch.zeros(opt.nClasses*(opt.nTrainDocs + opt.nTestDocs))
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    local order = torch.randperm(opt.nClasses*(opt.nTrainDocs+opt.nTestDocs))
    
    for i=1,opt.nClasses do
        for j=1,opt.nTrainDocs+opt.nTestDocs do
            local k = order[(i-1)*(opt.nTrainDocs+opt.nTestDocs) + j]
            
            local doc_size = 1
            
            local index = raw_data.index[i][j]
            -- standardize to all lowercase
            local document = ffi.string(torch.data(raw_data.content:narrow(1, index, 1))):lower()
            
            -- break each review into words and compute the document average
            for word in document:gmatch("%S+") do
                if wordvector_table[word:gsub("%p+", "")] then
                    doc_size = doc_size + 1
                    data[k]:add(wordvector_table[word:gsub("%p+", "")])
                end
            end

            data[k]:div(doc_size)
            labels[k] = i
        end
    end

    return data, labels
end

function train_model(model, criterion, data, labels, test_data, test_labels, opt)

    parameters, grad_parameters = model:getParameters()
    
    -- optimization functional to train the model with torch's optim library
    local function feval(x) 
        local minibatch = data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, data:size(2)):clone()
        local minibatch_labels = labels:sub(opt.idx, opt.idx + opt.minibatchSize):clone()
        
        model:training()
        local minibatch_loss = criterion:forward(model:forward(minibatch), minibatch_labels)
        model:zeroGradParameters()
        model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
        
        return minibatch_loss, grad_parameters
    end
    
    for epoch=1,opt.nEpochs do
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
            print("epoch: ", epoch, " batch: ", batch)
        end

        local accuracy = test_model(model, test_data, test_labels, opt)
        print("epoch ", epoch, " error: ", accuracy)

    end
end

function test_model(model, data, labels, opt)
    
    model:evaluate()

    local pred = model:forward(data)
    local _, argmax = pred:max(2)
    local err = torch.ne(argmax:double(), labels:double()):sum() / labels:size(1)

    --local debugger = require('fb.debugger')
    --debugger.enter()

    return err
end

function main()

    -- Configuration parameters
    opt = {}
    -- change these to the appropriate data locations
    opt.glovePath = "glove.txt" -- path to raw glove data .txt file
    opt.dataPath = "train.t7b"
    -- word vector dimensionality
    opt.inputDim = 50 
    -- nTrainDocs is the number of documents per class used in the training set, i.e.
    -- here we take the first nTrainDocs documents from each class as training samples
    -- and use the rest as a validation set.
    opt.nTrainDocs = 10000
    opt.nTestDocs = 0
    opt.nClasses = 5
    -- SGD parameters - play around with these
    opt.nEpochs = 5
    opt.minibatchSize = 128
    opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 0.001
    opt.momentum = 0.1
    opt.idx = 1

    print("Loading word vectors...")
    local glove_table = load_glove(opt.glovePath, opt.inputDim)
    
    print("Loading raw data...")
    local raw_data = torch.load(opt.dataPath)
    
    print("Computing document input representations...")
    local processed_data, labels = preprocess_data(raw_data, glove_table, opt)
    
    -- split data into makeshift training and validation sets
    local training_data = processed_data:sub(1, opt.nClasses*opt.nTrainDocs, 1, processed_data:size(2)):clone()
    local training_labels = labels:sub(1, opt.nClasses*opt.nTrainDocs):clone()
    
    -- make your own choices - here I have not created a separate test set
    local test_data = training_data:clone() 
    local test_labels = training_labels:clone()

    -- construct model:
    model = nn.Sequential()
   
    -- if you decide to just adapt the baseline code for part 2, you'll probably want to make this linear and remove pooling
    model:add(nn.TemporalConvolution(1, 20, 10, 1))
    
    --------------------------------------------------------------------------------------
    -- Replace this temporal max-pooling module with your log-exponential pooling module:
    --------------------------------------------------------------------------------------
    -- model:add(nn.TemporalMaxPooling(3, 1))
    print("Using our new TemporalLogExpPooling module...")
    model:add(nn.TemporalLogExpPooling(3, 1))
    
    model:add(nn.Reshape(20*39, true))
    model:add(nn.Linear(20*39, 5))
    model:add(nn.LogSoftMax())

    criterion = nn.ClassNLLCriterion()
   
    train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)
    local results = test_model(model, test_data, test_labels)
    print(results)
end

main()
