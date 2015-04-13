-- require('mobdebug').start()
-- require('mobdebug').off()

require 'torch'
require 'nn'
require 'optim'

ffi = require('ffi')

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
function preprocess_data(raw_data, wordvector_table, nDocs, opt)
    -- require('mobdebug').on()
    local data = torch.zeros(opt.nClasses*nDocs, opt.inputDim, 1)
    local labels = torch.zeros(opt.nClasses*nDocs)
    
    -- use torch.randperm to shuffle the data, since it's ordered by class in the file
    local order = torch.randperm(opt.nClasses*nDocs)
    
    for i=1,opt.nClasses do
        for j=1,nDocs do
            local k = order[(i-1)*nDocs + j]
            
            local doc_size = 1
            
            local index = raw_data.index[i][{j,1}]
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

function train_model(model, criterion, train_data, train_labels, test_data, test_labels, opt)

    parameters, grad_parameters = model:getParameters()
    
    -- optimization functional to train the model with torch's optim library
    local function feval(x) 
        local minibatch = train_data:sub(opt.idx, opt.idx + opt.minibatchSize, 1, train_data:size(2)):clone()
        local minibatch_labels = train_labels:sub(opt.idx, opt.idx + opt.minibatchSize):clone()
        
        model:training()
        local minibatch_loss = criterion:forward(model:forward(minibatch), minibatch_labels)
        model:zeroGradParameters()
        model:backward(minibatch, criterion:backward(model.output, minibatch_labels))
        
        return minibatch_loss, grad_parameters
    end

    local record_train_acc = torch.Tensor(opt.nEpochs)
    local record_test_acc = torch.Tensor(opt.nEpochs)
    for epoch=1,opt.nEpochs do
        model:training()
        local order = torch.randperm(opt.nBatches) -- not really good randomization
        for batch=1,opt.nBatches do
            opt.idx = (order[batch] - 1) * opt.minibatchSize + 1
            optim.sgd(feval, parameters, opt)
            -- print("epoch: ", epoch, " batch: ", batch)
        end

        local train_acc = test_model(model, train_data, train_labels, opt, epoch)
        local test_acc = test_model(model, test_data, test_labels, opt, epoch)
        record_train_acc[epoch] = train_acc
        record_test_acc[epoch] = test_acc
        print("epoch ", epoch, " test_acc: ", test_acc)
    end

    return record_train_acc, record_test_acc
end

function test_model(model, data, labels, opt, epoch)
    
    model:evaluate()

    local pred = model:forward(data)
    local pred_category = {}
    local acc = 0

    if opt.outputType == "categories" then
        local _, argmax = pred:max(2)
        acc = torch.eq(argmax:double(), labels:double()):sum() / labels:size(1)
        pred_category = argmax:squeeze()

    end
    if opt.outputType == "continuous" then
        pred_category = torch.round(pred)
        local correct = torch.eq(pred_category, labels)
        acc = torch.sum(correct) / labels:size(1) 

        -- local diff = torch.add(pred, -labels)
        -- local diff_abs = torch.abs(diff)
        -- local mae = torch.mean(diff_abs)
    end

    -- local debugger = require('fb.debugger')
    -- debugger.enter()
    -- local pred_category = temp:double()

    if opt.confusionMatrix then
        if epoch % 10 == 0 then
            -- This matrix records the current confusion across classes
            local confusion = optim.ConfusionMatrix(opt.nClasses)
            confusion:zero()
            confusion:batchAdd(pred_category, labels)
            print(confusion)
        end
    end

    return acc
end

function main()

    -- Configuration parameters
    opt = {}
    opt.loadPreprocessed = true
    opt.savePreprocessed = true
    -- word vector dimensionality
    opt.inputDim = 100
    opt.glovePath = "glove"..opt.inputDim..".txt" -- raw glove data file
    opt.dataPath = "traindata.t7b"
    opt.valdataPath = "valdata.t7b"
    -- nTrainDocs is the number of documents per class used in the training set
    opt.nTrainDocs = 117000 --130K-13K
    opt.nTestDocs = 13000 -- 13K
    opt.nClasses = 5
    -- SGD parameters - play around with these
    opt.nEpochs = 10
    opt.minibatchSize = 32
    opt.nBatches = math.floor(opt.nTrainDocs / opt.minibatchSize)
    opt.learningRate = 0.1
    opt.learningRateDecay = 0.00000001
    opt.momentum = 0.9
    opt.dropout = true
    opt.idx = 1
    opt.confusionMatrix = false
    opt.outputType = "categories" -- "categories" -- "continuous"

    local processed_traindata = {}
    local processed_valdata = {}
    local processed_trainlabels = {}
    local processed_vallabels = {}

    if opt.loadPreprocessed then
        print("Loading preprocessed data...")
        processed_traindata = torch.load("processed_traindata"..opt.inputDim..".t7b")
        processed_valdata = torch.load("processed_valdata"..opt.inputDim..".t7b")
        processed_trainlabels = torch.load("processed_trainlabels"..opt.inputDim..".t7b")
        processed_vallabels = torch.load("processed_vallabels"..opt.inputDim..".t7b")

    else
        print("Loading word vectors...")
        local glove_table = load_glove(opt.glovePath, opt.inputDim)
        
        print("Loading raw data...")''';'
        local raw_data = torch.load(opt.dataPath)
        local raw_valdata = torch.load(opt.valdataPath)
        
        print("Computing document input representations...1")
        processed_traindata, processed_trainlabels = preprocess_data(raw_data, glove_table, opt.nTrainDocs, opt)
        print("Computing document input representations...2")
        processed_valdata, processed_vallabels = preprocess_data(raw_valdata, glove_table, opt.nTestDocs, opt)

        if opt.savePreprocessed then
            print("Saving preprocessed data...")
            torch.save("processed_traindata"..opt.inputDim..".t7b", processed_traindata)
            torch.save("processed_valdata"..opt.inputDim..".t7b", processed_valdata)
            torch.save("processed_trainlabels"..opt.inputDim..".t7b", processed_trainlabels)
            torch.save("processed_vallabels"..opt.inputDim..".t7b", processed_vallabels)
        end
    end

    -- reformat
    local training_data = processed_traindata:sub(1, opt.nClasses*opt.nTrainDocs, 1, processed_traindata:size(2)):clone()
    local test_data = processed_valdata:sub(1, opt.nClasses*opt.nTestDocs, 1, processed_valdata:size(2)):clone()
    
    local training_labels = processed_trainlabels:sub(1, opt.nClasses*opt.nTrainDocs):clone()
    local test_labels = processed_vallabels:sub(1, opt.nClasses*opt.nTestDocs):clone()



    -- construct model:
    -- -- model:add(nn.Reshape(1, 500, true))
    -- -- model:add(nn.SpatialMaxPooling(51, 1, 1, 1))
    -- -- model:add(nn.Reshape(450, true))

    model = nn.Sequential()

    model:add(nn.Reshape(opt.inputDim, true))
    model:add(nn.ReLU())
    model:add(nn.Linear(opt.inputDim, 500))
    if opt.dropout then model:add(nn.Dropout(0.5)) end
    model:add(nn.ReLU())

    if opt.outputType == "continuous" then
        model:add(nn.Linear(500, 4))
        model:add(nn.Sigmoid())
        model:add(nn.Sum(2))
        model:add(nn.Add(1))
        criterion = nn.MSECriterion()
    end
    if opt.outputType == "categories" then
        model:add(nn.Linear(500, 5))
        model:add(nn.LogSoftMax())
        criterion = nn.ClassNLLCriterion()
    end

    print(model)

    local record_train_acc, record_test_acc = train_model(model, criterion, training_data, training_labels, test_data, test_labels, opt)

    local file = torch.DiskFile('record.txt', 'w')
    file:writeObject(record_train_acc)
    file:writeObject(record_test_acc)
    file:writeObject(opt)
    file:close()

    local results = test_model(model, test_data, test_labels, opt, 0)
    print(results)

    torch.save("trained_model_"..opt.outputType..".net", model)
    print("Trained model saved.")
end

main()
