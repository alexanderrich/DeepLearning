require 'nn'
require 'torch'



-- function to load glove for model 1 representation
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
-- load glove
glove_table = load_glove("glove.6B.100d.txt", 100)

-- function to preprocess string for model 1 representation
function preprocess_string(document)
   document = document:lower()
   doc_tensor = torch.zeros(100,1)
   local doc_size = 1
   for word in document:gmatch("%S+") do
      if glove_table[word:gsub("%p+", "")] then
         doc_size = doc_size + 1
         doc_tensor:add(glove_table[word:gsub("%p+", "")])
      end
   end
   return doc_tensor
end
   
-- alphabet for model 2 representation
local alphabet = {}
local dict = {}
local a2 = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
for i = 1,#a2 do
    table.insert(alphabet, a2:sub(i,i))
    dict[a2:sub(i,i)] = i
end

-- function to preprocess string for model 2 representation
function stringToTensor(str)
   local s = str:lower()
   local l = 1014
   local t = torch.Tensor(#alphabet, l)
   t:zero()
   for i = #s, math.max(#s - l + 1, 1), -1 do
      if dict[s:sub(i,i)] then
	 t[dict[s:sub(i,i)]][#s - i + 1] = 1
      end
   end
   return t
end

-- simple model averaging class for making predictions using 2 models
local Model_Average = torch.class("Model_Average")

function Model_Average:__init()
   self.model1 = torch.load("trained_model_categories.net")
   self.model2 = torch.load("trained_crepe_model.t7b")
end

function Model_Average:forward(data)
   self.output2 = self.model1:forward(preprocess_string(item):transpose(1,2))
   self.output1 = self.model2:forward(stringToTensor(item):transpose(1,2))
   -- sum log probabilities to get a measure of the "average" likelihood of the data
   -- no longer gives normalized probablity distribution but works fine for prediction
   return self.output1 + self.output2
end   
return Model_Average
