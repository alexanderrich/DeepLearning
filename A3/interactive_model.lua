require 'torch'
require 'nn'
require 'model_average'
model = Model_Average()
numitems = io.read()

local alphabet = {}
local dict = {}
local a2 = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
for i = 1,#a2 do
    table.insert(alphabet, a2:sub(i,i))
    dict[a2:sub(i,i)] = i
end

for i = 1, tonumber(numitems) do
   item = io.read()
   stringToTensor(i)
   print(i)
end


function Data:stringToTensor(str)
   local s = str:lower()
   local l = #s
   local t = torch.Tensor(#alphabet, l)
   t:zero()
   for i = #s, math.max(#s - l + 1, 1), -1 do
      if dict[s:sub(i,i)] then
	 t[dict[s:sub(i,i)]][#s - i + 1] = 1
      end
   end
   return t
end
