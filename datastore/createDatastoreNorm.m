function ds = createDatastoreNorm(dsPath)
inputData = fileDatastore(fullfile(dsPath + "Input/"),"ReadMode","file","ReadFcn",@load,"FileExtensions",".mat");
outputData = fileDatastore(fullfile(dsPath + "Output/"),"ReadMode","file","ReadFcn",@load,"FileExtensions",".mat");

inputData = transform(inputData,@(data) inputPreprocess(data));
outputData = transform(outputData,@(data) outputPreprocess(data));
ds = combine(inputData,outputData);
end

function img = inputPreprocess(data)
img = data.Input;
img = img/max(abs(img(:)));
img = {img};
end

function img = outputPreprocess(data)
img = data.Output;
img = img/max(abs(img(:)));
img = {img};
end