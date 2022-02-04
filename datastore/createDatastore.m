function ds = createDatastore(dsPath)
inputData = fileDatastore(fullfile(dsPath + "Input/"),"ReadFcn",@load,"FileExtensions",".mat");
outputData = fileDatastore(fullfile(dsPath + "Output/"),"ReadFcn",@load,"FileExtensions",".mat");

inputData = transform(inputData,@(data) inputPreprocess(data));
outputData = transform(outputData,@(data) outputPreprocess(data));
ds = combine(inputData,outputData);
end

function image = inputPreprocess(data)
image = data.Input;
image = {image};
end

function image = outputPreprocess(data)
image = data.Output;
image = {image};
end