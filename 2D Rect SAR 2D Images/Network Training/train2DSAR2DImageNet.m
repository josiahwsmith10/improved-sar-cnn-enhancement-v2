%% Include the directories
%-------------------------------------------------------------------------%
addpath(genpath("../../"))

%% Load in data
%-------------------------------------------------------------------------%
% all.rand_1.sar = load("sar2DMIMOAWR1243_256x256_250mm_rand").sarImageAll;
% all.rand_1.ideal = load("sar2DMIMOAWR1243_256x256_250mm_rand").idealImageAll;

% all.rand_2.sar = load("sar2DMIMOAWR1243_256x256_250mm_rand_2").sarImageAll;
% all.rand_2.ideal = load("sar2DMIMOAWR1243_256x256_250mm_rand_2").idealImageAll;

% all.rand_3.sar = load("sar2DMIMOAWR1243_256x256_250mm_rand_withNoise").sarImageAll;
% all.rand_3.ideal = load("sar2DMIMOAWR1243_256x256_250mm_rand_withNoise").idealImageAll;

% all.pngANDrand_1.sar = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand").sarImageAll;
% all.pngANDrand_1.ideal = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand").idealImageAll;

% all.pngANDrand_2.sar = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand_2").sarImageAll;
% all.pngANDrand_2.ideal = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand_2").idealImageAll;

% all.pngANDrand_3.sar = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand_3").sarImageAll;
% all.pngANDrand_3.ideal = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand_3").idealImageAll;

% sim.sar = load("sar2DMIMOAWR1243_256x256_250mm_5").sarImageAll;
% sim.ideal = load("sar2DMIMOAWR1243_256x256_250mm_5").idealImageAll;

sim.sar = load("sar2DMIMOAWR1243_256x128_250mm_rand_2").sarImageAll;
sim.ideal = load("sar2DMIMOAWR1243_256x128_250mm_rand_2").idealImageAll;

im = load("sar2DMIMOAWR1243_256x256_250mm_4").im;

all.names = string(fieldnames(all));

%% Combine all the data
%-------------------------------------------------------------------------%
sim.sar = [];
sim.ideal = [];

for indName = 1:length(all.names)
    sim.sar = cat(3,sim.sar,all.(all.names(indName)).sar);
    sim.ideal = cat(3,sim.ideal,all.(all.names(indName)).ideal);
end

numSample = size(sim.sar,3);

%% Plot some samples
%-------------------------------------------------------------------------%
randInd = randi(numSample);
figure;
subplot(121); mesh(im.x_m,im.y_m,sim.sar(:,:,randInd)','FaceColor','interp'); title("SAR " + randInd); view(2)
subplot(122); mesh(im.x_m,im.y_m,sim.ideal(:,:,randInd)','FaceColor','interp'); title("Ideal " + randInd); view(2)
clear randInd

%% Save the training data
%-------------------------------------------------------------------------%
save trainData_sar2DMIMOAWR1243_256x256_250mm sim im -v7.3

%% Train the Regression CNN
%-------------------------------------------------------------------------%
fcnnNet = fcnn_trainNetwork2D_custom(sim.sar,sim.ideal);

%% Test the Network on One Sample
%-------------------------------------------------------------------------%
randInd = randi(numSample);
testOneSample(fcnnNet,im,sim,randInd)

%% Test the Network on One Sample LOG
%-------------------------------------------------------------------------%
randInd = randi(numSample);
testOneSampleLog(fcnnNet,im,sim,randInd)

%% Make PSF and Diamond and UTD
%-------------------------------------------------------------------------%
demoNetwork(im,ant,sar,fmcw,fcnnNet)

%% Save the Network
%-------------------------------------------------------------------------%
save net_sar2DMIMOAWR1243_256x256_250mm fcnnNet im -v7.3

%%

predict(fcnnNet,ones(256));

%% Functions

function [Layers,Options] = fcnn_createNetwork2D_custom(inputSize)
%% Declare Network Layers
%-------------------------------------------------------------------------%
Layers = [ ...
    imageInputLayer(inputSize,"Normalization","rescale-zero-one")
    
%     convolution2dLayer(5,1,"Padding","same")
%     leakyReluLayer
    
%     convolution2dLayer(1,1,"Padding","same")
%     leakyReluLayer
    
    convolution2dLayer(5,1,"Padding","same","WeightsInitializer",@(sz) noisyDelta(sz,100,100),"BiasLearnRateFactor",1)
%     convolution2dLayer(1,1,"Padding","same","WeightsInitializer","zeros","BiasLearnRateFactor",0)
%     reluLayer
%     
%     convolution2dLayer(1,1,"Padding","same")
%     clippedReluLayer(1)
    
%     convolution2dLayer(1,1)
    
%     convolution2dLayer(1,1,"Padding","same","WeightsInitializer",@(sz) noisyDelta(sz,100,100),"BiasLearnRateFactor",1)
%     batchNormalizationLayer
%     clippedReluLayer(1)
    reluLayer
    regressionLayerJWS
    ];

% Layers = customLayers();

%% Declare Network Options
%-------------------------------------------------------------------------%
Options = trainingOptions("adam", ...
    "MaxEpochs",50, ...
    "InitialLearnRate",0.1,...
    "MiniBatchSize",128, ...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",true, ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.5, ...
    "LearnRateDropPeriod",4, ...
    "ExecutionEnvironment","multi-gpu");

end

function [netOut,Layers,Options] = fcnn_trainNetwork2D_custom(inputData,outputData)
%% Get Data Dimensions and Create Input Data Matrix
%-------------------------------------------------------------------------%
if ndims(inputData) == 3
    inputData = permute(inputData,[1 2 4 3]);
    outputData = permute(outputData,[1 2 4 3]);
end

if ~isa(inputData,"single")
    inputData = single(gather(inputData));
    outputData = single(gather(outputData));
end

%% Get the Network Layers and Options
%-------------------------------------------------------------------------%
[Layers,Options] = fcnn_createNetwork2D_custom(size(inputData,[1,2,3]));

%% Train the Network
%-------------------------------------------------------------------------%
disp('Training Network!')
netOut = trainNetwork(inputData,outputData,Layers,Options);
end

function lgraph = customLayers()
lgraph = layerGraph();

tempLayers = [
    imageInputLayer([256 256 1],"Name","imageinput")
    convolution2dLayer([40 40],16,"Name","conv_1","Padding","same")
    leakyReluLayer(0.01,"Name","leakyrelu_1")
    convolution2dLayer([25 25],40,"Name","conv_2","Padding","same")
    leakyReluLayer(0.01,"Name","leakyrelu_2")
    convolution2dLayer([16 16],16,"Name","conv_4","Padding","same")
    leakyReluLayer(0.01,"Name","leakyrelu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([16 16],12,"Name","conv_5")
    leakyReluLayer(0.01,"Name","leakyrelu_4")
    transposedConv2dLayer([16 16],9,"Name","transposed-conv")
    leakyReluLayer(0.01,"Name","leakyrelu_5")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    maxPooling2dLayer([16 16],"Name","maxpoolForUnpool","HasUnpoolingOutputs",true,"Padding","same","Stride",[16 16])
    maxUnpooling2dLayer("Name","maxunpool")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","depthcat")
    convolution2dLayer([5 5],1,"Name","conv_3","Padding","same")
    clippedReluLayer(3,"Name","clippedrelu")
    scalingLayer("Name","scaling")
    regressionLayer("Name","regressionoutput")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,"leakyrelu_3","conv_5");
lgraph = connectLayers(lgraph,"leakyrelu_3","maxpoolForUnpool");
lgraph = connectLayers(lgraph,"maxpoolForUnpool/indices","maxunpool/indices");
lgraph = connectLayers(lgraph,"maxpoolForUnpool/size","maxunpool/size");
lgraph = connectLayers(lgraph,"maxunpool","depthcat/in1");
lgraph = connectLayers(lgraph,"leakyrelu_5","depthcat/in2");


% Segmentation Network
filterSize = 3;
numFilters = 32;
conv = convolution2dLayer(filterSize,numFilters,'Padding',1);
relu = reluLayer();

poolSize = 2;
maxPoolDownsample2x = maxPooling2dLayer(poolSize,'Stride',2);
downsamplingLayers = [
    conv
    relu
    maxPoolDownsample2x
    conv
    relu
    maxPoolDownsample2x
    ];

filterSize = 4;
transposedConvUpsample2x = transposedConv2dLayer(4,numFilters,'Stride',2,'Cropping',1);

upsamplingLayers = [
    transposedConvUpsample2x
    relu
    transposedConvUpsample2x
    relu
    ];

conv1x1 = convolution2dLayer(1,1);
finalLayers = [
    conv1x1
    clippedReluLayer(1)
    regressionLayer
    ];

lgraph = [
    imageInputLayer([256 256 1],"Name","imageinput")    
    downsamplingLayers
    upsamplingLayers
    finalLayers
    ];
end

function demoNetwork(im,ant,sar,fmcw,fcnnNet)
target.xyz_m = single([mean(im.x_m),mean(im.y_m),0]);
target.amp = 1;

target.isAmplitudeFactor = true;
target.isMIMO = true;
target.isGPU = true;
target.zOffset_m = 0;

im = getImage(target,sar,ant,fmcw,im);

psf.sar = im.sarImage;
psf.denoised = predict(fcnnNet,im.sarImage);

figure;
mesh(im.x_m,im.y_m,psf.sar','FaceColor','interp')
xlim([im.x_m(1),im.x_m(end)])
ylim([im.y_m(1),im.y_m(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("PSF RMA Original");
figure;
mesh(im.x_m,im.y_m,psf.denoised','FaceColor','interp')
xlim([im.x_m(1),im.x_m(end)])
ylim([im.y_m(1),im.y_m(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("PSF Enhanced")

coord_x_m = [im.x_m(end/2),im.x_m(end/4),im.x_m(end*3/4),im.x_m(end/2)]';
coord_y_m = [im.y_m(end/4),im.y_m(end/2),im.y_m(end/2),im.y_m(end*3/4)]';
target.xyz_m = single([coord_x_m,coord_y_m,zeros(size(coord_x_m))]);

im = getImage(target,sar,ant,fmcw,im);

diamond.sar = im.sarImage;
diamond.denoised = predict(fcnnNet,im.sarImage);

figure;
mesh(im.x_m,im.y_m,diamond.sar','FaceColor','interp')
xlim([im.x_m(1),im.x_m(end)])
ylim([im.y_m(1),im.y_m(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("Diamond RMA Original");
figure;
mesh(im.x_m,im.y_m,diamond.denoised','FaceColor','interp')
xlim([im.x_m(1),im.x_m(end)])
ylim([im.y_m(1),im.y_m(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("Diamond Enhanced")

% make UTD Logo
coord_x_m = im.x_m(round([1,1,1,2,3,3,3,5,6,6,6,6,7,9,9,9,9,10,10,11,11]*end/12));
coord_y_m = im.y_m(round([6,5,4,3,4,5,6,6,6,5,4,3,6,6,5,4,3,6,3,5,4]*end/8))';
target.xyz_m = single([coord_x_m,coord_y_m,zeros(size(coord_x_m))]);

im = getImage(target,sar,ant,fmcw,im);

utd.sar = im.sarImage;
utd.denoised = predict(fcnnNet,im.sarImage);

figure;
mesh(im.x_m,im.y_m,utd.sar','FaceColor','interp')
xlim([im.x_m(1),im.x_m(end)])
ylim([im.y_m(1),im.y_m(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("UTD RMA Original");
figure;
mesh(im.x_m,im.y_m,utd.denoised','FaceColor','interp')
xlim([im.x_m(1),im.x_m(end)])
ylim([im.y_m(1),im.y_m(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("UTD Enhanced")
end

function im = getImage(target,sar,ant,fmcw,im)
% Get echo signal
sarData = updatetargetNAU(target,sar,fmcw);

% Convert multistatic-to-monostatic
sarData_y_x_k = reshape(permute(sarData,[1,2,4,3,5]),[],sar.numX,fmcw.ADCSamples);
if target.isMIMO
    pc = exp(-1j * reshape(fmcw.k,1,1,[]) .* repmat(ant.tx.xyz_m(:,:,2) - ant.rx.xyz_m(:,:,2),16,1).^2 / (4 * abs(ant.tx.z0_m - target.zOffset_m)));
    sarDataPC = pc .* sarData_y_x_k;
else
    sarDataPC = sarData_y_x_k;
end

% Consider the SISO virtual array
sar.yStep_m = fmcw.lambda_m/4;
sar.numY = 256;

fig.isFig = false;
im = uniform_SISO_2D_array_reconstructImage_2DNAU(sarDataPC,target,fmcw,ant,sar,im,fig);
end

function testOneSample(fcnnNet,im,sim,randInd)
sim.denoisedTest = predict(fcnnNet,sim.sar(:,:,randInd));

sim.denoisedTest = imresize(sim.denoisedTest,[im.numX,im.numY]);

figure
subplot(131);
mesh(im.x_m,im.y_m,imresize(sim.sar(:,:,randInd)',[im.numX,im.numY]),'FaceColor','interp');
title("SAR Image (Input)");
view(2)


subplot(132);
mesh(im.x_m,im.y_m,sim.denoisedTest','FaceColor','interp');
title("Denoised (Output)");
view(2)

subplot(133);
mesh(im.x_m,im.y_m,imresize(sim.ideal(:,:,randInd)',[im.numX,im.numY]),'FaceColor','interp');
title("Ideal");
view(2)
end

function testOneSampleLog(fcnnNet,im,sim,randInd)
sim.denoisedTest = predict(fcnnNet,sim.sar(:,:,randInd));

figure
subplot(131);
temp = mag2db(sim.sar(:,:,randInd)');
temp = temp - max(temp(:));
temp(temp<-100) = -101;
mesh(im.x_m,im.y_m,temp,'FaceColor','interp');
zlim([-100,0])
xlim([im.x_m(1),im.x_m(end)])
ylim([im.y_m(1),im.y_m(end)])
title("SAR Image (Input)");
view(2)


subplot(132);
temp = mag2db(sim.denoisedTest');
temp = temp - max(temp(:));
temp(temp<-100) = -101;
mesh(im.x_m,im.y_m,temp,'FaceColor','interp');
zlim([-100,0])
xlim([im.x_m(1),im.x_m(end)])
ylim([im.y_m(1),im.y_m(end)])
title("Denoised (Output)");
view(2)

subplot(133);
temp = mag2db(sim.ideal(:,:,randInd)');
temp = temp - max(temp(:));
temp(temp<-100) = -101;
mesh(im.x_m,im.y_m,temp,'FaceColor','interp');
zlim([-100,0])
xlim([im.x_m(1),im.x_m(end)])
ylim([im.y_m(1),im.y_m(end)])
title("Ideal");
view(2)

cb = colorbar;
ylabel(cb,"dB","FontSize",20)
end

function weights = noisyDelta(sz,numIn,numOut)

Z = 2*rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = zeros(sz,'single');
weights(round(end/2),round(end/2),:) = 1;

end