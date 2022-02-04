%% Include the directories
%-------------------------------------------------------------------------%
addpath(genpath("../../"))

%% Load in data
%-------------------------------------------------------------------------%
% all.rand_1.sar = load("sar2DMIMOAWR1243_256x256_250mm_rand").sarImageAll;
% all.rand_1.ideal = load("sar2DMIMOAWR1243_256x256_250mm_rand").idealImageAll;

all.pngANDrand_1.sar = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand").sarImageAll;
all.pngANDrand_1.ideal = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand").idealImageAll;

all.pngANDrand_2.sar = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand_2").sarImageAll;
all.pngANDrand_2.ideal = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand_2").idealImageAll;

all.pngANDrand_3.sar = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand_3").sarImageAll;
all.pngANDrand_3.ideal = load("sar2DMIMOAWR1243_256x256_250mm_pngANDrand_3").idealImageAll;

im.nFFTx = 512;
im.nFFTy = 512;

im.numX = 256;
im.numY = 256;
im.x_m = linspace(0.1/im.numX-0.05,0.05,im.numX)';
im.y_m = linspace(0.1/im.numY-0.05,0.05,im.numY);

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
sim.denoisedTest = predict(fcnnNet,sim.sar(:,:,randInd));

figure
subplot(131); mesh(im.x_m,im.y_m,sim.sar(:,:,randInd)','FaceColor','interp'); title("SAR Image (Input)");view(2)
subplot(132); mesh(im.x_m,im.y_m,sim.denoisedTest','FaceColor','interp'); title("Denoised (Output)");view(2)
subplot(133); mesh(im.x_m,im.y_m,sim.ideal(:,:,randInd)','FaceColor','interp'); title("Ideal");view(2)

% figure; 
% mesh(im.x_m,p.zT,sim.sar(:,:,randInd)','FaceColor','interp'); 
% title("RMA Noisy (Input)");
% view(2)
% xlabel("Cross-Range (m)")
% ylabel("Range (m)")
% 
% figure; 
% mesh(im.x_m,p.zT,sim.denoisedTest','FaceColor','interp'); 
% title("Denoised (Output)");
% view(2)
% xlabel("Cross-Range (m)")
% ylabel("Range (m)")
% 
% figure; 
% mesh(im.x_m,im.y_m,sim.ideal(:,:,randInd)','FaceColor','interp'); 
% title("Ideal");
% view(2)
% xlabel("Cross-Range (m)")
% ylabel("Range (m)")

%% Make PSF and Diamond and UTD - DOES NOT WORK
%-------------------------------------------------------------------------%
demoNetwork(p,Params,fcnnNet)

%% Save the Network
%-------------------------------------------------------------------------%
save net_sar2DMIMOAWR1243_256x256_250mm fcnnNet im -v7.3

%% Functions

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
end

function [Layers,Options] = fcnn_createNetwork2D_custom(inputSize)
%% Declare Network Layers
%-------------------------------------------------------------------------%
Layers = [ ...
    imageInputLayer(inputSize)
    
    convolution2dLayer(40,4,"Padding","same")
    batchNormalizationLayer
    leakyReluLayer
    
    convolution2dLayer(25,4,"Padding","same")
    batchNormalizationLayer
    leakyReluLayer
    
    convolution2dLayer(15,4,"Padding","same")
    batchNormalizationLayer
    leakyReluLayer
    
    convolution2dLayer(3,1,"Padding","same")
    batchNormalizationLayer
    clippedReluLayer(1)
    
    regressionLayer
    ];

% Layers = customLayers();

%% Declare Network Options
%-------------------------------------------------------------------------%
Options = trainingOptions("adam", ...
    "MaxEpochs",50, ...
    "InitialLearnRate",1e-3,...
    "MiniBatchSize",64, ...
    "Shuffle","every-epoch", ...
    "Plots","training-progress", ...
    "Verbose",true, ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",0.5, ...
    "LearnRateDropPeriod",10, ...
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

function demoNetwork(im,Params,fcnnNet)
% DOES NOT WORK
im.coord_y_m = mean(im.yT);
im.coord_z_m = mean(im.zT);
im.amp_yz = 1;

psf.echo = AWR1243_createEchoCoords_MIMO(Params,im).*Params.phaseCorrectionFactor;
psf.rma = gather(single(reconstructImage_2D_RMAgpu_mini_custom(psf.echo,Params,im)));
psf.denoised = predict(fcnnNet,psf.rma);

figure;
mesh(im.yT,im.zT,psf.rma','FaceColor','interp')
xlim([im.yT(1),im.yT(end)])
ylim([im.zT(1),im.zT(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("PSF RMA Original");
figure;
mesh(im.yT,im.zT,psf.denoised','FaceColor','interp')
xlim([im.yT(1),im.yT(end)])
ylim([im.zT(1),im.zT(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("PSF Enhanced")

im.coord_y_m = [im.yT(end/2),im.yT(end/4),im.yT(end*3/4),im.yT(end/2)];
im.coord_z_m = [im.zT(end/4),im.zT(end/2),im.zT(end/2),im.zT(end*3/4)];
im.amp_yz = ones(size(im.coord_y_m));

diamond.echo = AWR1243_createEchoCoords_MIMO(Params,im).*Params.phaseCorrectionFactor;
diamond.rma = gather(single(reconstructImage_2D_RMAgpu_mini_custom(diamond.echo,Params,im)));
diamond.denoised = predict(fcnnNet,diamond.rma);

figure;
mesh(im.yT,im.zT,diamond.rma','FaceColor','interp')
xlim([im.yT(1),im.yT(end)])
ylim([im.zT(1),im.zT(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("Diamond RMA Original");
figure;
mesh(im.yT,im.zT,diamond.denoised','FaceColor','interp')
xlim([im.yT(1),im.yT(end)])
ylim([im.zT(1),im.zT(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("Diamond Enhanced")

% TODO: make UTD Logo
im.coord_y_m = im.yT(round([1,1,1,2,3,3,3,5,6,6,6,6,7,9,9,9,9,10,10,11,11]*end/12));
im.coord_z_m = im.zT(round([6,5,4,3,4,5,6,6,6,5,4,3,6,6,5,4,3,6,3,5,4]*end/8));
im.amp_yz = ones(size(im.coord_y_m));

utd.echo = AWR1243_createEchoCoords_MIMO(Params,im).*Params.phaseCorrectionFactor;
utd.rma = gather(single(reconstructImage_2D_RMAgpu_mini_custom(utd.echo,Params,im)));
utd.denoised = predict(fcnnNet,utd.rma);

figure;
mesh(im.yT,im.zT,utd.rma','FaceColor','interp')
xlim([im.yT(1),im.yT(end)])
ylim([im.zT(1),im.zT(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("UTD RMA Original");
figure;
mesh(im.yT,im.zT,utd.denoised','FaceColor','interp')
xlim([im.yT(1),im.yT(end)])
ylim([im.zT(1),im.zT(end)])
xlabel("Cross-Range (m)")
ylabel("Range (m)")
view(2)
title("UTD Enhanced")
end