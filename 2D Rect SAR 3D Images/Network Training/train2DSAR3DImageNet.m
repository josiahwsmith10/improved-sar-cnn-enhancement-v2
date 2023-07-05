%% Include the directories
%-------------------------------------------------------------------------%
addpath(genpath("../../"))
addpath(genpath("C:/Data"))

%% Initialize Figures
%-------------------------------------------------------------------------%
fig = initializeNetworkFigures3D();

%% Load in Data
%-------------------------------------------------------------------------%
% load sar2DMIMOAWR1243_3D_rand_noAmp_1 % up to 256 points load
% sar2DMIMOAWR1243_3D_rand_noAmp_2 % up to 512 points loop1 =
% load("sar2DMIMOAWR1243_3D_rand_noAmp_2","loop").loop; % up to 512 points
% load sar2DMIMOAWR1243_3D_pngANDrand_noAmp loop2 =
% load("sar2DMIMOAWR1243_3D_png_yesAmp","loop").loop; load
% sar2DMIMOAWR1243_3D_pngORrand_yesAmp

% load sar2DMIMOAWR1243_3D_png_yesAmp
% load sar2DMIMOAWR1243_3D_rand_yesAmp
load sar2DMIMOAWR1243_3D_rand_yesAmp_v2

%% Change loop.dsPath
loop.dsPath = "C:/Data/2D Rect SAR 3D Images/sar2DMIMOAWR1243_3D_png_yesAmp/";

%% Change loop.dsPath (combine all data)
loop.dsPath = ["C:/Data/2D Rect SAR 3D Images/sar2DMIMOAWR1243_3D_png_yesAmp/","C:/Data/2D Rect SAR 3D Images/sar2DMIMOAWR1243_3D_rand_yesAmp_v2/"];

%% Create the Datastore
%-------------------------------------------------------------------------%
% ds = createDatastoreNorm({loop1.dsPath,loop2.dsPath}); ds =
% createDatastoreNorm(loop.dsPath);
ds = matRegressionDatastore(loop.dsPath);
ds.MiniBatchSize = 32;

%% Preview the Datset
%-------------------------------------------------------------------------%
previewSample3D(ds,im,-25,fig);

%% Train the Regression CNN
%-------------------------------------------------------------------------%
netParams.name = "improved";
netParams.C = [4 6 8 32];
netParams.B = [1 1 1 1];
netParams.w = 7;

netParams.num_epochs = 200000;
netParams.lr = 0.001;
netParams.lr_drop_factor = 0.5;
netParams.lr_drop_period = 100;
fcnnNet = fcnn_trainNetwork3D_ds(ds,netParams);

%% Retrain the Regression CNN
%-------------------------------------------------------------------------%
fcnnNet = fcnn_trainNetwork3D_ds(ds,fcnnNet);

%% Test the Network on One Sample
%-------------------------------------------------------------------------%
testOneSample3D(fcnnNet,ds,im,-25,fig)

%% Demo PSF, Diamond, and UTD
%-------------------------------------------------------------------------%
demoNetworkXYZ(im,ant,sar,fmcw,target,fcnnNet,-25)

%% Demo PSF Slices
%-------------------------------------------------------------------------%
demoNetworkPSFSlicesXYZ(im,ant,sar,fmcw,target,fcnnNet)
% add _BPA to also compare against BPA (takes about 2-hours)

%% Demo Cutout
%-------------------------------------------------------------------------%
demoNetworkCutoutXYZ(im,ant,sar,fmcw,target,fcnnNet,-25,"cable.png")

%% Test the Network with Real Data
%-------------------------------------------------------------------------%
load rectilinearTest3.mat knife
testReal(fcnnNet,knife,im,-25,fig);

%% Create Nice Figure (bothNet)
testReal(fcnnNet,knife,im,-20,fig);

%% Set Figure Aspects -10dB
setFigure(fig.enhancedImage)
exportgraphics(fig.enhancedImage.h,"./figures_jun2023/knife_enhanced.png","Resolution",600)

%% Set Figure Aspects -20dB
setFigure(fig.sarImage)
exportgraphics(fig.sarImage.h,"./figures_jun2023/knife_raw.png","Resolution",600)


%% Set Non-labeled figure -10dB
removeLabels(fig.enhancedImage)
exportgraphics(fig.enhancedImage.h,"./figures_jun2023/knife_enhanced_nolabels.png","Resolution",600)

%% Set Non-labeled figure -20dB
removeLabels(fig.sarImage)
exportgraphics(fig.sarImage.h,"./figures_jun2023/knife_raw_nolabels.png","Resolution",600)

%% Save the Network
%-------------------------------------------------------------------------%
save ./nets_jan22/sar2DMIMOAWR1243_3D_pngRand_net_improved fcnnNet im -v7.3

%% Functions

function setFigure(im)
h = im.h;

colorbar(h,"off")
h.FontSize = 12;
h.View = [-30,20];
h.XLim = [-0.1,0.1];
h.YLim = [0.225,0.375];
h.ZLim = [-0.1,0.15];
h.Title.String = "";
h.LineWidth = 1.5;
h.XLabel.FontSize = 20;
h.YLabel.FontSize = 20;
h.ZLabel.FontSize = 20;
h.XLabel.Interpreter = "latex";
h.YLabel.Interpreter = "latex";
h.ZLabel.Interpreter = "latex";
end

function removeLabels(im)
h = im.h;

colorbar(h,"off")
h.FontSize = 12;
h.View = [-30,20];
h.XLim = [-0.1,0.1];
h.YLim = [0.225,0.375];
h.ZLim = [-0.1,0.15];
h.XTickLabel = [];
h.XLabel.String = "";
h.YTickLabel = [];
h.YLabel.String = "";
h.ZTickLabel = [];
h.ZLabel.String = "";
h.Title.String = "";
colorbar(h,"off")
h.LineWidth = 1.5;
end

function [Layers,Options] = fcnn_createNetwork3D_ds(ds,netParams)
%% Declare Network Layers
%-------------------------------------------------------------------------%
if netParams.name == "basic"
    Layers = [ ...
        image3dInputLayer(ds.inSize,"Normalization","none")

        convolution3dLayer([15,1,1],4,"Padding","same")
        leakyReluLayer

        convolution3dLayer([1,15,1],4,"Padding","same")
        leakyReluLayer

        convolution3dLayer(3,1,"Padding","same")
        clippedReluLayer(1)

        regressionLayer
        ];
elseif netParams.name == "improved"
    Layers = [ ...
        image3dInputLayer(ds.inSize,"Normalization","none")

        convolution3dLayer(7,netParams.C(1),"Padding","same")
        leakyReluLayer

        convolution3dLayer(5,netParams.C(2),"Padding","same")
        leakyReluLayer

        convolution3dLayer(3,netParams.C(3) ,"Padding","same")
        leakyReluLayer

        convolution3dLayer(3,1,"Padding","same")
        clippedReluLayer(1)

        regressionLayer
        ];
elseif netParams.name == "ConvNeXt"
    % ConvNeXt
    C = netParams.C;
    B = netParams.B;
    w = netParams.w;
    Layers = ConvNeXt(C,B,w,ds);
elseif netParams.name == "UNet"
    Layers = customLayers();
end

%% Declare Network Options
%-------------------------------------------------------------------------%
Options = trainingOptions("adam", ...
    "MaxEpochs",netParams.num_epochs, ...
    "InitialLearnRate",netParams.lr,...
    "MiniBatchSize",ds.MiniBatchSize, ...
    "Plots","training-progress", ...
    "Verbose",true, ...
    "LearnRateSchedule","piecewise", ...
    "LearnRateDropFactor",netParams.lr_drop_factor, ...
    "LearnRateDropPeriod",netParams.lr_drop_period, ...
    "ExecutionEnvironment","gpu");

end

function [netOut,Layers,Options] = fcnn_trainNetwork3D_ds(ds,netParams,netIn)
%% Get the Network Layers and Options
%-------------------------------------------------------------------------%
[Layers,Options] = fcnn_createNetwork3D_ds(ds,netParams);

%% Train the Network
%-------------------------------------------------------------------------%
disp('Training Network!')
if nargin > 2
    Layers = netIn.Layers;
    netOut = trainNetwork(ds,Layers,Options);
else
    netOut = trainNetwork(ds,Layers,Options);
end
end

function Layers = ConvNeXt(C,B,w,ds)
% Using notation from the paper C - 1x4 vector of channel sizes for each
% stage B - 1x4 vector of number of blocks in each stage

counter(0);

Layers = layerGraph();

Layers = addLayers(Layers,image3dInputLayer(ds.inSize,"Normalization","none","Name","inputLayer"));
inName = "inputLayer";
outName = "addition_1";
for ind = 1:4
    inName2 = "convUp_"+(counter([])+1);
    Layers = addLayers(Layers,convolution3dLayer(1,C(ind),"Padding","same","Name","convUp_"+(counter([])+1)));
    Layers = connectLayers(Layers,inName,inName2);
    inName = inName2;
    for d = 1:B(ind)
        Layers = add_ConvNeXt_res_block(Layers,C(ind),inName,outName,w);
        inName = outName;
        outName = "addition_"+(counter([])+1);
    end
end

Layers = addLayers(Layers,[...
    convolution3dLayer(3,1,"Padding","same","Name","lastConv")
    clippedReluLayer(1)
    regressionLayer]);

Layers = connectLayers(Layers,inName,"lastConv");
end

function lgraph = add_ConvNeXt_res_block(lgraph,c,inName,outName,w)
if nargin < 2
    w = 7;
end

n = counter();

tempLayers = [
    convolution3dLayer([w w w],c,"Name","conv3d_1_"+n,"Padding","same")
    layerNormalizationLayer("Name","layernorm"+n)
    convolution3dLayer([1 1 1],c*4,"Name","conv3d_2_"+n,"Padding","same")
    leakyReluLayer("Name","relu"+n)
    convolution3dLayer([1 1 1],c,"Name","conv3d_3_"+n,"Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name",outName);
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

lgraph = connectLayers(lgraph,inName,"conv3d_1_"+n);
lgraph = connectLayers(lgraph,inName,outName+"/in1");
lgraph = connectLayers(lgraph,"conv3d_3_"+n,outName+"/in2");
end

function count = counter(initialize)
persistent currentCount;

if nargin == 1
    if ~isempty(initialize)
        currentCount = initialize;
    end
else
    currentCount = currentCount + 1;
end

count = currentCount;
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

function demoNetworkXYZ(im,ant,sar,fmcw,target,fcnnNet,dBMin)
%% PSF
target.isAmplitudeFactor = false;
target.xyz_m = single([mean(im.x_m),mean(im.y_m),mean(im.z_m)]);
target.amp = 1;

target.zOffset_m = mean(im.z_m);

im = getImageXYZ(target,sar,ant,fmcw,im);

psf.sar = im.sarImage;
psf.enhanced = predict(fcnnNet,im.sarImage);

if nnz(psf.enhanced) == 0
    warning("PSF ALL ZEROS!")
    return;
end

f = figure;
h = handle(axes);
plotXYZdB(h,f,psf.sar,im.x_m,im.y_m,im.z_m,[],dBMin,"Original PSF",12);

f = figure;
h = handle(axes);
plotXYZdB(h,f,psf.enhanced,im.x_m,im.y_m,im.z_m,[],dBMin,"Enhanced PSF",12);

%% Grid of Points
coord_x_m = reshape(im.x_m(round([1,1,3,3,2,1,2,3,2]*end/4)),[],1);
coord_y_m = reshape(im.y_m(round([1,3,3,1,2,2,3,2,1]*end/4)),[],1);
coord_z_m = reshape(im.z_m(round([1,1,1,1,2,3,3,3,3]*end/4)),[],1);

target.xyz_m = single([coord_x_m,coord_y_m,coord_z_m]);

im = getImageXYZ(target,sar,ant,fmcw,im);

diamond.sar = im.sarImage;
diamond.enhanced = predict(fcnnNet,im.sarImage);

if nnz(diamond.enhanced) == 0
    warning("DIAMOND ALL ZEROS!")
    return;
end

f = figure;
h = handle(axes);
plotXYZdB(h,f,diamond.sar,im.x_m,im.y_m,im.z_m,[],dBMin,"Original Grid",12);

f = figure;
h = handle(axes);
plotXYZdB(h,f,diamond.enhanced,im.x_m,im.y_m,im.z_m,[],dBMin,"Enhanced Grid",12);

%% UTD Letters
coord_x_m = reshape(im.x_m(round([1,1,1,2,3,3,3,5,6,6,6,6,7,9,9,9,9,10,10,11,11]*end/12)),[],1);
coord_y_m = reshape(im.y_m(round([6,5,4,3,4,5,6,6,6,5,4,3,6,6,5,4,3,6 ,3 ,5 ,4 ]*end/8 )),[],1);
coord_z_m = reshape(im.z_m(round([1,1,1,1,1,1,1,2,2,2,2,2,2,3,3,3,3,3 ,3 ,3 ,3 ]*end/4 )),[],1);

target.xyz_m = single([coord_x_m,coord_y_m,coord_z_m]);

im = getImageXYZ(target,sar,ant,fmcw,im);

utd.sar = im.sarImage;
utd.enhanced = predict(fcnnNet,im.sarImage);

if nnz(utd.enhanced) == 0
    warning("UTD ALL ZEROS!")
    return;
end

f = figure;
h = handle(axes);
plotXYZdB(h,f,utd.sar,im.x_m,im.y_m,im.z_m,[],dBMin,"Original UTD",12);

f = figure;
h = handle(axes);
plotXYZdB(h,f,utd.enhanced,im.x_m,im.y_m,im.z_m,[],dBMin,"Enhanced UTD",12);
end

function demoNetworkPSFSlicesXYZ(im,ant,sar,fmcw,target,fcnnNet)
%% PSF
target.xyz_m = single([mean(im.x_m),mean(im.y_m),mean(im.z_m)]);
target.amp = 1;

target.zOffset_m = mean(im.z_m);

im = getImageXYZ(target,sar,ant,fmcw,im);

psf.sar = im.sarImage./im.sarImage(end/2,end/2,end/2);
psf.enhanced = predict(fcnnNet,im.sarImage);
psf.enhanced = psf.enhanced./psf.enhanced(end/2,end/2,end/2);

if nnz(psf.enhanced) == 0
    warning("PSF ALL ZEROS!")
    return;
end

rma.x = psf.sar(:,end/2,end/2);
rma.y = psf.sar(end/2,:,end/2);
rma.z = squeeze(psf.sar(end/2,end/2,:));

enhanced.x = psf.enhanced(:,end/2,end/2);
enhanced.y = psf.enhanced(end/2,:,end/2);
enhanced.z = squeeze(psf.enhanced(end/2,end/2,:));

figure
subplot(131)
plot(im.x_m,rma.x,"-")
hold on
plot(im.x_m,enhanced.x,"--")
legend("RMA","Enhanced")
title("PSF X Slice")
xlabel("x (m)")
ylabel("Normalized")
ylim([0,1])

subplot(132)
plot(im.y_m,rma.y,"-")
hold on
plot(im.y_m,enhanced.y,"--")
legend("RMA","Enhanced")
title("PSF Y Slice")
xlabel("y (m)")
ylabel("Normalized")
ylim([0,1])

subplot(133)
plot(squeeze(im.z_m),rma.z,"-")
hold on
plot(squeeze(im.z_m),enhanced.z,"--")
legend("RMA","Enhanced")
title("PSF Z Slice")
xlabel("z (m)")
ylabel("Normalized")
ylim([0,1])

psf.sar = db(psf.sar);
psf.enhanced = db(psf.enhanced);

psf.sar(psf.sar<-100) = -100;
psf.enhanced(psf.enhanced<-100) = -100;

rma.x = psf.sar(:,end/2,end/2);
rma.y = psf.sar(end/2,:,end/2);
rma.z = squeeze(psf.sar(end/2,end/2,:));

enhanced.x = psf.enhanced(:,end/2,end/2);
enhanced.y = psf.enhanced(end/2,:,end/2);
enhanced.z = squeeze(psf.enhanced(end/2,end/2,:));

figure
subplot(131)
plot(im.x_m,rma.x,"-")
hold on
plot(im.x_m,enhanced.x,"--")
legend("RMA","Enhanced")
title("PSF X Slice")
xlabel("x (m)")
ylabel("dB")
ylim([-70,0])

subplot(132)
plot(im.y_m,rma.y,"-")
hold on
plot(im.y_m,enhanced.y,"--")
legend("RMA","Enhanced")
title("PSF Y Slice")
xlabel("y (m)")
ylabel("dB")
ylim([-70,0])

subplot(133)
plot(squeeze(im.z_m),rma.z,"-")
hold on
plot(squeeze(im.z_m),enhanced.z,"--")
legend("RMA","Enhanced")
title("PSF Z Slice")
xlabel("z (m)")
ylabel("dB")
ylim([-70,0])
end

function demoNetworkCutoutXYZ(im,ant,sar,fmcw,target,fcnnNet,dBMin,pngName)
%% Cutout
target.xStep_m = 0.5e-3;
target.yStep_m = 0.5e-3;
target.xOffset_m = -0.025; %-0.0125,0.0125
target.yOffset_m = 0.05; %-0.025,0.025
target.zOffset_m = 0.25;
target.o_x = 2e-3;
target.o_y = 2e-3;
target.ampAdjust = 1;
target.downSample = 4;
fig.isFig = false;
target = gettarget2DpngNAU(pngName,target,im,fig);

im = getImageXYZ(target,sar,ant,fmcw,im);

cutout.sar = im.sarImage;
sarImage = single(zeros(size(im.sarImage)));
sarImage(db(im.sarImage/max(im.sarImage(:)))>-25) = im.sarImage(db(im.sarImage/max(im.sarImage(:)))>-25);
sarImage = sarImage/max(sarImage(:));
cutout.enhanced = predict(fcnnNet,sarImage);

if nnz(cutout.enhanced) == 0
    warning("CUTOUT ALL ZEROS!")
    return;
end

f = figure;
h = handle(axes);
plotXYZdB(h,f,cutout.sar,im.x_m,im.y_m,im.z_m,[],dBMin,"Original Cutout",12);

f = figure;
h = handle(axes);
plotXYZdB(h,f,cutout.enhanced,im.x_m,im.y_m,im.z_m,[],dBMin,"Enhanced Cutout",12);
end

function im = getImageXYZ(target,sar,ant,fmcw,im)
% Get echo signal
sarData = updatetargetNAU(target,sar,fmcw);

% Convert multistatic-to-monostatic
sarData_y_x_k = reshape(permute(sarData,[1,2,4,3,5]),[],sar.numX,fmcw.ADCSamples);
if target.isMIMO
    pc = exp(-1j * reshape(fmcw.k,1,1,[]) .* repmat(ant.tx.xyz_m(:,:,2) - ant.rx.xyz_m(:,:,2),size(sarData,4),1).^2 / (4 * abs(ant.tx.z0_m - target.zOffset_m)));
    sarDataPC = pc .* sarData_y_x_k;
else
    sarDataPC = sarData_y_x_k;
end

% Consider the SISO virtual array
sar.yStep_m = fmcw.lambda_m/4;

fig.isFig = false;
im = uniform_SISO_2D_array_reconstructImage_3DNAU(sarDataPC,target,fmcw,ant,sar,im,fig);
im.sarImage = im.sarImage/max(im.sarImage(:));
end

function testOneSample3D(fcnnNet,ds,im,dBMin,fig)
if ds.isShuffleable
    ds = ds.shuffle;
end

prev = ds.read;
sarImage = prev.Var1{1};
idealImage = prev.Var2{1};
enhancedImage = predict(fcnnNet,sarImage);
if nnz(enhancedImage) == 0
    warning("ALL ZEROS!")
    return;
end

h = fig.idealImage.h;
f = fig.idealImage.f;
plotXYZdB(h,f,idealImage,im.x_m,im.y_m,im.z_m,[],dBMin,"Original Reflectivity Function",12);

h = fig.sarImage.h;
f = fig.sarImage.f;
plotXYZdB(h,f,sarImage,im.x_m,im.y_m,im.z_m,[],dBMin,"Reconstructed Image",12);

h = fig.enhancedImage.h;
f = fig.enhancedImage.f;
plotXYZdB(h,f,enhancedImage,im.x_m,im.y_m,im.z_m,[],dBMin,"Enhanced Image",12);
end

function testReal(fcnnNet,sarImage,im,dBMin,fig)
sarImage = exp(sarImage);
sarImage = sarImage/max(abs(sarImage(:)));
sarImage_dB = db(sarImage);
sarImage(sarImage_dB < -10) = 0;
sarImage = sarImage*10;
enhancedImage = predict(fcnnNet,sarImage);
if nnz(enhancedImage) == 0
    warning("ALL ZEROS!")
    return;
end

h = fig.sarImage.h;
f = fig.sarImage.f;
plotXYZdB(h,f,sarImage,im.x_m,im.y_m,im.z_m,[],dBMin,"Reconstructed Image",12);

h = fig.enhancedImage.h;
f = fig.enhancedImage.f;
plotXYZdB(h,f,enhancedImage,im.x_m,im.y_m,im.z_m,[],dBMin,"Enhanced Image",12);
end

function fig = initializeNetworkFigures3D
set(0,'DefaultFigureWindowStyle','docked')

fig.idealImage.f = figure;
fig.idealImage.h = handle(axes);

fig.sarImage.f = figure;
fig.sarImage.h = handle(axes);

fig.enhancedImage.f = figure;
fig.enhancedImage.h = handle(axes);
end

function previewSample3D(ds,im,dBMin,fig)
if ds.isShuffleable
    ds = ds.shuffle;
end

prev = ds.read;
sarImage = prev.Var1{1};
idealImage = prev.Var2{1};

h = fig.idealImage.h;
f = fig.idealImage.f;
plotXYZdB(h,f,idealImage,im.x_m,im.y_m,im.z_m,[],dBMin,"Original Reflectivity Function",12);

h = fig.sarImage.h;
f = fig.sarImage.f;
plotXYZdB(h,f,sarImage,im.x_m,im.y_m,im.z_m,[],dBMin,"Reconstructed Image",12);
end