% Test Bench for the Non-App Versions of the Utilities
addpath(genpath("../../"))

%% Load in Saved Array and FMCW
% These can be created in the app
%-------------------------------------------------------------------------%
ant = load("AWR1243").savedant;
fmcw = load("fmcw_v1").savedfmcw;
figReal = initializeFiguresNAU();
figReal.isFig = true;

%% Create Array
%-------------------------------------------------------------------------%
ant.tx.z0_m = 0;
ant.rx.z0_m = 0;
ant = updateantNAU(ant,fmcw,figReal);

%% Create the SAR Scenario
%-------------------------------------------------------------------------%
sar.method = "Rectilinear";
sar.numX = 256;
sar.numY = 32;
sar.xStep_m = fmcw.lambda_m/4;
sar.yStep_m = fmcw.lambda_m*2;

sar.thetaMax_deg = 360;
sar.numTheta = 1024;
sar = updatesarNAU(sar,ant,figReal);

%% Set Imaging Parameters
%-------------------------------------------------------------------------%
im.nFFTx = 512;
im.nFFTy = 512;
im.nFFTz = 512;

im.numX = 128;
im.numY = 128;
im.numZ = 128;
im.x_m = linspace(0.3/im.numX-0.15,0.15,im.numX)';
im.y_m = linspace(0.3/im.numY-0.15,0.15,im.numY);
im.z_m = reshape(linspace(0.1+0.4/im.numZ,0.5,im.numZ),1,1,[]);

%% Load Data
%-------------------------------------------------------------------------%
% sar.numY = 256;
% sar.isTwoDirectionScanning = false;
% % sarData = nRx x nTx x nX x nY x nK
% sarData = dataReadTSWNAU("rectilinearTest1.bin",sar,fmcw);

load rectilinearTest3

%% Reconstruct Image
%-------------------------------------------------------------------------%
% Consider the SISO virtual array
sar.yStep_m = fmcw.lambda_m/4;
target.isGPU = true;
target.isAmplitudeFactor = true;
im.dBMin = -5;
im = uniform_SISO_2D_array_reconstructImage_3DNAU(sarData,target,fmcw,ant,sar,im,figReal);

%% Try Enhancing the Data
%-------------------------------------------------------------------------%
% load sar2DMIMOAWR1243_3D_rand_noAmp_1_net fcnnNet
% load bothNet.mat fcnnNet
load pngOnlyNet.mat fcnnNet

%% Test the Network with Real Data
%-------------------------------------------------------------------------%
testReal(fcnnNet,im.sarImage,im,-25);

%% Functions
%-------------------------------------------------------------------------%
function testReal(fcnnNet,sarImage,im,dBMin)
% sarImage = exp(sarImage);
sarImage = sarImage/max(abs(sarImage(:)));
sarImage_dB = db(sarImage);
sarImage(sarImage_dB < -8) = 0;
sarImage = sarImage*10;
enhancedImage = predict(fcnnNet,sarImage);
if nnz(enhancedImage) == 0
    warning("ALL ZEROS!")
    return;
end

f = figure;
h = handle(axes);
plotXYZdB(h,f,sarImage,im.x_m,im.y_m,im.z_m,[],dBMin,"Reconstructed Image",12);
view(-30,17)

f = figure;
h = handle(axes);
plotXYZdB(h,f,enhancedImage,im.x_m,im.y_m,im.z_m,[],dBMin,"Enhanced Image",12);
view(-30,17)
end
