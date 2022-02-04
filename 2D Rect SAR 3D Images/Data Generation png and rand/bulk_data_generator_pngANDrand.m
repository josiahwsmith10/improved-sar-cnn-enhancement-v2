% Test Bench for the Non-App Versions of the Utilities
addpath(genpath("../../"))

%% Load in Saved Array and FMCW
% These can be created in the app
%-------------------------------------------------------------------------%
ant = load("AWR1243").savedant;
fmcw = load("fmcw_v1").savedfmcw;
fig = initializeFiguresNAU();
fig.isFig = true;

%% Create Array
%-------------------------------------------------------------------------%
ant.tx.z0_m = 0;
ant.rx.z0_m = 0;
ant = updateantNAU(ant,fmcw,fig);

%% Create the SAR Scenario
%-------------------------------------------------------------------------%
sar.method = "Rectilinear";
sar.numX = 256;
sar.numY = 32;
sar.xStep_m = fmcw.lambda_m/4;
sar.yStep_m = fmcw.lambda_m*2;

sar.thetaMax_deg = 360;
sar.numTheta = 1024;
sar = updatesarNAU(sar,ant,fig);

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

%% Create target from random points
%-------------------------------------------------------------------------%
target.isGPU = true;
target.numTargetMax = 0;
target.o_x = 3e-3;
target.o_y = 3e-3;
target.o_z = 3e-3;
target.xOffset_m = im.x_m(end*3/8);
target.yOffset_m = im.y_m(end/4);
target.xOffset_m = -0.025; %-0.0125,0.0125
target.yOffset_m = 0.05; %-0.025,0.025
target.zOffset_m = im.z_m(end/2);
target.xStep_m = 1e-3;
target.yStep_m = 1e-3;
target.downSample = 4;
target.ampAdjust = 1.9;
target.ampMin = 0.5;
target.ampMax = 1;
target.dBMin = -25;
target = gettargetXYZpngANDrandNAU("circle.png",target,im,fig);
showImScenarioNAU(target,sar,fig);

%% Simulate Echo Signal
%-------------------------------------------------------------------------%
target.isAmplitudeFactor = true;
target.isMIMO = true;
tic
sarData = updatetargetNAU(target,sar,fmcw);
toc

%% Load noise
%-------------------------------------------------------------------------%
sim.noise = reshape(load("fmcw_v1_noise524288").emptyNoise,4,2,1,1,79,[]);

%% Add noise
%-------------------------------------------------------------------------%
sarData = addNoiseAWR1243(sarData,sim.noise,7e2);

%% Reconstruct Image
%-------------------------------------------------------------------------%
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

im.dBMin = -15;
im = uniform_SISO_2D_array_reconstructImage_3DNAU(sarDataPC,target,fmcw,ant,sar,im,fig);

%% Setup the loop
%-------------------------------------------------------------------------%
loop.numLoop = 1024;
loop.bulkName = "sar2DMIMOAWR1243_3D_pngORrand_yesAmp";
loop.dsPath = "../Datastores/" + loop.bulkName + "/";

% loop.wNoiseMax = 7e2;
% loop.wNoiseMin = 1e2;

loop.xOffsetMin = im.x_m(end*3/8);
loop.xOffsetMax = im.x_m(end*5/8);
loop.yOffsetMin = im.y_m(end/4);
loop.yOffsetMax = im.y_m(end*3/4);
loop.zOffsetMin = im.z_m(end/4);
loop.zOffsetMax = im.z_m(end/4);

loop.xStepMin = 0.5e-3;
loop.xStepMax = 1e-3;
loop.yStepMin = 0.5e-3;
loop.yStepMax = 1e-3;

loop.times = zeros(1,loop.numLoop);
loop.pngnames = ["circle.png","cutout1.png","diamond.png","square.png","star.png","triangle.png"];

mkdir(loop.dsPath)
%%% DOES THE DIRECTORY ALREADY EXIST?
%%% PLEASE DO NOT OVERWRITE!
mkdir(loop.dsPath + "Input/")
mkdir(loop.dsPath + "Output/")

fig.isFig = false;

%% Run the loop
%-------------------------------------------------------------------------%
for indLoop = 1:loop.numLoop
    tic
    % Get the latest target parameters
    target.xOffset_m = loop.xOffsetMin + (loop.xOffsetMax-loop.xOffsetMin)*rand();
    target.yOffset_m = loop.yOffsetMin + (loop.yOffsetMax-loop.yOffsetMin)*rand();
    target.zOffset_m = loop.zOffsetMin + (loop.zOffsetMax-loop.zOffsetMin)*rand();
    target.xStep_m = loop.xStepMin + (loop.xStepMax-loop.xStepMin)*rand();
    target.yStep_m = loop.yStepMin + (loop.yStepMax-loop.yStepMin)*rand();
    
    isPNG = randi(2)-1;
    if isPNG
        target.pngname = loop.pngnames(randi(length(loop.pngnames)));
        target.numTargetMax = 0;
    else
        target.pngname = [];
        target.numTargetMax = 512;
    end
    % Get and show the target
    target = gettargetXYZpngANDrandNAU(target.pngname,target,im,fig);
    
    % Get the SAR data
    sarData = updatetargetNAU(target,sar,fmcw);
    
    % Optional: add real noise (requires step above to load noise)
%     target.wNoise = loop.wNoiseMin + (loop.wNoiseMax-loop.wNoiseMin)*rand();
%     sarData = addNoiseAWR1243(sarData,sim.noise,1e3);
    
    % Convert multistatic-to-monostatic
    sarData_y_x_k = reshape(permute(sarData,[1,2,4,3,5]),[],sar.numX,fmcw.ADCSamples);
    if target.isMIMO
        sarDataPC = pc .* sarData_y_x_k;
    else
        sarDataPC = sarData_y_x_k;
    end

    % Consider the SISO virtual array
    sar.yStep_m = fmcw.lambda_m/4;
    
    % Reconstruct the image
    im = uniform_SISO_2D_array_reconstructImage_3DNAU(sarDataPC,target,fmcw,ant,sar,im,fig);
    
    % Copy the ideal image and SAR image
    Output = target.ideal3D;
    Input = im.sarImage;
    
    % Save the iteration
    saveInputOutputFiles(loop.dsPath,Input,Output,indLoop)
    
    % Show loop info
    loop.times(indLoop) = toc;
    showLoopInfo(loop,indLoop);
end


%% Save the data
%-------------------------------------------------------------------------%
if ~exist(loop.dsPath + loop.bulkName + ".mat",'file')
    save(loop.dsPath + loop.bulkName + ".mat",'fmcw','ant','sar','target','loop','im','-v7.3');
else
    warning("File: " + loop.bulkName + " already exists!")
end

%% Functions
%-------------------------------------------------------------------------%
function showLoopInfo(loop,indLoop)
loop.timeRemaining_s = mean(loop.times(1:indLoop))*(loop.numLoop - indLoop);
if loop.timeRemaining_s < 60
    disp(indLoop + "/" + loop.numLoop + "    Est. time remaining: " + loop.timeRemaining_s + " seconds")
elseif loop.timeRemaining_s < 60*60
    disp(indLoop + "/" + loop.numLoop + "    Est. time remaining: " + loop.timeRemaining_s/60 + " minutes")
elseif loop.timeRemaining_s < 60*60*24
    disp(indLoop + "/" + loop.numLoop + "    Est. time remaining: " + loop.timeRemaining_s/60/60 + " hours")
else
    disp(indLoop + "/" + loop.numLoop + "    Est. time remaining: " + loop.timeRemaining_s/60/60/24 + " days")
end
end