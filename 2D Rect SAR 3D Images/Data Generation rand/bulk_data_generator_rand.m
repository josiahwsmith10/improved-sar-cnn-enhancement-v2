% Test Bench for the Non-App Versions of the Utilities
addpath(genpath("../../"))
addpath(genpath("C:/Data"))

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
target.numTargetMax = 128;
target.o_x = 2e-3;
target.o_y = 2e-3;
target.o_z = 2e-3;
target.ampMin = 0.5;
target.ampMax = 1;
target.dBMin = -25;
target = gettargetXYZrandNAU(target,im,fig);
showImScenarioNAU(target,sar,fig);

%% Simulate Echo Signal
%-------------------------------------------------------------------------%
target.isAmplitudeFactor = false;
target.isMIMO = true;
tic
sarData = updatetargetNAU(target,sar,fmcw);
toc

%% Load noise
%-------------------------------------------------------------------------%
sim.noise = reshape(load("fmcw_v1_noise65536.mat").emptyNoise,4,2,1,1,79,[]);

%% Add noise
%-------------------------------------------------------------------------%
sarData = addNoiseAWR1243(sarData,sim.noise,-10);

%% Add AWGN noise
%-------------------------------------------------------------------------%
sarData = addNoiseAWGN(sarData,20);

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

im.dBMin = -25;
im = uniform_SISO_2D_array_reconstructImage_3DNAU(sarDataPC,target,fmcw,ant,sar,im,fig);

%% Setup the loop
%-------------------------------------------------------------------------%
loop.numLoop = 1024;
loop.bulkName = "sar2DMIMOAWR1243_3D_rand_noAmp";
loop.dsPath = "../Datastores/" + loop.bulkName + "/";
loop.dsPath = "C:/Data/2D Rect SAR 3D Images/" + loop.bulkName + "/";

loop.isAWR1243noise = false;
loop.isAWGN = true;
loop.minSNR_dB = -10;
loop.maxSNR_dB = 20;
loop.SNR_dB = zeros(1,loop.numLoop);

loop.times = zeros(1,loop.numLoop);

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
    % Get and show the target
    target = gettargetXYZrandNAU(target,im,fig);
    
    % Get the SAR data
    sarData = updatetargetNAU(target,sar,fmcw);
    
    % Optional: add real noise (requires step above to load noise)
    if loop.isAWGN
        loop.SNR_dB(indLoop) = loop.minSNR_dB + (loop.maxSNR_dB-loop.minSNR_dB)*rand();
        sarData = addNoiseAWGN(sarData,loop.SNR_dB(indLoop));
    elseif loop.isAWR1243noise
        loop.SNR_dB(indLoop) = loop.minSNR_dB + (loop.maxSNR_dB-loop.minSNR_dB)*rand();
        sarData = addNoiseAWR1243(sarData,sim.noise,loop.SNR_dB(indLoop));
    end
        
    
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