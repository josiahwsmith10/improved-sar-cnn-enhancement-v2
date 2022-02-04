% Test Bench for the Non-App Versions of the Utilities
%% Load in Saved Array and FMCW
% These can be created in the app
%-------------------------------------------------------------------------%
addpath(genpath("../../"))
ant = load("AWR1243").savedant;
fmcw = load("fmcw_v1").savedfmcw;
fig = initializeFiguresNAU();
fig.isFig = true;

%% Create Array 
%-------------------------------------------------------------------------%
ant.tx.z0_m = 0.25;
ant.rx.z0_m = 0.25;
ant = updateantNAU(ant,fmcw,fig);

%% Create the SAR Scenario
%-------------------------------------------------------------------------%
sar.method = "Rectilinear";
sar.numX = 256;
sar.numY = 16;
sar.xStep_m = fmcw.lambda_m/4;
sar.yStep_m = fmcw.lambda_m*2;

sar.thetaMax_deg = 360;
sar.numTheta = 1024;
sar = updatesarNAU(sar,ant,fig);

%% Set Imaging Parameters
%-------------------------------------------------------------------------%
im.nFFTx = 512;
im.nFFTy = 512;

im.numX = 256;
im.numY = 256;
im.x_m = linspace(0.1/im.numX-0.05,0.05,im.numX)';
im.y_m = linspace(0.1/im.numY-0.05,0.05,im.numY);

%% Create target from png and random points
%-------------------------------------------------------------------------%
target.numTargetMax = 25;
target.xStep_m = 0.5e-3;
target.yStep_m = 0.5e-3;
target.xOffset_m = -0.0125; %-0.0125,0.0125
target.yOffset_m = 0.0125; %-0.025,0.025
target.zOffset_m = 0;
target.o_x = 1e-3;
target.o_y = 1e-3;
target.ampMin = 0.5;
target.ampMax = 1;
% target.ampAdjust = 0.65; 128 x 128
target.ampAdjust = 0.3; % 1.9 if o_x is 2e-3
target.downSample = 2;
target.isGPU = true;
target = gettargetXYpngANDrandNAU("cutout1.png",target,im,fig);
showImScenarioNAU(target,sar,fig);

%% Simulate Echo Signal
%-------------------------------------------------------------------------%
target.isAmplitudeFactor = true;
target.isMIMO = true;
target.isGPU = true;
tic
sarData = updatetargetNAU(target,sar,fmcw);
toc

%% Load noise
%-------------------------------------------------------------------------%
sim.noise = reshape(load("fmcw_v1_noise524288").emptyNoise,4,2,1,1,79,[]);

%% Add noise
%-------------------------------------------------------------------------%
sarData = addNoiseAWR1243(sarData,sim.noise,5e2);

%% Reconstruct Image
%-------------------------------------------------------------------------%
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

im = uniform_SISO_2D_array_reconstructImage_2DNAU(sarDataPC,target,fmcw,ant,sar,im,fig);

%% Setup the loop
%-------------------------------------------------------------------------%
loop.numIterations = 8192;
loop.bulkName = "sar2DMIMOAWR1243_256x128_250mm_pngANDrand";
loop.pngnames = ["circle.png","cutout1.png","diamond.png","square.png","star.png","triangle.png"];

loop.xOffsetMax_m = 0.012;
loop.xOffsetMin_m = -0.012;
loop.yOffsetMax_m = 0.012;
loop.yOffsetMin_m = -0.012;

loop.wNoiseMax = 1.5e3;
loop.wNoiseMin = 1e2;

% mkdir("./saved/" + loop.bulkName)
%%% DOES THE DIRECTORY ALREADY EXIST? 
%%% PLEASE DO NOT OVERWRITE!

% targetAll(loop.numIterations) = target;
% imAll(loop.numIterations) = im;
% sarDataAll = single(zeros([size(sarData),loop.numIterations]));
% sarData_y_x_kAll = single(zeros([size(sarData_y_x_k),loop.numIterations]));
idealImageAll = single(zeros([size(target.ideal2D),loop.numIterations]));
sarImageAll = single(zeros([size(im.sarImage),loop.numIterations]));

fig.isFig = false;

%% Run the loop
%-------------------------------------------------------------------------%
for indLoop = 1:loop.numIterations
    % Show iteration number
    disp("")
    disp("Starting iteration #" + indLoop)
    tic
    
    % Choose png
    target.pngname = loop.pngnames(randi(length(loop.pngnames)));
    
    % Randomly select position
    target.xOffset_m = loop.xOffsetMin_m + (loop.xOffsetMax_m-loop.xOffsetMin_m)*rand();
    target.yOffset_m = loop.yOffsetMin_m + (loop.yOffsetMax_m-loop.yOffsetMin_m)*rand();
    
    % Get and show the target
    target = gettargetXYpngANDrandNAU(target.pngname,target,im,fig);
    showImScenarioNAU(target,sar,fig);
    
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
    im = uniform_SISO_2D_array_reconstructImage_2DNAU(sarDataPC,target,fmcw,ant,sar,im,fig);
    
    % Copy the ideal image and SAR image
    idealImageAll(:,:,indLoop) = target.ideal2D;
    sarImageAll(:,:,indLoop) = im.sarImage;
    
    % Save the iteration
%     save("./saved/" + loop.bulkName + "/iteration" + indLoop,"fmcw","ant","sar","target","im","fig","sarData","sarDataPC","-v7.3")
    
    
    % Show iteration number
    disp("Finished iteration #" + indLoop + " in " + toc + "sec");
end


%% Save the data
%-------------------------------------------------------------------------%
if ~exist(loop.bulkName + ".mat",'file')
    save("./saved/" + loop.bulkName,'fmcw','ant','sar','idealImageAll','sarImageAll','loop','-v7.3');
else
    warning("File: " + loop.bulkName + " already exists!")
end