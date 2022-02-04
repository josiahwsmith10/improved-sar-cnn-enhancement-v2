function sarData = dataReadTSWNAU(fileName,sar,ant,fmcw)
%% Declare Optional Parameters

if ~isfield(sar,'isTwoDirectionScanning')
    sar.isTwoDirectionScanning = false;
end

%% Read from Bin File
fileID = fopen(fileName + ".bin",'r');
rawData4D = fread(fileID,'uint16','l') - 2^15;

%% Set default parameters
numXY = sar.numX*sar.numY; % number of total measurements

%% Reshape Row Data and Calculate Complex Row Data
rawData4D = reshape(rawData4D,2*ant.rx.numRx,[]);
rawData4D = rawData4D([1,3,5,7],:) + 1i*rawData4D([2,4,6,8],:);

%% Reshape Row Data Accordingly
try
    rawData4D = squeeze(reshape(rawData4D,ant.rx.numRx,fmcw.ADCSamples,ant.rx.numTx,1,1,numXY));
catch
    warning("Something is wrong with the number of captures!")
    fclose(fileID);
    return;
end

fclose(fileID);

%% Create Virtual Array
if ant.rx.numTx > 1
    rawData4D = reshape(permute(rawData4D,[1,3,2,4]),ant.rx.numRx*ant.rx.numTx,fmcw.ADCSamples,numXY);
end

%% Rearrange rawData if it is obtained after fast processing scanning scenario
rawData4D = reshape(rawData4D,ant.rx.numTx*ant.rx.numRx,fmcw.ADCSamples,sar.numX,sar.numY);
if isTwoDirectionScanning
    for n = 1:sar.numY
        % reverse all even vertical scans
        if rem(n-1,2)
            rawData4D(:,:,:,n) = flip(rawData4D(:,:,:,n),3);
        end
    end
end

rawData4D = reshape(rawData4D,ant.rx.numTx*ant.rx.numRx,fmcw.ADCSamples,[]);

%% Reshape Row Data for Old 3D FFT Proessing
sarData = reshape(rawData4D,ant.rx.numTx*ant.rx.numRx,fmcw.ADCSamples,sar.numX,sar.numY);

end %% End of data Read