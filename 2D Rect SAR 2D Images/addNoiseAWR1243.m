function sarData = addNoiseAWR1243(sarData,noise,SNR_dB)
numNoiseSample = size(noise,6);
numX = size(sarData,3);
numY = size(sarData,4);
numXY = numX*numY;

noise = permute(noise(:,:,:,:,:,randperm(numNoiseSample,numXY)),[1,2,3,4,6,5]);
noise = reshape(noise,size(sarData));

P_sar = powerSig(sarData(:));
P_n = powerSig(noise(:));

sarData = sarData/P_sar + 10^(-SNR_dB/10)*noise/P_n;
end

function p = powerSig(x)
p = sqrt(sum(abs(x).^2));
end