function sarData = addNoiseAWGN(sarData,SNR_dB)
noise = randn(size(sarData)) + 1j*randn(size(sarData));

P_sar = powerSig(sarData(:));
P_n = powerSig(noise(:));

sarData = sarData/P_sar + 10^(-SNR_dB/10)*noise/P_n;
end

function p = powerSig(x)
p = sqrt(sum(abs(x).^2));
end