function sarData = phasedUpdatetargetNAU(target,sar,fmcw)

sar.xyz_m = double(sar.vx.xyz_m.');
target.xyz_m = double(target.xyz_m.');

waveform = phased.FMCWWaveform('SweepTime',fmcw.ADCSamples/fmcw.fS,'SweepBandwidth',fmcw.RampEndTime_s*fmcw.K,'SweepDirection','Up','SampleRate',fmcw.fS);

ant = design(patchMicrostrip,fmcw.fC);

transmitter = phased.Transmitter('PeakPower',1e-3,'Gain',30);
radiator = phased.Radiator('Sensor',ant,'OperatingFrequency',fmcw.fC);

receiver = phased.ReceiverPreamp('Gain',30,'NoiseFigure',4.5,'SampleRate',waveform.SampleRate);
collector = phased.Collector('Sensor',ant,'OperatingFrequency',fmcw.fC);

target.reflector = phased.RadarTarget('MeanRCS',100,'PropagationSpeed',physconst('lightspeed'),'OperatingFrequency',fmcw.fC);

channel = phased.WidebandFreeSpace('PropagationSpeed',physconst('lightspeed'),'OperatingFrequency',fmcw.fC,'SampleRate',fmcw.fS,'TwoWayPropagation',true);

% Get echo signal
sig = waveform();
txsig = transmitter(sig);

txsig = radiator(txsig,0);

txsig2 = channel(repmat(txsig(:),1,size(sar.xyz_m,2)),sar.xyz_m,target.xyz_m(:,1),zeros(size(sar.xyz_m)),zeros(3,1));
for indTarget = 2:size(target.xyz_m,2)
    txsig2 = txsig2 + channel(repmat(txsig(:),1,size(sar.xyz_m,2)),sar.xyz_m,target.xyz_m(:,indTarget),zeros(size(sar.xyz_m)),zeros(3,1));
end
txsig = target.reflector(txsig2);

% txsig = collector(txsig,zeros(2,size(txsig,2)));
txsig = receiver(txsig);
sarData = dechirp(txsig,sig).';

% Reshape echo signal
sarData = reshape(sarData,[sar.size,fmcw.ADCSamples]);
