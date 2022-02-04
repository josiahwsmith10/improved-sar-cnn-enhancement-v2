function demoNetworkPSFSlicesXYZ_BPA(im,ant,sar,fmcw,target,fcnnNet)
%% PSF
target.xyz_m = single([mean(im.x_m),mean(im.y_m),mean(im.z_m)]);
target.amp = 1;

target.zOffset_m = mean(im.z_m);

im = getImageXYZ_BPA(target,sar,ant,fmcw,im);
psf.bpa = im.sarImage;

im = getImageXYZ(target,sar,ant,fmcw,im);

psf.rma = im.sarImage;
psf.enhanced = predict(fcnnNet,im.sarImage);

if nnz(psf.enhanced) == 0
    warning("PSF ALL ZEROS!")
    return;
end

rma.x = psf.rma(:,end/2,end/2);
rma.y = psf.rma(end/2,:,end/2);
rma.z = squeeze(psf.rma(end/2,end/2,:));

bpa.x = psf.bpa(:,end/2,end/2);
bpa.y = psf.bpa(end/2,:,end/2);
bpa.z = squeeze(psf.bpa(end/2,end/2,:));

enhanced.x = psf.enhanced(:,end/2,end/2);
enhanced.y = psf.enhanced(end/2,:,end/2);
enhanced.z = squeeze(psf.enhanced(end/2,end/2,:));

figure
subplot(131)
plot(im.x_m,rma.x,"-")
hold on
plot(im.x_m,bpa.x,"--")
plot(im.x_m,enhanced.x,"--")
legend("RMA","BPA","Enhanced")
title("PSF X Slice")
xlabel("x (m)")
ylabel("Normalized")

subplot(132)
plot(im.y_m,rma.y,"-")
hold on
plot(im.y_m,bpa.y,"--")
plot(im.y_m,enhanced.y,"--")
legend("RMA","BPA","Enhanced")
title("PSF Y Slice")
xlabel("y (m)")
ylabel("Normalized")

subplot(133)
plot(squeeze(im.z_m),rma.z,"-")
hold on
plot(squeeze(im.z_m),bpa.z,"--")
plot(squeeze(im.z_m),enhanced.z,"--")
legend("RMA","BPA","Enhanced")
title("PSF Z Slice")
xlabel("z (m)")
ylabel("Normalized")

psf.rma = db(psf.rma/max(psf.rma(:)));
psf.bpa = db(psf.bpa/max(psf.bpa(:)));
psf.enhanced = db(psf.enhanced/max(psf.enhanced(:)));

psf.rma(psf.rma<-100) = -100;
psf.bpa(psf.bpa<-100) = -100;
psf.enhanced(psf.enhanced<-100) = -100;

rma.x = psf.rma(:,end/2,end/2);
rma.y = psf.rma(end/2,:,end/2);
rma.z = squeeze(psf.rma(end/2,end/2,:));

bpa.x = psf.bpa(:,end/2,end/2);
bpa.y = psf.bpa(end/2,:,end/2);
bpa.z = squeeze(psf.bpa(end/2,end/2,:));

enhanced.x = psf.enhanced(:,end/2,end/2);
enhanced.y = psf.enhanced(end/2,:,end/2);
enhanced.z = squeeze(psf.enhanced(end/2,end/2,:));
figure
subplot(131)
plot(im.x_m,rma.x,"-")
hold on
plot(im.x_m,bpa.x,"--")
plot(im.x_m,enhanced.x,"--")
legend("RMA","BPA","Enhanced")
title("PSF X Slice")
xlabel("x (m)")
ylabel("dB")

subplot(132)
plot(im.y_m,rma.y,"-")
hold on
plot(im.y_m,bpa.y,"--")
plot(im.y_m,enhanced.y,"--")
legend("RMA","BPA","Enhanced")
title("PSF Y Slice")
xlabel("y (m)")
ylabel("dB")

subplot(133)
plot(squeeze(im.z_m),rma.z,"-")
hold on
plot(squeeze(im.z_m),bpa.z,"--")
plot(squeeze(im.z_m),enhanced.z,"--")
legend("RMA","BPA","Enhanced")
title("PSF Z Slice")
xlabel("z (m)")
ylabel("dB")
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

function im = getImageXYZ_BPA(target,sar,ant,fmcw,im)
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
sar.y_m = sar.x_m;
im = uniform_SISO_2D_array_reconstructImage_3D_BPANAU(sarDataPC,target,fmcw,ant,sar,im,fig);
im.sarImage = im.sarImage/max(im.sarImage(:));
end