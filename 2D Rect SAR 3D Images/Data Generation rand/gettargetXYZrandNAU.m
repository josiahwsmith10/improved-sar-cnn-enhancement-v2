function target = gettargetXYZrandNAU(target,im,fig)
%% Inputs
%   filename
%   target
%       isGPU
%       numTargetMax
%       o_x
%       o_y
%       o_z
%       ampMin
%       ampMax
%       dBMin

%% Create the target locations and amplitudes
target.numTarget = randi(target.numTargetMax);
target.xyz_m = single([im.x_m(1) + (im.x_m(end)-im.x_m(1))*rand(target.numTarget,1),im.y_m(1) + (im.y_m(end)-im.y_m(1))*rand(target.numTarget,1),im.z_m(1) + (im.z_m(end)-im.z_m(1))*rand(target.numTarget,1)]);
target.amp = target.ampMin + (target.ampMax-target.ampMin)*rand(1,target.numTarget);

target.zOffset_m = mean(im.z_m);

fail = true;
tic

while fail
    R_same = pdist2(target.xyz_m,target.xyz_m) + 1e3*eye(target.numTarget);
    indGood = min(R_same,[],1)>(target.o_y*4);
    numGood = sum(indGood);
    if numGood == target.numTarget
        fail = false;
    else
        xyz_m_good = target.xyz_m(indGood,:);
        xyz_m_new = single([im.x_m(1) + (im.x_m(end)-im.x_m(1))*rand(target.numTarget-size(xyz_m_good,1),1),im.y_m(1) + (im.y_m(end)-im.y_m(1))*rand(target.numTarget-size(xyz_m_good,1),1),im.z_m(1) + (im.z_m(end)-im.z_m(1))*rand(target.numTarget-size(xyz_m_good,1),1)]);
        target.xyz_m = cat(1,xyz_m_good,xyz_m_new);
        
        if toc > 10
            indGood = min(R_same,[],1)>(target.o_y*4);
            target.xyz_m = target.xyz_m(indGood,:);
            target.amp = target.amp(indGood);
            target.numTarget = sum(indGood);
            warning("Could not place targets correctly in 10s, reducing number of targets")
            break;
        end
    end
end

%% Create the ideal reflectivity function
target.ideal3D = single(zeros(im.numX,im.numY,im.numZ));
if ~target.isGPU
    x_m = im.x_m;
    y_m = im.y_m;
    z_m = im.z_m;
    o_x = target.o_x;
    o_y = target.o_y;
    o_z = target.o_z;
    xyz_m = target.xyz_m;
else
    x_m = gpuArray(im.x_m);
    y_m = gpuArray(im.y_m);
    z_m = gpuArray(im.z_m);
    o_x = gpuArray(target.o_x);
    o_y = gpuArray(target.o_y);
    o_z = gpuArray(target.o_z);
    xyz_m = gpuArray(target.xyz_m);
end

for indTarget = 1:target.numTarget
    temp = single(exp(-(o_x)^(-2)*(x_m-xyz_m(indTarget,1)).^2 -(o_y)^(-2)*(y_m-xyz_m(indTarget,2)).^2 -(o_z)^(-2)*(z_m-xyz_m(indTarget,3)).^2));
    tempMax = max(temp(:));
    if tempMax == 0
        tempMax = 1;
    end
    
    temp = temp*target.amp(indTarget)/tempMax;
    target.ideal3D = target.ideal3D + temp;
end
if abs(max(target.ideal3D(:))-max(target.amp)) > 5e-2
    warning("Amplitude error! The ideal image is distorted!")
end
target.ideal3D(target.ideal3D>1) = 1;
target.ideal3D(target.ideal3D<0) = 0;

target.ideal3D = single(gather(target.ideal3D));

%% Show the reflectivity function
if ~fig.isFig
    return;
end
h = fig.Target2D.h;
f = fig.Target2D.f;
plotXYZdB(h,f,target.ideal3D,im.x_m,im.y_m,im.z_m,[],target.dBMin,"Original Reflectivity Function, " + target.numTarget + " targets",12)
