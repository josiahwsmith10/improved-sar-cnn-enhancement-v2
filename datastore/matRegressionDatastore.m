classdef matRegressionDatastore < matlab.io.Datastore & ...
        matlab.io.datastore.Shuffleable & ...
        matlab.io.datastore.Partitionable & ...
        matlab.io.datastore.MiniBatchable
    
    properties
        inDatastore
        outDatastore
        MiniBatchSize
        inSize
        outSize
    end
    
    properties(SetAccess = protected)
        NumObservations
    end
    
    properties(Access = private)
        CurrentFileIndex
    end
    
    methods
        function ds = matRegressionDatastore(dsPath)
            % ds = matFilesDatastore(dsPath) creates a file-to-file
            % regression datastore from the folder dsPath
            
            % Create input file datastore
            ds.inDatastore = fileDatastore(fullfile(dsPath + "Input/"),...
                "ReadMode","file","ReadFcn",@load,"FileExtensions",".mat");
            ds.outDatastore = fileDatastore(fullfile(dsPath + "Output/"),...
                "ReadMode","file","ReadFcn",@load,"FileExtensions",".mat");
            
            ds.inDatastore = transform(ds.inDatastore,@(data) inputPreprocess(data));
            ds.outDatastore = transform(ds.outDatastore,@(data) outputPreprocess(data));
            
            ds.NumObservations = numel(ds.inDatastore.UnderlyingDatastore.Files);
            
            % Initialize datastore properties.
            ds.CurrentFileIndex = 1;
            ds.inSize = size(ds.inDatastore.preview{1});
            ds.outSize = size(ds.outDatastore.preview{1});
        end
        
        function tf = hasdata(ds)
            % tf = hasdata(ds) returns true if more data is available.
            
            tf = ds.CurrentFileIndex - 1 ...
                <= ds.NumObservations;
        end
        
        function [data,info] = read(ds)
            % [data,info] = read(ds) read one data sample
            
            ds.CurrentFileIndex = ds.CurrentFileIndex + 1;
            if hasdata(ds)
                ds.reset();
            end

            info = struct;
            data = table(ds.inDatastore.read,ds.outDatastore.read);
        end
        
        function reset(ds)
            % reset(ds) resets the datastore to the start of the data.
            reset(ds.inDatastore);
            reset(ds.outDatastore);
            ds.CurrentFileIndex = 1;
        end
        
        function dsNew = shuffle(ds)
            % dsNew = shuffle(ds) shuffles the files and the corresponding
            % labels in the datastore.
            
            % Create copy of datastore.
            dsNew = copy(ds);
            dsNew.inDatastore = copy(ds.inDatastore);
            dsNew.outDatastore = copy(ds.outDatastore);
            
            % Shuffle files and corresponding labels.
            numObservations = dsNew.NumObservations;
            indRand = randperm(numObservations);
            dsNew.inDatastore.UnderlyingDatastore.Files = dsNew.inDatastore.UnderlyingDatastore.Files(indRand);
            dsNew.outDatastore.UnderlyingDatastore.Files = dsNew.outDatastore.UnderlyingDatastore.Files(indRand);
        end
        
        function subds = partition(ds, numPartitions, indPartition)
            subds = copy(ds);
            subds.inDatastore = partition(ds.inDatastore, numPartitions, indPartition);
            subds.outDatastore = partition(ds.outDatastore, numPartitions, indPartition);
            subds.NumObservations = numel(subds.inDatastore.UnderlyingDatastore.Files);
            reset(subds);
        end
    end
    
    methods(Access = protected)
        function n = maxpartitions(ds)
            n = ds.NumObservations;
        end
    end
    
    methods (Hidden = true)
        function frac = progress(ds)
            % frac = progress(ds) returns the percentage of observations
            % read in the datastore.
            
            frac = (ds.CurrentFileIndex - 1) / ds.NumObservations;
        end
    end
end

function img = inputPreprocess(data)
img = data.Input;
img = img/max(abs(img(:)));
img = {img};
end

function img = outputPreprocess(data)
img = data.Output;
img = img/max(abs(img(:)));
img = {img};
end