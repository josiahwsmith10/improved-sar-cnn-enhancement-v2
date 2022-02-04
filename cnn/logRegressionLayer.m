classdef logRegressionLayer < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    
    methods
        function layer = logRegressionLayer(name)
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = 'Josiahs custom log loss function';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the log loss between
            % the predictions Y and the training targets T.
            
            loss = -sum(T .* log(Y) + (1-T) .* log(1-Y),'all');
        end
    end
end