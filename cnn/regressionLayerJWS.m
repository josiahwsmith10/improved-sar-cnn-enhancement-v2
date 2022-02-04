function layer = regressionLayerJWS( varargin )
%%%% MODIFIED BY JOSIAH W SMITH
%%% Penalizes zero outputs, useful for regression on sparse images
%
% regressionLayer   Regression output layer for a neural network
%
%   layer = regressionLayer() creates a regression output layer for
%   a neural network. The regression output layer holds the name of the
%   loss function that is used for training the network.
%
%   layer = regressionLayer('PARAM1', VAL1) specifies optional
%   parameter name/value pairs for creating the layer:
%
%       'Name'                    - A name for the layer. The default is
%                                   ''.
%       'ResponseNames'           - Names of the responses, specified as a
%                                   string vector, a  cell array of
%                                   character vectors, or empty. At
%                                   training time, the software
%                                   automatically sets the response names
%                                   according to the training data. The
%                                   default is {}.
%
%   Example:
%       % Create a regression output layer.
%
%       layer = regressionLayer();
%
%   See also nnet.cnn.layer.RegressionOutputLayer, classificationLayer.
%
%   <a href="matlab:helpview('deeplearning','list_of_layers')">List of Deep Learning Layers</a>

%   Copyright 2016-2018 The MathWorks, Inc.

% Parse the input arguments
args = iParseInputArguments(varargin{:});

internalLayer = MeanSquaredErrorJWS( ...
    args.Name);
internalLayer.ResponseNames = args.ResponseNames;

% Pass the internal layer to a function to construct
layer = nnet.cnn.layer.RegressionOutputLayer(internalLayer);

end

function inputArguments = iParseInputArguments(varargin)
parser = iCreateParser();
parser.parse(varargin{:});
inputArguments = iConvertToCanonicalForm(parser);
end

function p = iCreateParser()
p = inputParser;
defaultName = '';
defaultResponseNames = {};
addParameter(p, 'Name', defaultName, @iAssertValidLayerName);
addParameter(p, 'ResponseNames', defaultResponseNames, ...
    @iAssertValidResponseNames);
end

function iAssertValidLayerName(name)
nnet.internal.cnn.layer.paramvalidation.validateLayerName(name);
end

function iAssertValidResponseNames(names)
nnet.internal.cnn.layer.paramvalidation.validateResponseNames(names);
end

function names = iConvertResponseNamesToCanonicalForm(names)
names = nnet.internal.cnn.layer.paramvalidation...
    .convertResponseNamesToCanonicalForm(names);
end

function inputArguments = iConvertToCanonicalForm(p)
inputArguments = struct;
inputArguments.ResponseNames = iConvertResponseNamesToCanonicalForm(...
    p.Results.ResponseNames);
inputArguments.Name = char(p.Results.Name); % make sure strings get converted to char vectors
end
