clc;
clear;

%vymněn vrstvy v síti a dotrénuj

imds=imageDatastore('datastorage','IncludeSubfolders',true, 'LabelSource','foldernames');
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7);

net=resnet50;
lgraph = layerGraph(net);
inputSize = net.Layers(1).InputSize;

%které vrstvy v síti je mozné vymenint
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
[learnableLayer,classLayer]


numClasses = numel(categories(imdsTrain.Labels));

if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);


newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%vykreslí aktuální kostru
figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
plot(lgraph)
ylim([0,10])




miniBatchSize = 10;
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',3e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress');


net = trainNetwork(imdsTrain,lgraph,options);

save net

