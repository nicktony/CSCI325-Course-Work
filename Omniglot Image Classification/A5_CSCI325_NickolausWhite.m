% % % % % % % % % % % % % % % % % % %
% Nickolaus White (CSCI325)
% % % % % % % % % % % % % % % % % % %


% Load in data
%---------------------------------------------------------------
downloadFolder = "C:\Users\white\Desktop";
downloadFolder2 = "C:\Users\white\Desktop\images_background";

url = "https://github.com/brendenlake/omniglot/raw/master/python";
urlTrain = url + "/images_background.zip";
urlTest = url + "/images_evaluation.zip";

filenameTrain = fullfile(downloadFolder,"images_background.zip");
filenameTest = fullfile(downloadFolder,"images_evaluation.zip");

dataFolderTrain = fullfile(downloadFolder,"images_background"); % Omniglot training data
dataFolderTest = fullfile(downloadFolder,"images_evaluation"); % Omniglot test data
dataFolderImds = fullfile(downloadFolder2,"Latin"); % Lowercase latin alphabet data

if ~exist(dataFolderTrain,"dir")
    fprintf("Downloading Omniglot training data set (4.5 MB)... ")
    websave(filenameTrain,urlTrain);
    unzip(filenameTrain,downloadFolder);
    fprintf("Done.\n")
end

if ~exist(dataFolderTest,"dir")
    fprintf("Downloading Omniglot test data (3.2 MB)... ")
    websave(filenameTest,urlTest);
    unzip(filenameTest,downloadFolder);
    fprintf("Done.\n")
end


% Specify Training and Validation Sets
%---------------------------------------------------------------
imdsTrainO = imageDatastore(dataFolderTrain, ...
    'IncludeSubfolders',true,'LabelSource','foldernames'); % Omniglot
imdsValidationO = imageDatastore(dataFolderTest, ...
    'IncludeSubfolders',true,'LabelSource','foldernames'); % Omniglot
imds = imageDatastore(dataFolderImds, ...
    'IncludeSubfolders',true,'LabelSource','foldernames'); % Lowercase alphabet


% Specify Size of Image (Both Data Sets)
%---------------------------------------------------------------
imgTrain = readimage(imdsTrainO,1);
imgValidation = readimage(imdsValidationO,1);
imgAlphabet = readimage(imds,1);

if imgTrain >= imgValidation
    if imgTrain >= imgAlphabet
        img = imgTrain;
    end
else
    img = imgAlphabet;
end

size(img)


%---------------------------------------------------------------
%---------------------Data set to Omniglot----------------------
%---------------------------------------------------------------


% Preview Some of the Training Images in the Omniglot Dataset
%---------------------------------------------------------------
figure (1);
perm = randperm(10000,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imdsTrainO.Files{perm(i)});
end


% Calculate Number of Images in Each Category (Omniglot)
%---------------------------------------------------------------
labelCountTrain = countEachLabel(imdsTrainO)
labelCountValidation = countEachLabel(imdsValidationO)


% Define the Network Structure (Omniglot)
%---------------------------------------------------------------
layersO = [
    imageInputLayer([105 105 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(47)
    softmaxLayer
    classificationLayer];


% Train Dataset (Omniglot)
%---------------------------------------------------------------
optionsO = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidationO, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

netO = trainNetwork(imdsTrainO,layersO,optionsO);


% Validation (Omniglot)
%---------------------------------------------------------------
YPred = classify(netO,imdsValidationO);
YValidation = imdsValidationO.Labels;

accuracyO = sum(YPred == YValidation)/numel(YValidation)


%---------------------------------------------------------------
%--------------Data Changed to Lowercase Alphabet---------------
%---------------------------------------------------------------


% Preview Some of the Training Images in the Lowercase Alphabet Dataset
%---------------------------------------------------------------
figure (2);
perm = randperm(500,20);
for i = 1:20
    subplot(4,5,i);
    imshow(imds.Files{perm(i)});
end


% Split Data Set (Lowercase Alphabet)
%---------------------------------------------------------------
numTrainFiles = 15;
[imdsTrainA,imdsValidationA] = splitEachLabel(imds,numTrainFiles,'randomize');


% Define the Network Structure (Lowercase Alphabet)
%---------------------------------------------------------------
layersA = [
    imageInputLayer([105 105 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(26)
    softmaxLayer
    classificationLayer];


% Train Dataset (Lowercase Alphabet)
%---------------------------------------------------------------
optionsA = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidationA, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

netA = trainNetwork(imdsTrainA,layersA,optionsA);


% Validation (Lowercase Alphabet)
%---------------------------------------------------------------
YPred = classify(netA,imdsValidationA);
YValidation = imdsValidationA.Labels;

accuracyA = sum(YPred == YValidation)/numel(YValidation)


% Save File Contents, End of Program
%---------------------------------------------------------------
filename = 'A5_CSCI325_NickolausWhite.mat';
save(filename)




