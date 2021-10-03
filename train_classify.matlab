clc;
clear all;
close all;
load Xtrain
load Xtest
load Ltrain
load ltest
%========================
% load the trainded images
Img=Xtrain;
figure('Name','Trained Images');
rw=ceil(size(Img,1)/3);
for ii=1:size(Img,1)
    Im{ii}=mat2gray(reshape(Img(ii,:),[80 100]));
    subplot(rw,3,ii);imshow(Im{ii},[]);    
end
Ig=Xtest;
figure('Name','Testing Query Image');
rc=ceil(size(Ig,1)/3);
for ii=1:size(Ig,1)
    Ic{ii}=mat2gray(reshape(Ig(ii,:),[80 100]));
    subplot(rw,3,ii);imshow(Ic{ii},[]);    
end


% == create a multi level stack dae ====
 hiddenSize1 = 100;
 autoenc1 = trainAutoencoder(Im,hiddenSize1, ...
     'MaxEpochs',10, ...
     'L2WeightRegularization',0.004, ...
     'SparsityRegularization',4, ...
     'SparsityProportion',0.25, ...
     'ScaleData', false);
 view(autoenc1)
%  figure,plotWeights(autoenc1);

feat1 = encode(autoenc1,Im);
hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',10, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.2, ...
    'ScaleData', false);

feat2 = encode(autoenc2,feat1);
softnet = trainSoftmaxLayer(feat2,Ltrain,'MaxEpochs',10);
view(softnet)

% create  a stack deep auto encoder
deepnet = stack(autoenc1,autoenc2,softnet);

%=== For classification =======
imageWidth = 80;
imageHeight = 100;
inputSize = imageWidth*imageHeight;
xTest = zeros(inputSize,numel(Ic));
for i = 1:numel(Ic)
    xTest(:,i) = Ic{i}(:);
end


% with a deep neural netwrok 
y = deepnet(xTest);
str={'Balayya','chiru'};
figure,
for i=1:numel(Ic)
    if sum(round(y(:,i)))~=0
    dx(i)=find(round(y(:,i))==1);
    else
    dx(i)=find(max(y(:,i))==y(:,i));
    end
    
    disp(['the person is identified as ', num2str(dx(i))]);
    subplot(rw,3,i);imshow(Ic{i},[]);title(str{dx(i)});  
end

varargin='face_run_1''lbpCode';
roc % roc calculation of the proposed work and LBP

    


