% this file is to create database 
clc;
clear all;
close all;
d={'P1','P2'};
FD=vision.CascadeObjectDetector;
 kk=1;
h=waitbar(0,'Please wait creating database');
for dd=1:length(d)
    
    F=dir(d{dd})
    F=char(F.name)
    sz=size(F,1)-2;
   
    for ii=1:sz
    str=F(ii+2,:);
    cd(d{dd})    
    I=imread(str);
    cd ..
    fc=step(FD,I);
    crf=imcrop(I,fc(end,:));
    crf=imresize(crf,[80 100]);    
    crg=double(rgb2gray(crf));   
    FT{ii}=reshape(crg,[1, size(crg,1)*size(crg,2)]);
    [a1,b1,c1,d1]=dwt2(crg,'haar');
    [a2,b2,c2,d2]=dwt2(a1,'haar');     
    HF(ii)=entropy(a2)+entropy(b2)+entropy(c2)+entropy(d2)+entropy(b1)+entropy(c1)+entropy(d1);       
end
mnh=min(HF);mxh=max(HF);
for ii=1:sz
    m(ii)=(HF(ii)-mnh)./(mxh-mnh);
end
mum=mean(m);stm=std(m);
dx=find(m>=(mum+stm/2));
l=length(dx);
for pp=1:length(dx)
    FV(kk,:)=FT{dx(pp)};
    tmp=zeros(1,length(d));
    tmp(dd)=1;
    lb(:,kk)=tmp;
    kk=kk+1;
end      
waitbar(dd/length(d))   
end
close(h);


% randomly create the reaining and testing data
% TR=ceil(size(FV,1).*.8);
% TS=ceil(size(FV,1).*.5);
% 
% R1=unique(randi([1 size(FV,1)],1,TR));
% 
% R2=unique(randi([1 size(FV,1)],1,TS));
% tt=1;
% for cc=1:length(R2)
%     dp=find(R2(cc)==R1);
%     if numel(dp)==0
%         R2f(tt)=R2(cc);
%         tt=tt+1;
%     end
% end
% manual selection of training and testing dataset 

R1=[1 3 12 16 ];
R2f=[2 7  12   20 ];

Xtrain=FV(R1,:);
Xtest=FV(R2f,:);
Ltrain=lb(:,R1);
Ltest=lb(:,R2f);
save Xtrain Xtrain
save Xtest Xtest
save Ltrain Ltrain
save Ltest Ltest



