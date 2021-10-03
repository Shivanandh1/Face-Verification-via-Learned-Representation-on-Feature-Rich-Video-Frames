clc;
clear all;
close all;
% the first is to select the frame section a in the base paper
FD=vision.CascadeObjectDetector;
F=dir('P2');
F=char(F.name);
sz=size(F,1)-2;
total_frames=sz;
nf=20;figure,
for ii=1:nf
    str=F(ii+2,:);
    cd P2
    I=imread(str);
    cd ..
    fc=step(FD,I);
    crf=imcrop(I,fc(end,:));
    
    crf=imresize(crf,[80 100]);
    
    crg=double(rgb2gray(crf));
    
    [a1,b1,c1,d1]=dwt2(crg,'haar');
    [a2,b2,c2,d2]=dwt2(a1,'haar');
     
    HF(ii)=entropy(a2)+entropy(b2)+entropy(c2)+entropy(d2)+entropy(b1)+entropy(c1)+entropy(d1);
    
   subplot(4,5,ii),imshow(crf)
end
mnh=min(HF);mxh=max(HF);

for ii=1:nf
    m(ii)=(HF(ii)-mnh)./(mxh-mnh);
end
mum=mean(m);stm=std(m);
dx=find(m>=(mum+stm/2));
selected_frames=length(dx);
figure,bar(m,1:nf,15);xlabel('Feature Richness Value');
ylabel('No of frames');grid on;ylim([1 nf+2])
