%% SPIKING NEURAL NETWORK FOR HANDWRITING RECOGNITION (MNIST)

%% clear data
clc;
clear all;
close all;
tic
%% Dataset Load
oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
filenameImagesTrain = 'train-images.idx3-ubyte';
filenameLabelsTrain = 'train-labels.idx1-ubyte';
filenameImagesTest = 't10k-images.idx3-ubyte';
filenameLabelsTest = 't10k-labels.idx1-ubyte';

XTrain = processImagesMNIST(filenameImagesTrain);
YTrain = processLabelsMNIST(filenameLabelsTrain);
XTest = processImagesMNIST(filenameImagesTest);
YTest = processLabelsMNIST(filenameLabelsTest);

sh_i = randperm(size(XTrain,4)); %shuffle training data
XTrain = XTrain(:,:,:,sh_i);
YTrain = YTrain(sh_i);

load I_nor2.mat;
%% 
num_training_samples=5000;  % no of training instances of each image

for i=1:num_training_samples
    x(i,:)=reshape(extractdata(XTrain(:,:,1,i)),[1,784]);
end

% Fixed parameters
timeStepS = 1;                          % 1 msec
epochs = num_training_samples;          % No of training epochs
InNeurons = 784;                  % No of input neurons
num_out_fire=0;
% mem_th=160;
mem_th0=80;
mem_factor=1.02; %Vth increase for homeostasis
mem_max=400;

% Tunable parameters

OpNeurons = 100;                         % No of output neurons (10의 배수)
durationS = 290;                         % 40 msec simulation for each image
tau_EPSP = 50;                               % EPSP/STDP response time in msec
tau_Inh = 50;

loc_OpNeurons=ones(1,OpNeurons);

for i=1:OpNeurons/10
    for k=0:9
        loc_OpNeurons(k*OpNeurons/10+1:k*OpNeurons/10+OpNeurons/10)=k;
    end
end

sh_l = randperm(size(loc_OpNeurons,2));
loc_OpNeurons=loc_OpNeurons(sh_l); %Output Neuron supervised label shuffling


W_scale=1; %weight update scale
TW_scale=1;
STDP_TW=20/TW_scale;

Inh = 50000;                                % Inhibitory strength

sum_weights = zeros(1,OpNeurons);
volt = zeros(OpNeurons);
mem_th = mem_th0*ones(1,OpNeurons);
weights_e = randi(32,InNeurons,OpNeurons)/32;  % synaptic weight matrix (1~32)
weights_com = zeros(56,280);

for num=0: OpNeurons-1
    weights_com(fix(num/10)*28+1:fix(num/10)*28+28,mod(num,10)*28+1:mod(num,10)*28+28)=reshape(weights_e(:,num+1),[28,28]);
    colormap('jet');
    imagesc(weights_com)
    drawnow
    pause(0.04);
end

for tt = 1:1
    tt  
for i = 1:epochs %replace by epochs
    fprintf('\n  epoch is : %d \n',i);
    % initial conditions
    spikesPerS=255/4*x(i,:);
    spikes = zeros(InNeurons,durationS/timeStepS);
    EPSP = zeros(InNeurons,durationS/timeStepS+tau_EPSP);
    u = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);
    I = zeros(1,OpNeurons);    
    t_post = zeros(1,OpNeurons);
    t_pre = zeros(1,InNeurons);

    
    % generate spikes for a particular input according to Poisson process
    for train = 1:InNeurons
        vt = rand(1,durationS/timeStepS);
        if x(i,train)>0
           spikes(train, :) = ((spikesPerS(1,train)*timeStepS)/1000 > vt);
        end;
    end
    
    % generate EPSP corresponding to spike train
    for train = 1:InNeurons
        for t = 1:durationS/timeStepS
            if spikes(train,t) == 1
                EPSP(train,t:t+tau_EPSP-1) = ones(1,tau_EPSP);
            end;
        end;
    end;
    
    %Run the simulation
    for t = 1:durationS/timeStepS+tau_EPSP-1
        for train = 1:InNeurons
            if t<= durationS/timeStepS
            if spikes(train,t) == 1
                t_pre(1,train) = t;               
                for l = 1:OpNeurons                  
%                     if (t_pre(train)-t_post(l))< STDP_TW+1 && (t_pre(train)-t_post(l)>0) && (double(YTrain(i))==loc_OpNeurons(l))
%                         weights_e(train,l) = weights_e(train,l) - fix(4/W_scale)/32 ; %DEP
% %                         weights_e(train,l) = weights_e(train,l) - fix(STDP(t_pre(train)-t_post(l))/W_scale)/2;%DEP
%                     end
                    if weights_e(train,l)>1
                        weights_e(train,l)=1;
                    end
                    
                    if weights_e(train,l)<0.1
                        weights_e(train,l)=0;
                    end;
                end;
            end;
            end;
        end;
        for j = 1:OpNeurons
            I(j) = 0;
            for kk = 1:OpNeurons
                if t-t_post(kk) < tau_Inh && kk~=j && t_post(kk)~=0
                    I (j) = Inh;
                end;
            end;
       
            u(j,t+1) = weights_e(:,j)'*EPSP(:,t)-I(j); %current sum
            if u(j,t+1)<0
                u(j,t+1)=0;
            end;
 
            if u(j,t+1)>mem_th(j)
                num_out_fire=num_out_fire+1;
                t_post(j)=t+1;
%                 u(j,t+2:t+22)=0;
                mem_th(j)=mem_th(j)*mem_factor;
%                 if mem_th(j) > mem_max;
%                     mem_th(j) = mem_max;
%                 end;
                for k = 1:InNeurons   
                    if(t_post(j)-t_pre(k)<STDP_TW+1)&&(t_post(j)-t_pre(k)>0)&& double(YTrain(i))==loc_OpNeurons(j)
                             weights_e(k,j) = weights_e(k,j) + fix(STDP(t_post(j)-t_pre(k))/W_scale)/32;%POT                             
                    end
                    
                    if weights_e(k,j)>1
                        weights_e(k,j)=1;
                    end
                    
                    if weights_e(k,j)<0.1
                        weights_e(k,j)=0;
                    end;
                end;
            end;
%             K(j) = K(j) - K_leak;
%             if K(j) < Ki
%                 K(j) = Ki;
%             end;
        end;
    end;
    % Update and show image
    for num=0: OpNeurons-1
        weights_com(fix(num/10)*28+1:fix(num/10)*28+28,mod(num,10)*28+1:mod(num,10)*28+28)=reshape(weights_e(:,num+1),[28,28]);
        colormap('jet');
        imagesc(weights_com/32)
        drawnow
        pause(0.04);
    end

    %Calculate voltage and conductances
    
    
end;
end;  
toc
% 
% %Plot current vs learning epoch
% i_max_avg = mean(i_max,2);
% prob_MTJ_avg = mean(prob_MTJ,2);
% i = 1:100;
% plot(i,i_max_avg(1:100));
% figure();
% plot(i,prob_MTJ_avg(1:100));
%     
% Inh = 500;                                % Inhibitory strength
% Kconst = .3e06;
% tau_Inh = 50;
% class = zeros(OpNeurons,num_digits);
% spikes = zeros(InNeurons,durationS/timeStepS,epochs);
% z = zeros(OpNeurons,durationS/timeStepS+tau_EPSP,epochs);
% prob = zeros(OpNeurons,durationS/timeStepS+tau_EPSP,epochs);
% 
% for num=1:OpNeurons
%    sum_weights(num)=sum(weights_e(:,num));
% end;
% 
% Gunity=0.1/max(sum_weights)/400;
% volt=1/(Kconst*Gunity)
% 
% 
% for i = 1:epochs %replace by epochs
% % initial conditions
%     spikesPerS=255/4*x(i,:);
% %     spikes = zeros(InNeurons,durationS/timeStepS,i);
%     EPSP = zeros(InNeurons,durationS/timeStepS+tau_EPSP);
%     u = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);
%     
% %     z = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);
%     I = zeros(1,OpNeurons);    
%     t_post = zeros(1,OpNeurons);
%     t_pre = zeros(1,InNeurons);
% 
%     
%     % generate spikes for a particular input according to Poisson process
%     for train = 1:InNeurons
%         vt = rand(1,durationS/timeStepS);
%         if x(i,train)>0
%            spikes(train, :,i) = ((spikesPerS(1,train)*timeStepS)/1000 > vt);
%         end;
%     end
%     
%     % generate EPSP corresponding to spike train
%     for train = 1:InNeurons
%         for t = 1:durationS/timeStepS
%             if spikes(train,t,i) == 1
%                 EPSP(train,t:t+tau_EPSP-1) = ones(1,tau_EPSP);
%             end;
%         end;
%     end;
%     
%     %Run the simulation
%     for t = 1:durationS/timeStepS+tau_EPSP-1
%         for train = 1:InNeurons
%             if t<= durationS/timeStepS
%             if spikes(train,t,i) == 1
%                 t_pre(1,train) = t;
%                
%             end;
%             end;
%         end;
%         for j = 1:OpNeurons
%             I(j) = 0;
%             for kk = 1:OpNeurons
%                 if t-t_post(kk) < tau_Inh && t_post(kk)~=0
%                     I (j) = Inh;
%                 end;
%             end;
%        
%             u(j,t+1) = weights_e(:,j)'*EPSP(:,t)-I(j);
%             if u(j,t+1)<0
%                 u(j,t+1)=0;
%             end;
%             if u(j,t+1)/Kconst>1.3e-04
%                 prob(j,t+1,i) = 1;
%             elseif u(j,t+1)/Kconst<3e-05
%                 prob(j,t+1,i) = 0;
%             else
%                 prob(j,t+1,i) = interp1(Ich,P,u(j,t+1)/Kconst,'pchip');
%             end;
%             if rand < prob(j,t+1,i)
%                 z(j,t+1,i) = 1;
%                 t_post(j)=t+1;
%                
%                 
%             end;
%             
%         end;
%  
%     end;
%     [val,ind] = max(sum(z(:,:,i),2));
%     [vald,digit] = max(y(i,:)); 
%     class(ind,digit) = class(ind,digit) + 1;    
%     
%    
% end;
% 
% 
% max_class = 0;
% for i = 1:size(class,1)%num of rows of class
%     max_class = max_class + max(class(i,:));
% end;
% 
% accuracy = max_class/num_train/num_digits
% 
% 
% count_spikes=zeros(OpNeurons,100);
% for i=1:100
%     for j=1:9
%         for k=1:340
%             count_spikes(j,i)=count_spikes(j,i)+z(j,k,i);
%         end
%     end
% end
% 
% 

