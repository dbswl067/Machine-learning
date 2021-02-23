%% SPIKING NEURAL NETWORK FOR HANDWRITING RECOGNITION (MNIST)---UNSUPERVISED LEARNING APPROACH
%  coded by Abhronil Sengupta on 24th May, 2015

%% clear data
clc;
clear all;
close all;

%% pre-processing of input images for digits 0,1,2 and 3 with 250 training instances for each digit
num_train=100;  % no of training instances of each image
load P_vs_I_20kT_PW.mat;
load MNIST_Greyscale_0_9.mat; % load MNIST dataset
num_digits=10;   % no of digits used for recognition

temp=diag(ones(1,num_digits));
y=[];           % label for each image

for i=0:num_train-1
    x(1+num_digits*i,:)=Zero(:,1+i);
    x(2+num_digits*i,:)=One(:,1+i);
    x(3+num_digits*i,:)=Two(:,1+i);
    x(4+num_digits*i,:)=Three(:,1+i);
    x(5+num_digits*i,:)=Four(:,1+i);
    x(6+num_digits*i,:)=Five(:,1+i);
    x(7+num_digits*i,:)=Six(:,1+i);
    x(8+num_digits*i,:)=Seven(:,1+i);
    x(9+num_digits*i,:)=Eight(:,1+i);
    x(10+num_digits*i,:)=Nine(:,1+i);    
    
    y=[y;temp];
end;



%% images applied as input spike trains and synaptic weights modified by STDP

% Fixed parameters
timeStepS = 1;                          % 1 msec
epochs = num_train*num_digits;          % No of training epochs
InNeurons = size(x,2);                  % No of input neurons
num_PF=0;
% Tunable parameters

OpNeurons = 10;                         % No of output neurons
durationS = 290;                         % 40 msec simulation for each image
tau_EPSP = 50;                               % EPSP/STDP response time in msec


Inh = 500;                                % Inhibitory strength
K_leak = 0.018;
Kconst = 300;


Ki = .05e05;                                 % scaling factor for probability
Kf = .05e09;
del_K = 0.018;
tau_STDP1 = 4.5;
tau_STDP2 = 5;
tau_Inh = 50;
eta1 = 0.03;                              % Learning rate
eta2 = 0.01;




sum_weights = zeros(1,OpNeurons);
volt = zeros(OpNeurons);
K = Ki*ones(1,OpNeurons);
weights_e = 0.13*rand(InNeurons,OpNeurons);  % synaptic weight matrix
weights_com = zeros(280,280);
% Run the simulation for 1000 images (each digit 0-3 presented with 250
% different instances for 10 times)
% Update and show image
for num=0: OpNeurons-1
    weights_com(1:28,num*28+1:(num+1)*28)=reshape(weights_e(:,num+1),[28,28]);
    colormap(jet);
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
    prob = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);
%     z = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);
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
                    weights_e(train,l) = weights_e(train,l) - eta2*weights_e(train,l)*exp((t_post(l)-t_pre(train))/tau_STDP2) ;
                    if weights_e(train,l)>1
                        weights_e(train,l)=1;
                    elseif weights_e(train,l)<0
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
            i_curr(j,t+1) = u(j,t+1)/K(j);
            pwr_mtj(j,t+1) = i_curr(j,t+1)*i_curr(j,t+1)*400*0.5e-09;
            if u(j,t+1)/K(j)>1.3e-04
                prob(j,t+1) = 1;
            elseif u(j,t+1)/K(j)<3e-05
                prob(j,t+1) = 0;
            else
                prob(j,t+1) = interp1(Ich,P,u(j,t+1)/K(j),'pchip');
            end;
            if rand < prob(j,t+1)
                num_PF=num_PF+1;
                z(j,t+1) = 1;
                t_post(j)=t+1;
                K(j) = K(j) + del_K*K(j);
                if K(j) > Kf
                    K(j) = Kf;
                end;
                for k = 1:InNeurons
                           %if EPSP(k) == 1
                             weights_e(k,j) = weights_e(k,j) + eta1*weights_e(k,j)*exp((t_pre(k)-t_post(j))/tau_STDP1);
                           %else
                                %weights_e(k,j) = weights_e(k,j) - eta2/10;
                           %end;
                    if weights_e(k,j)>1
                        weights_e(k,j)=1;
                    elseif weights_e(k,j)<0
                        weights_e(k,j)=0;
                    end;
                end;
            end;
            K(j) = K(j) - K_leak;
            if K(j) < Ki
                K(j) = Ki;
            end;
        end;
        i_max(i,t+1) = max(i_curr(:,t+1));
        prob_MTJ(i,t+1) = max(prob(:,t+1));
    end;
    pwr_mtj_avg(i)=max(max(pwr_mtj));
    % Update and show image
    for num=0: OpNeurons-1
        weights_com(1:28,num*28+1:(num+1)*28)=reshape(weights_e(:,num+1),[28,28]);
        colormap(jet);
        imagesc(weights_com)
        drawnow
        pause(0.04);
    end

    %Calculate voltage and conductances
    
    
end;
end;  

%Plot current vs learning epoch
i_max_avg = mean(i_max,2);
prob_MTJ_avg = mean(prob_MTJ,2);
i = 1:100;
plot(i,i_max_avg(1:100));
figure();
plot(i,prob_MTJ_avg(1:100));
    
Inh = 500;                                % Inhibitory strength
Kconst = .3e06;
tau_Inh = 50;
class = zeros(OpNeurons,num_digits);
spikes = zeros(InNeurons,durationS/timeStepS,epochs);
z = zeros(OpNeurons,durationS/timeStepS+tau_EPSP,epochs);
prob = zeros(OpNeurons,durationS/timeStepS+tau_EPSP,epochs);

for num=1:OpNeurons
   sum_weights(num)=sum(weights_e(:,num));
end;

Gunity=0.1/max(sum_weights)/400;
volt=1/(Kconst*Gunity)


for i = 1:epochs %replace by epochs
% initial conditions
    spikesPerS=255/4*x(i,:);
    spikes = zeros(InNeurons,durationS/timeStepS,i);
    EPSP = zeros(InNeurons,durationS/timeStepS+tau_EPSP);
    u = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);
    
    z = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);
    I = zeros(1,OpNeurons);    
    t_post = zeros(1,OpNeurons);
    t_pre = zeros(1,InNeurons);

    
    % generate spikes for a particular input according to Poisson process
    for train = 1:InNeurons
        vt = rand(1,durationS/timeStepS);
        if x(i,train)>0
           spikes(train, :,i) = ((spikesPerS(1,train)*timeStepS)/1000 > vt);
        end;
    end
    
    % generate EPSP corresponding to spike train
    for train = 1:InNeurons
        for t = 1:durationS/timeStepS
            if spikes(train,t,i) == 1
                EPSP(train,t:t+tau_EPSP-1) = ones(1,tau_EPSP);
            end;
        end;
    end;
    
    %Run the simulation
    for t = 1:durationS/timeStepS+tau_EPSP-1
        for train = 1:InNeurons
            if t<= durationS/timeStepS
            if spikes(train,t,i) == 1
                t_pre(1,train) = t;
               
            end;
            end;
        end;
        for j = 1:OpNeurons
            I(j) = 0;
            for kk = 1:OpNeurons
                if t-t_post(kk) < tau_Inh && t_post(kk)~=0
                    I (j) = Inh;
                end;
            end;
       /////
            u(j,t+1) = weights_e(:,j)'*EPSP(:,t)-I(j);
            if u(j,t+1)<0
                u(j,t+1)=0;
            end;
            if u(j,t+1)/Kconst>1.3e-04
                prob(j,t+1,i) = 1;
            elseif u(j,t+1)/Kconst<3e-05
                prob(j,t+1,i) = 0;
            else
                prob(j,t+1,i) = interp1(Ich,P,u(j,t+1)/Kconst,'pchip');
            end;
            if rand < prob(j,t+1,i)
%                 z(j,t+1,i) = 1;
                t_post(j)=t+1;
               
                
            end;
            
        end;
 
    end;
    [val,ind] = max(sum(z(:,:,i),2));
    [vald,digit] = max(y(i,:)); 
    class(ind,digit) = class(ind,digit) + 1;    
    
   
end;


max_class = 0;
for i = 1:size(class,1)%num of rows of class
    max_class = max_class + max(class(i,:));
end;

accuracy = max_class/num_train/num_digits


count_spikes=zeros(OpNeurons,100);
for i=1:100
    for j=1:9
        for k=1:340
            count_spikes(j,i)=count_spikes(j,i)+z(j,k,i);
        end
    end
end



