%% SPIKING NEURAL NETWORK FOR HANDWRITING RECOGNITION (MNIST)---UNSUPERVISED LEARNING APPROACH
%  coded by Abhronil Sengupta on 24th May, 2015

%% clear data
clc;
clear all;
close all;

%% pre-processing of input images for digits 0,1,2 and 3 with 250 training instances for each digit
num_train=100;  % no of training instances of each image % 각각의 이미지마다 트레이닝을 몇 번 할 것인가
load P_vs_I_20kT_PW.mat; % 파일 로드
load MNIST_Greyscale_0_9.mat; % load MNIST dataset % 파일 로드
num_digits=10;   % no of digits used for recognition % 인식에 사용된 자릿수 즉, 아웃풋 노드의 개수

temp=diag(ones(1,num_digits)); % 레이블링을 위한 임시행렬 % diag는 정사각형의 대각행렬을 만드는 것으로 temp는 1의 숫자를 하나씩 가진 10 * 10 행렬이 만들어진다.
y=[];           % label for each image % 정답 레이블을 담기 위한 것으로 추측

for i=0:num_train-1 % 100회 for문 돌림
    x(1+num_digits*i,:)=Zero(:,1+i); % 100개의 MNIST dataset안에 들어있는 Zero sample을 x의 각 1 + num_digits * i의 행에 집어 넣었다.
    x(2+num_digits*i,:)=One(:,1+i); % 100개의 MNIST dataset안에 들어있는 One sample을 x의 각 2 + num_digits * i의 행에 집어 넣었다.
    x(3+num_digits*i,:)=Two(:,1+i); 
    x(4+num_digits*i,:)=Three(:,1+i);
    x(5+num_digits*i,:)=Four(:,1+i);
    x(6+num_digits*i,:)=Five(:,1+i);
    x(7+num_digits*i,:)=Six(:,1+i);
    x(8+num_digits*i,:)=Seven(:,1+i);
    x(9+num_digits*i,:)=Eight(:,1+i);
    x(10+num_digits*i,:)=Nine(:,1+i); % Nine까지 위와 동일하며 총 1000개의 데이터 샘플을 가진 x행렬이 만들어짐.   
    
    y=[y;temp]; % 정답 레이블로 추측
end;



%% images applied as input spike trains and synaptic weights modified by STDP

% Fixed parameters % 상수
timeStepS = 1;                          % 1 msec % 1000 msec == 1 sec이다.
epochs = num_train*num_digits;          % No of training epochs % 훈련 횟수 = 100 * 10
InNeurons = size(x,2);                  % No of input neurons % 인풋 뉴런 개수 = 784 % size(행렬, 차원)으로 현재와 같은 경우 2차원 안에 있는 784개의 요소를 나타낸 것
num_PF=0;                               % 보류
% Tunable parameters

OpNeurons = 10;                         % No of output neurons % 아웃풋 뉴런 개수 = 10
durationS = 290;                         % 40 msec simulation for each image % 신호를 받는 주기 
tau_EPSP = 50;                               % EPSP/STDP response time in msec % 


Inh = 500;                                % Inhibitory strength % 억제성 연접후 막전위 % 억제 힘 
K_leak = 0.018;                             % 고정된 상수
Kconst = 300;


Ki = .05e05;                            %5000     % scaling factor for probability % 확률에 대한 척도 계수
Kf = .05e09;                                    % 50000000

del_K = 0.018;
tau_STDP1 = 4.5;
tau_STDP2 = 5;
tau_Inh = 50;
eta1 = 0.03;                              % Learning rate % Estimated Time of Arrival의 약자로 도착 예정 시간을 의미
eta2 = 0.01;




sum_weights = zeros(1,OpNeurons);                          % 1X10 zeros 행렬 [0,0,0,0,0,0,0,0,0,0]
volt = zeros(OpNeurons);                                  % 10X10 zeros 행렬
K = Ki*ones(1,OpNeurons);                                 % 1X10 ones 행렬 * Ki
weights_e = 0.13*rand(InNeurons,OpNeurons);                 % synaptic weight matrix 784X10
weights_com = zeros(280,280);                                % 280X280 행렬
% Run the simulation for 1000 images (each digit 0-3 presented with 250 % 1000개 이미지에 대해 시뮬레이션을 실행한다(각 숫자 0-3은 250으로 표시됨).
% different instances for 10 times) % 10회에 걸쳐 각기 다른 예
% Update and show image  % 이미지 업데이트 및 표시
for num=0: OpNeurons-1 % 10회 for문 돌림
    weights_com(1:28, num*28+1:(num+1)*28) = reshape(weights_e(:, num+1), [28,28]); % weights_e를 784X1로 쪼개서 28X28로 reshape한 후 weights_com에 넣는다.
    colormap(jet);                          % 제트 컬러맵을 가져온다.
    imagesc(weights_com)                    % 스케일링된 색으로 이미지를 표시한다.
    drawnow                                 % for문을 다 돌고 난 후 그래프가 그려진다.
    pause(0.04);                            % 일시정지 기능을 한다.
end

for tt = 1:1    % 1회 for문 돌림
    tt  
for i = 1:epochs %replace by epochs % 1000회 for문 돌림
    fprintf('\n  epoch is : %d \n',i);  % epoch 횟수 출력
    
    % initial conditions % 초기 조건
    spikesPerS=255/4*x(i,:);            % 1x784 matrix. * (255 / 4)
    spikes = zeros(InNeurons,durationS/timeStepS);              % 784x290 zeros matrix
    EPSP = zeros(InNeurons,durationS/timeStepS+tau_EPSP);       % 784x(290 + 50) zeros matrix
    u = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);          % 10x(290 + 50) zeros matrix
    prob = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);       % 10x(290 + 50) zeros matrix
%     z = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);
    I = zeros(1,OpNeurons);                                     % 1x10 zeros matrix
    t_post = zeros(1,OpNeurons);                                % 1x10 zeros matrix
    t_pre = zeros(1,InNeurons);                                 % 1x784 zeros matrix

    
    % generate spikes for a particular input according to Poisson process 
    % 포아송 프로세스에 따라 특정 입력에 대해 스파이크 생성
    for train = 1:InNeurons % 784회 for문 돌림
        vt = rand(1,durationS/timeStepS); % 1x290 균일하게 생성된 난수 생성 
        if x(i,train)>0  % 푸아송 프로세스에 따라서 x matrix의 현재 epoch부터 추출
           spikes(train, :) = ((spikesPerS(1,train)*timeStepS)/1000 > vt); 
           % col for문을 돌면서 = 784열에서 요소 하나씩 뽑아서 스파이크 생성
        end;
    end
    
    % generate EPSP corresponding to spike train % 스파이크 훈련에 해당하는 EPSP를 생성한다
    for train = 1:InNeurons % 784회 for문 돌림
        for t = 1:durationS/timeStepS  % 290회 for문 돌림
            if spikes(train,t) == 1     % spikes의 요소가 1일 경우
                EPSP(train,t:t+tau_EPSP-1) = ones(1,tau_EPSP);  % EPSP의 50만큼 1의 수가 채워진다.
            end;
        end;
    end;
    
    %Run the simulation % 시뮬레이션 실행
    for t = 1:durationS/timeStepS + tau_EPSP-1            % 290/1 + 50 - 1 = 339회 for문 돌림
        for train = 1:InNeurons                       % 784회 for문 돌림
            if t<= durationS/timeStepS                  % t가 290보다 작거나 같을 경우
            if spikes(train,t) == 1                     % spikes의 요소가 1일 경우
                t_pre(1,train) = t;                     % 해당 위치에 timing 저장
                for l = 1:OpNeurons                     % 10회 for문 돌림
                    weights_e(train,l) = weights_e(train,l) - eta2*weights_e(train,l)*exp((t_post(l)-t_pre(train))/tau_STDP2) ;  % 가중치 업데이트 w = w - 0.01 * w * e^((t - y) / 5)
                    if weights_e(train,l)>1         % 1보다 클 경우
                        weights_e(train,l)=1;       % 값을 1로 설정
                    elseif weights_e(train,l)<0     % 0보다 작을 경우
                        weights_e(train,l)=0;       % 값을 0으로 설정
                    end;
                end;
            end;
            end;
        end;
        for j = 1:OpNeurons                 % 10회 for문 돌림
            I(j) = 0;                       % 0으로 초기화
            for kk = 1:OpNeurons            % 10회 for문 돌림
                if t-t_post(kk) < tau_Inh && kk~=j && t_post(kk)~=0     % 억제해주는 역할로 추측
                    I (j) = Inh;
                end;
            end;
       
            u(j, t+1) = weights_e(:, j)'*EPSP(:, t) - I(j); %current sum
            if u(j, t+1) < 0
                u(j, t+1) =0 ;
            end;
            
            % u / K
            i_curr(j, t+1) = u(j, t+1) / K(j);
            % i_curr square * 400 * 0.0000000005
            pwr_mtj(j, t+1) = i_curr(j, t+1) * i_curr(j, t+1) * 400*0.5e-09;
            
            % u / K > 0.00013
            if u(j, t+1) / K(j) > 1.3e-04
                prob(j, t+1) = 1;
            % u / K < 0.000003
            elseif u(j, t+1) / K(j) < 3e-05
                prob(j, t+1) = 0;
            else
                % interpolation. 해당 쿼리 점에서 보간된 값을 반환. pchip으로 보간
                % Ich는 샘플 점을 포함하며 P는 대응값을 포함한다.
                % u/K는 쿼리 점의 좌표를 포함한다.
                prob(j, t+1) = interp1(Ich, P, u(j,t+1)/K(j), 'pchip');
            end;
            
            if rand < prob(j,t+1) % 0-1 사이의 난수보다 클 경우
                num_PF = num_PF+1;
                z(j, t+1) = 1;
                t_post(j) = t+1;
                K(j) = K(j) + del_K * K(j);
                if K(j) > Kf% 500,000,000 보다 클 경우
                    K(j) = Kf; % 500,000,000
                end;
                
                for k = 1:InNeurons % 784회 for문 돌림
                           %if EPSP(k) == 1
                             weights_e(k,j) = weights_e(k,j) + eta1*weights_e(k,j)*exp((t_pre(k)-t_post(j))/tau_STDP1);
                           %else
                                %weights_e(k,j) = weights_e(k,j) - eta2/10;
                           %end;
                    % 
                    if weights_e(k,j) > 1
                        weights_e(k,j) = 1;
                    elseif weights_e(k,j) < 0
                        weights_e(k,j) = 0;
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
    for num=0: OpNeurons-1      % 10회 for문 돌림
        weights_com(1:28,num*28+1:(num+1)*28)=reshape(weights_e(:,num+1),[28,28]);   % weights_e를 784X1로 쪼개서 28X28로 reshape한 후 weights_com에 넣는다.
    colormap(jet);                          % 제트 컬러맵을 가져온다.
    imagesc(weights_com)                    % 스케일링된 색으로 이미지를 표시한다.
    drawnow                                 % for문을 다 돌고 난 후 그래프가 그려진다.
    pause(0.04);                            % 일시정지 기능을 한다.
    end

    %Calculate voltage and conductances
    
    
end;
end;  

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
%     spikes = zeros(InNeurons,durationS/timeStepS,i);
%     EPSP = zeros(InNeurons,durationS/timeStepS+tau_EPSP);
%     u = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);
%     
%     z = zeros(OpNeurons,durationS/timeStepS+tau_EPSP);
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
% %                 z(j,t+1,i) = 1;
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



