clear all
Labels5000=load('MNISTnumLabels5000.txt');
Images5000=load('MNISTnumImages5000.txt');
%%
wij_4=-0.5+rand(784,144);
wjk_4=-0.5+rand(144,784);


rand_index_5000=randperm(5000);
index_4000 = rand_index_5000(1:4000);
index_1000 = rand_index_5000(4001:5000);
Label_4000 = Labels5000(index_4000)'; % The actual number value of the index
Label_1000 = Labels5000(index_1000)';
training_data_4000=Images5000(index_4000,:);
testing_data_1000=Images5000(index_1000,:);

rho=0.005;
M=100;   % number of data points in the dataset
beta=3;
lamda=0.001;
eta_o4=0.1;
eta_h4=0.1;
a=0.5;
Error_storage=[];

% wij_4=-0.5+rand(784,144);
% wjk_4=-0.5+rand(144,784);
for m=1:200 % 200 epochs
    
    rand_index_4000=randperm(4000);
    index_100 = rand_index_4000(1:100);
    Label_iter_100=Label_4000(index_100); % Training data label
    training_data_100=training_data_4000(index_100,:);
    for n=1:100
        %Calculate value of each layer with existing weights
        sj_4 = training_data_100 * wjk_4';
        hid_layer_4 = sigmoid(sj_4);
        si_4 = hid_layer_4 * wij_4';
        y_hat_4 = sigmoid(si_4);
        difference_4 = training_data_100 - y_hat_4;        
        % Calculate Sparseness
        rho_j_hat = sum(hid_layer_4)/M;
        KL = rho*log10(rho./rho_j_hat)+(1-rho)*log10((1-rho)./(1-rho_j_hat));        
        % Calculate Original Loss Function J
        Error_each_data = sum(difference_4.^2,2);
        Error_4 = 0.5*Error_each_data;  %Error of all 100 testing data points, 100*1        
        % Calculate New Loss Function
        Error_4_new = Error_4(n) + beta*sum(KL);  
        
       
        % Update wij_2 and wjk_2 using new delta_j_q
        % Update wij using weight decay
        delta_i_q = difference_4(n,:) .* fprime(si_4(n,:));
        delta_wij4= eta_o4*(difference_4(n,:).*fprime(si_4(n,:)))'*hid_layer_4(n,:)-lamda*wij_4;
        % Update wjk
        delta_j_q = (delta_i_q*wij_4 + beta*((1-rho)./(1-rho_j_hat)-rho./rho_j_hat)).*fprime(sj_4(n,:));
        delta_wjk4 = eta_h4 * delta_j_q' * training_data_100(n,:);
        if n>1
            % Update both wij & wjk, considering momentum, set a=0.5
            wij_4=wij_4+delta_wij4+a*delta_wij_previous;
            wjk_4=wjk_4+delta_wjk4+a*delta_wjk_previous;    
        else
            % Update both wij & wjk, considering momentum, set a=0.5
            wij_4=wij_4+delta_wij4;
            wjk_4=wjk_4+delta_wjk4;
        end
        
        % Store the present delta_wij in a box as the previous delta_w for
        % momentum
        delta_wij_previous=delta_wij4;
        delta_wjk_previous=delta_wjk4;
        
        %         wij_4 = wij_4+delta_wij4-lamda*wij_4;
%         break
    end
    disp(m);
%     if mod(m,10)==0
        Error_storage=[Error_storage,Error_4_new];
%     end
%     break
end
%% Ploting Error for every 10 epochs (from epoch 10 to epoch 200)
% Plotting the error of training data
  class=10:10:200;
  plot(class,Error_storage(class));

%% Performance
box_training=[];
box_testing=[];
for i=1:10                              
    position_temp_training=(find(Label_4000==(i-1)))';
    position_temp_testing=(find(Label_1000==(i-1)))';
    NumberSet_training=training_data_4000(position_temp_training,:);
    NumberSet_testing=testing_data_1000(position_temp_testing,:);
    Size_NumberSet_training = length(position_temp_training);
    Size_NumberSet_testing = length(position_temp_testing);
    
    % Loss function of specific number
    Calc_training=( wij_4 * wjk_4*(NumberSet_training)')';
    Calc_testing=(wij_4*wjk_4*(NumberSet_testing)')';
    
    difference_perf_training=training_data_4000(position_temp_training,:)-Calc_training;
    TotalLoss_training=0.5*sum(difference_perf_training.^2,2);
    AvgLoss_training=sum(TotalLoss_training,1)/size(TotalLoss_training,1);
    box_training(i)=AvgLoss_training;
    
    difference_perf_testing=testing_data_1000(position_temp_testing,:)-Calc_testing;
    TotalLoss_testing=0.5*sum(difference_perf_testing.^2,2);
    AvgLoss_testing=sum(TotalLoss_testing,1)/size(TotalLoss_testing,1);
    box_testing(i)=AvgLoss_testing;
end

bar(0:1:9,(vertcat(box_training, box_testing))');
legend('Training Set','Testing Set');
title('Loss Function Comparison ');
xlabel('Number of Digit');
ylabel('Loss Function');
%% 









