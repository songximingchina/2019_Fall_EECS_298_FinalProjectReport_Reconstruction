clear all

Labels5000=load('MNISTnumLabels5000.txt');
Images5000=load('MNISTnumImages5000.txt');
clc
fprintf('Please input number of hidden layers to be 1, or the system would collapse ;)\n');
n_hid_neuron=input('Please input the number of the hidden neurons: \n');
col=size(Images5000,2);
hid_layer=zeros(1,n_hid_neuron);
OutputLabel=zeros(10,10);
%Learning Rate
eta_o=0.01; eta_h=0.01;
% Building the output label, from '0' to '9'
for n=1:10
    OutputLabel(n,n)=1;
end
clear n;
%Building Weights wij & wjk between three layers
wij=-0.5+rand(10,n_hid_neuron);
wjk=-0.5+rand(n_hid_neuron,col);
% Select random 4000 datapoints out of 5000 for training.
% Select the other 1000 datapoints for testing.
rand_index_5000=randperm(5000);
index_4000 = rand_index_5000(1:4000);
index_1000 = rand_index_5000(4001:5000);
Label_4000 = Labels5000(index_4000)'; % The actual number value of the index
Label_1000 = Labels5000(index_1000)';
training_data_4000=Images5000(index_4000,:);
testing_data_1000=Images5000(index_1000,:);

%SGD. Selecting 100 different datapoints out of index_4000 during each epoch
error=2;
sj_temp=[];si_temp=[];
epoch=1:1:200;
a=0.5; %momentum coefficient a=0.5
% while error >0.1
for    count=epoch
    rand_index_4000=randperm(4000);
    index_100 = rand_index_4000(1:100);
    Label_iter_100=Label_4000(index_100); % Training data label
    training_data_100=training_data_4000(index_100,:);
    
    
    for n=1:100
        %         for j=1:n_hid_neuron %calculate hj of the hidden layer
        %             for k=1:size(training_data_4000,2)
        %                 w_x_jk=wjk(j,k)*training_data_100(n,k);
        %                 sj_temp=[sj_temp,w_x_jk];
        %             end
        %             sj(j)=sum(sj_temp); % The jth value of sj
        %             sj_temp=[]; %Clear the temp box for next iteration
        %             hid_layer(j)=sigmoid(sj(j)); % The jth value of hj
        %         end
        
        sj=sum(training_data_100(n,:).*wjk,2)';
        hid_layer=sigmoid(sum(training_data_100(n,:).*wjk,2))';
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %         for i=1:10  % Calculate the 10-bit output value using sigmoid function
        %             for j=1:n_hid_neuron
        %                 w_h_ij=wij(i,j)*hid_layer(j);
        %                 si_temp=[si_temp,w_h_ij]; %#ok<*AGROW>
        %             end
        %             si(i)=sum(si_temp); %#ok<*SAGROW>
        %             si_temp=[];
        %             y_hat(i)=sigmoid(si(i));
        %         end
        %
        si=sum(hid_layer.*wij,2)';
        y_hat1=sigmoid(si);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         Output the first number in 10 bits, assuming 1st is '0'.
        y=OutputLabel(Label_iter_100(n)+1,:);
        difference=y-y_hat1;
        
        
        %         Updating delta_wij
        %         for i=1:10
        %             for j=1:n_hid_neuron
        %                 delta_wij(i,j)=eta_o*difference(i)*fprime(si(i))*hid_layer(j);
        %             end
        %         end
        delta_wij1= eta_o*(difference.*fprime(si))'*hid_layer;
        
        sum_i1=sum(wij.*(difference.*fprime(si))');
        delta_wjk1=eta_h*(fprime(sj).*sum_i1)'.*training_data_100(n,:);
        %         Updating delta_wjk
        %         for j=1:n_hid_neuron
        %             for i=1:10
        %                 sum_i_temp(i)=wij(i,j)*difference(i)*fprime(si(i));
        %             end
        %             sum_i=sum(sum_i_temp);
        %             for k=1:size(training_data_4000,2)
        %                 delta_wjk(j,k)=eta_h*fprime(sj(j))*sum_i*training_data_100(n,k);
        %             end
        %         end
        
        
        if count>1
            % Update both wij & wjk, considering momentum, set a=0.5
            wij=wij+delta_wij1+a*delta_wij_previous;
            wjk=wjk+delta_wjk1+a*delta_wjk_previous;
            
        else
            % Update both wij & wjk, considering momentum, set a=0.5
            wij=wij+delta_wij1;
            wjk=wjk+delta_wjk1;
        end
        % Store the present delta_wij in a box as the previous delta_w for
        % momentum
        delta_wij_previous=delta_wij1;
        delta_wjk_previous=delta_wjk1;
        %%%%%%%%%
        
    end
    layer_j_epoch=sigmoid(training_data_100*(wjk)');
    layer_i_epoch=sigmoid(layer_j_epoch*(wij)');
    [row_epoch col_epoch]=max(layer_i_epoch');
    label_epoch=col_epoch-1;
    n_correct_epoch=length(find(Label_iter_100==label_epoch));
    HitRate(count)=n_correct_epoch/100;
    for i=1:length(difference)
        error_temp(i)=0.5*difference(i)^2;
    end
    error(count)=sum(error_temp);
         disp(count);
    %     a=input('epoch \n');
    %     disp(count);
    %     fprintf('\n');
    % if count==50
    %     continue
    % end
    
end
% end
Time_Series_of_Error=1-HitRate;
figure(1);
plot(epoch,error);
xlabel('Epoch');
ylabel('Error');
figure(2);
plot(epoch,Time_Series_of_Error);
xlabel('Epoch');
ylabel('Time Series of Error');
%% Performance
layer_j_training=sigmoid(training_data_4000*(wjk)');
layer_i_training=sigmoid(layer_j_training*(wij)');
layer_j_testing=sigmoid(testing_data_1000*(wjk)');
layer_i_testing=sigmoid(layer_j_testing*(wij)');

[row_training col_training]=max(layer_i_training');
[row_testing col_testing]=max(layer_i_testing');

label_training_final=col_training-1;
label_testing_final=col_testing-1;

n_correct_training=length(find(label_training_final==Label_4000));
n_correct_testing=length(find(label_testing_final==Label_1000));

% plotconfusion(0:9,[Label_4000',label_training_final']);
% plotconfusion(0:9,[Label_1000',label_testing_final']);
% a=[0 1 0 0 0 0 0 0 0 0];
% b=[0 0 1 0 0 0 0 0 0 0];
% plotconfusion(0:9,[a',b'])











