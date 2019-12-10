error_2=20;
sj_temp=[];si_temp=[];
epoch=1:1:200;
wij_2=-0.5+rand(784,n_hid_neuron);
wjk_2=-0.5+rand(n_hid_neuron,col);
a=0.5; %momentum coefficient a=0.5
count=0;
J_W=[]; Jq_W=[];
% while error_2 >10
%     count=count+1;
% for    count=epoch
    rand_index_4000=randperm(4000);
    index_100 = rand_index_4000(1:100);
    Label_iter_100=Label_4000(index_100); % Training data label
    training_data_100=training_data_4000(index_100,:);
    
    for n=1:100                
        sj_2=sum(training_data_100(n,:).*wjk_2,2)';
        hid_layer_2=sigmoid(sum(training_data_100(n,:).*wjk_2,2))';        
        si_2=sum(hid_layer_2.*wij_2,2)';
        y_hat2=sigmoid(si_2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %         Output the first number in 10 bits, assuming 1st is '0'.
        y=training_data_100(n,:);
        difference_2=y-y_hat2;
        
        %         Updating delta_wij
        %         delta_i_q=difference_2.*fprime(si_2);
        delta_wij2= eta_o*(difference_2.*fprime(si_2))'*hid_layer_2;
        
        sum_i2=sum(wij_2.*(difference_2.*fprime(si_2))');
        delta_wjk2=eta_h*(fprime(sj_2).*sum_i2)'.*training_data_100(n,:);
        %         Updating delta_wjk
        
        
        
        if count>1
            % Update both wij & wjk, considering momentum, set a=0.5
            wij_2=wij_2+delta_wij2+a*delta_wij_previous;
            wjk_2=wjk_2+delta_wjk2+a*delta_wjk_previous;
            
        else
            % Update both wij & wjk, considering momentum, set a=0.5
            wij_2=wij_2+delta_wij2;
            wjk_2=wjk_2+delta_wjk2;
        end
        % Store the present delta_wij in a box as the previous delta_w for
        % momentum
        delta_wij_previous=delta_wij2;
        delta_wjk_previous=delta_wjk2;
        %%%%%%%%%
        for i=1:length(difference_2)
            error_temp(i)=0.5*difference_2(i)^2;
        end
        Jq_W(n,1)=sum(error_temp);
    end
    
    
%     J_W(count,1)=sum(Jq_W);
%     error_2=J_W(end);
%     data=[count error_2]
    
    %     layer_j_epoch=sigmoid(training_data_100*(wjk_2)');
    %     layer_i_epoch=sigmoid(layer_j_epoch*(wij_2)');
    %     [row_epoch col_epoch]=max(layer_i_epoch');
    %     label_epoch=col_epoch-1;
    %     n_correct_epoch=length(find(Label_iter_100==label_epoch));
    %     HitRate(count)=n_correct_epoch/100;
    %     error(count)=sum(error_temp);
    %     disp(error(count));
%     disp(count);
% end

%% Loss function(error) over epoch
% plot(1:size(J_W,1),J_W)
% xlabel('Epoch');
% ylabel('Overall Error');
title ('Overall Error Over each Epoch');
% Features
U = wjk_2;

for i=1:12
    for j = 1:12
        v = reshape(U((i-1)*12+j,:),28,28);
        subplot(12,12,(i-1)*12+j)
        image(64*v)
        colormap(gray(64));
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        set(gca,'dataaspectratio',[1 1 1]);
    end
end
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
    Calc_training=(wij_2*wjk_2*(NumberSet_training)')';
    Calc_testing=(wij_2*wjk_2*(NumberSet_testing)')';
    
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


