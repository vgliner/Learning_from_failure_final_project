%% Plot of training and test
folder_name= 'C:\Users\vgliner\OneDrive - JNJ\Desktop\Private\PhD\Courses\236802\Final project\Learning_from_failure_extension\LfF_data\'
ResultsFolderInfo = dir(folder_name)
legend_strs={'q=0.95','q=0.05','q=0.25','q=0.45','q=0.65','q=0.80','Vanilla'}
for i=3:length(ResultsFolderInfo)
    full_name = [folder_name ResultsFolderInfo(i).name];
    Data = importfile(full_name, [2, Inf]); 
    subplot(2,1,1)
    plot(Data(:,1),Data(:,2))
    xlabel('Epoch')
    ylabel('Accuracy %')
    title('Accuracy on Training Set')
    hold all
    subplot(2,1,2)
    plot(Data(:,1),Data(:,3))
    xlabel('Epoch')
    ylabel('Accuracy %')    
    title('Accuracy on Test Set')    
    hold all    
end
subplot(2,1,1)
legend(legend_strs)
subplot(2,1,2)
legend(legend_strs)