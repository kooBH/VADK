% Get sample result  

close all;
clear;

%% Param
threshold = 0.9;
threshold_up = 4;
threshold_down = 30; 
%% Load

%vad_path  = '\\163.239.192.82\nas1_data\user\kbh\workspace\VADK\measure\WebRTC\99_200629-1721.bin';
%label_path  = '\\163.239.192.82\nas1_data\user\kbh\workspace\VADK\measure\WebRTC\99_200629-1721.mat';

vad_path  = '\\163.239.192.82\nas1_data\user\kbh\workspace\VADK\measure\WebRTC\25_200420-1157.bin';
label_path  = '\\163.239.192.82\nas1_data\user\kbh\workspace\VADK\measure\WebRTC\25_200420-1157.mat';

fid = fopen(vad_path);
vad = fread(fid,'single');
vad = vad>threshold;
fclose(fid);

label = load(label_path);
label = label.label;

%% Post Process
cnt = 0;
flag = 0;
for idx  = 1:length(vad)
    % curretnly active
    if vad(idx)
        % been active
        if flag == 1
            cnt = 0;
        else
         % been deactive
            cnt = cnt + 1;
            if cnt > threshold_up
                flag = 1;
                cnt = 0;
            end
        end
     % currently deactive
    else
        % been active
        if flag == 1
            cnt = cnt +1;
            if cnt > threshold_down
                flag = 0;
                cnt = 0;
            end
        % been deactive
        else
            cnt = 0;
        end
    end
    vad(idx)=flag;
end
    
%% Plot

 figure;
hold on;

x_range =  1:length(vad);

plot(x_range,vad*0.8,'-','LineWidth',2,'color','#ff3b37');
plot(x_range,label*0.6,'-','LineWidth',2,'color','#3135ff');
legend('WebRTC-RNN-VAD','label')

title(['threshold : ' num2str(threshold) ', up : ' num2str(threshold_up) ', down :' num2str(threshold_down) ]);
ylim([0 1]);
hold off