% Get sample result  

close all;
clear;

threshold = 0.5;

vad_path  = '\\163.239.192.82\nas1_data\user\kbh\workspace\VADK\measure\WebRTC\99_200629-1721.bin';
label_path  = '\\163.239.192.82\nas1_data\user\kbh\workspace\VADK\measure\WebRTC\99_200629-1721.mat';

fid = fopen(vad_path);
vad = fread(fid,'single');
vad = vad>threshold;
fclose(fid);

label = load(label_path);
label = label.label;

 figure;
hold on;

x_range =  1:length(vad);

plot(x_range,vad*0.8,'x','LineWidth',2,'color','#ff3b37');
plot(x_range,label*0.6,'x','LineWidth',2,'color','#3135ff');
legend('WebRTC-RNN-VAD','label')

title(['threshold : ' num2str(threshold)]);
ylim([0 1]);
hold off