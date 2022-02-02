clear;
train = importdata('test_omnidepth.txt');
file = fopen('test_360d_matterport.txt','w');
for i = 1:length(train)
    trainfile = train{i};
    if contains(trainfile,'Matterport') == 1
        fprintf(file, '%s\n', trainfile);
    end
end
fclose(file);
