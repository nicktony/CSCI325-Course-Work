% % % % % % % % % % % % % % % % % % %
% Nickolaus White (CSCI325)
% % % % % % % % % % % % % % % % % % %


% Load in Data of 25,000 Heights, Weights, and BMI's
%---------------------------------------------------------------
data = readtable('heightweight_inputs.xlsx');
heightweightinputs = table2array(data);

clusters = clusterdata(heightweightinputs,'maxclust',16,'distance','correlation','linkage','average');


% Plot Figure 1, Hierarchical Cluster Tree
%---------------------------------------------------------------
figure(1)
for i = 1:16
    subplot(4,4,i);
    plot(heightweightinputs((clusters == i),:)');
    axis tight
end
subtitle('Hierarchical Cluster Tree');


% Plot Figures 2 & 3, PCA & K-Means (Lower Dimensions)
%---------------------------------------------------------------
figure(2)
[~,score,~,~,explainedVar] = pca(heightweightinputs);
bar(explainedVar)
title('Explained Variance: 100% explained by first two principal components')
ylabel('PC')

% Retain first two principal components
bmiPC = score(:,1:2);

figure(3)
[clusters, centroid] = kmeans(bmiPC,6);
gscatter(bmiPC(:,1),bmiPC(:,2),clusters)
legend('location','southeast')
xlabel('First Principal Component');
ylabel('Second Principal Component');
title('Principal Component Scatter Plot with Colored Clusters');


% Plot Figure 4, PCA & Self-Organizing Maps (Lower Dimensions)
%---------------------------------------------------------------
net = newsom(bmiPC',[4 4]);
net = train(net,bmiPC');

distances = dist(bmiPC,net.IW{1}');
[d,center] = min(distances,[],2);
% center gives the cluster index

figure(4)
gscatter(bmiPC(:,1),bmiPC(:,2),center); legend off;
hold on
plotsom(net.iw{1,1},net.layers{1}.distances);
hold off


% Save file contents, end of program
%---------------------------------------------------------------
filename = 'A4_CSCI325_NickolausWhite.mat';
save(filename)




