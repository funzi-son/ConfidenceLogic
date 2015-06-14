function [R,T] = extract_rbm_l(model,vis_dir,N,cpen,spr,proN)
% Extract rules from rbm consisting of labels
% T: top rules for labels
% R: lower rules for intermediate proposition
if nargin<5, spr = 0; end
if nargin<6, proN = 0; end
TW  = [model.U model.labB];
T.c = zeros(1,size(TW,1));
T.r = zeros(size(TW));
for i=1:size(TW,1)
    [T.c(i) T.r(i,:)] = extract_rule(TW(i,:));
end

RW = [model.W' model.hidB];
R.c = zeros(1,size(RW,1));
R.r = zeros(size(RW));

for i=1:size(RW,1)
    [R.c(i) R.r(i,:)] = extract_rule(RW(i,:),spr,proN);   
%      h = figure;    
%      subplot(1,2,1); imshow(reshape(((RW(i,1:end-1)>0.5) + -1*(RW(i,1:end-1)<-0.5)+1)/2,[28,28]));
%      subplot(1,2,2); imshow(reshape((R.r(i,1:end-1)+1)/2,[28,28]));
%      waitforbuttonpress
end

if exist('N','var') && N>0
    T = impruning(T,R,TW,N);
    %T = impruning_1(T,R);
end
if exist('cpen','var') && cpen>0
    T.c = T.c*cpen;%mean(abs(TW),2)';
    R.c = R.c*cpen;%mean(abs(RW),2)';
end
if exist(vis_dir,'var') && ~isempty(vis_dir)
    T.r = T.r(:,1:end-1);
    R.r = R.r(:,1:end-1);
    save_img_rules(T,R,vis_dir);
end
end