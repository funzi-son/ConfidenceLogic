function [R,T] = extract_rbm_d(model)
% extract rules from RBM with label
% using discriminative nature of the model
% hidB: hidNumx1
RW = [model.W' model.hidB];
R.c = zeros(1,size(RW,1));
R.r = zeros(size(RW));

for i=1:size(RW,1)
    [R.c(i) R.r(i,:)] = extract_rule(RW(i,:));
end

[lNum,pNum] = size(model.U);

for i=1:lNum
    T(i).r = ones(pNum+1,1);
    T(i).c = exp([model.U(i,:)';model.labB(i)]);
end

end

