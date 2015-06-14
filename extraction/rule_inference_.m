function output = rule_inference_(Rs,input,no_bias)
% Inference using confidence-based rule
% sontran2013
output = input;
for i=1:size(Rs,2)   
    
    size(Rs(i).r)
    if exist('no_bias','var') && no_bias
        output = logistic(bsxfun(@times,Rs(i).r(:,1:end-1),Rs(i).c')*output);     
    else    
        output = logistic(bsxfun(@times,Rs(i).r,Rs(i).c')*[output;ones(1,size(input,2))]);
    end
end
%output  = output;
end

