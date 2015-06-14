function output = rule_inference_d(R,T,input,no_bias)
% note that the confidence is computed as: 
         % step 1: normalize the input confidence to 0 - inf: IC = exp(input)
         % step 2: applying normal inference rules: IC
         % step 3: multiply with confidence value of rules 
    %input  = input';
    if no_bias
        input = bsxfun(@times,R.r(:,1:end-1),R.c')*input;     
    else           
        %size([input;ones(1,size(input,2))])       
        %size(R.r)
        %size(R.c)
        %size(bsxfun(@times,R.r,R.c'))        
        %pause
        input = bsxfun(@times,R.r,R.c')*[input;ones(1,size(input,2))];        
    end
    
%    input = bsxfun(@plus,input'*model.W,model.hidB)';
    %do normalization
    input = exp(input); % normalize to 0-inf
    
    output = zeros(size(input,2),size(T,2));
    for i=1:size(output,2)
        cv = sum(log(1 + bsxfun(@times,input,T(i).c(1:end-1))));
        if ~no_bias, cv = cv + log(T(i).c(end)+1); end 
        output(:,i) = cv';
    end
    %output(1:15,:)
    [~,output] = max(output,[],2);
    output = output';
end