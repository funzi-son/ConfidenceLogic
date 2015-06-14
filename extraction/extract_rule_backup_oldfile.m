function [c rule] = extract_rule(w_vec,spr)
% Extract rules from weight vectors
% sontran2013
% spr: sparsity of rule
c = 0;
o_c = -1;
www = abs(w_vec);
rule = (w_vec>0)*2-1;
if nargin==1, spr =0; end
while c~=o_c
    o_c = c;
    %update c    
    c = mean(www(find(rule~=0,1,'first'):end));
    fprintf('%.5f %.5f\n',o_c,c);
    pause
    %update rule
    rule(find(www<=(c/2 + spr/(2*c)))) = 0;    
end

end

