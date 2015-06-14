function [c rule] = extract_rule(w_vec,spr,N)
% Extract rules from weight vectors
% sontran2013
% spr: sparsity of rule
c = 0;
o_c = -1;
www = abs(w_vec);
rule = (w_vec>0)*2-1;
if nargin==1, spr =0;end
if nargin<3, N = 0; end
%fprintf('starttting\n');
while c~=o_c && sum(rule.^2) >N
    o_c = c;
    %update c 
    %www.*rule
    %c = mean(www(find(rule~=0,1,'first'):end));
    c = sum(w_vec.*rule)/sum(rule.^2);    
    %update rule    
    rule(find(www<=(c/2 + spr/(2*c)))) = 0;    
    %fprintf('%.5f %.5f %d\n',o_c,c,sum(rule.^2));
    %pause
end

end

