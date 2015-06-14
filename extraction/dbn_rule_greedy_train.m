function [vld_acc,tst_acc] = dbn_rule_greedy_train(confs)
%% Greedy train DBN & extract rulesm
%sontran2015
depth = size(confs,2);
for i=1:depth-1
    if i==1, row_dat = confs(i).row_dat;
    else row_dat = 0; end
    
    if ~exist(confs(i).mod_f,'file')
        fprintf('Start training deep layer %d\n',i);
        rbm = gen_rbm_train(confs(i));
        save(confs(i).mod_f,'rbm');
    end       
    if ~exist('rbm','var'), load(confs(i).mod_f); end
    % change the dimentions back to oldversion    
    if size(rbm.hidB,1)<size(rbm.hidB,2), rbm.hidB = rbm.hidB'; rbm.visB = rbm.visB'; end
    if ~exist(confs(i).trn_out_file,'file')        
       vis2hid_file(rbm,confs(i).trn_dat_file,confs(i).trn_out_file,row_dat);
       %vis2hid_rule_file(rbm,confs(i).trn_dat_file,confs(i).trn_rule_out_file);
    end
    if ~isempty(confs(i).vld_dat_file) && ~exist(confs(i).vld_out_file,'file')
       vis2hid_file(rbm,confs(i).vld_dat_file,confs(i).vld_out_file,row_dat);
       vis2hid_rule_file(rbm,confs(i).vld_dat_file,confs(i).vld_rule_out_file,row_dat);
    end
        if ~isempty(confs(i).tst_dat_file) && ~exist(confs(i).tst_out_file,'file')
           vis2hid_file(rbm,confs(i).tst_dat_file,confs(i).tst_out_file,row_dat);
           vis2hid_rule_file(rbm,confs(i).tst_dat_file,confs(i).tst_rule_out_file,row_dat);
        end
        clear rbm;    
end

if exist(confs(depth).mod_f,'file'), vld_acc = -1; tst_acc=-1; return; end
fprintf('Start training top layer');
[rbm,vld_acc,tst_acc] = class_rbm_train_rule(confs(depth));

save(confs(depth).mod_f,'rbm');

end