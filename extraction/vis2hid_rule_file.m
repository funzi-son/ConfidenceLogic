function vis2hid_rule_file(model,dat_file,new_file,row_dat)
    if nargin<4, row_dat = 0; end
    % Extraction
    R = extract_rbm(model,[],0);
    % Get confidences
    %dat = logistic(bsxfun(@plus,get_data_from_file(dat_file)*model.W,model.hidB));
    dat  = rule_inference_(R,get_data_from_file(dat_file,row_dat),0);
    save(new_file,'dat');
end

