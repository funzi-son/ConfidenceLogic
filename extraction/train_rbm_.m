function model = train_rbm_(conf,dat)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training RBM                                                       %  
% conf: training setting                                             %
% -*-sontran2012-*-                                                  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

assert(~isempty(dat),'[KBRBM] Data is empty'); 
%% initialization
visNum  = size(dat,2);
hidNum  = conf.hidNum;
sNum  = conf.sNum;
lr    = conf.params(1);                                                         % Number of epoch training with lr_1                     


model.W = 0.01*randn(visNum,hidNum);
DW    = zeros(size(model.W));
model.visB  = zeros(1,visNum);
DVB   = zeros(1,visNum);
model.hidB  = zeros(1,hidNum);
DHB   = zeros(1,hidNum);


%% Reconstruction error 
mse    = 0;
%% ==================== Start training =========================== %%
for i=1:conf.eNum
    inx = randperm(size(dat,1));        
    omse = mse;
    mse = 0;
    for j=1:conf.bNum
        iiii = inx((j-1)*sNum+1:j*sNum);
        visP = dat(iiii,:);               
       %up
       hidP = logistic(visP*model.W + repmat(model.hidB,sNum,1));
       hidPs =  1*(hidP >rand(sNum,hidNum));
       hidNs = hidPs;
       for k=1:conf.gNum
           % down
           visN  = logistic(hidNs*model.W' + repmat(model.visB,sNum,1));
           visNs = 1*(visN>rand(sNum,visNum));
%            if j==5 && k==1, save_images(visN,'',sNum,i,28,28); end
           % up
           hidN  = logistic(visNs*model.W + repmat(model.hidB,sNum,1));
           hidNs = 1*(hidN>rand(sNum,hidNum));
       end
       % Compute MSE for reconstruction       
       mse = mse + sum(sum((visP-visN).^2,1)/sNum,2);
       % Update W,visB,hidB
       diff = (visP'*hidP - visNs'*hidN)/sNum;
       DW  = lr*(diff - conf.params(4)*model.W) +  conf.params(3)*DW;
       model.W   = model.W + DW;
       DVB  = lr*sum(visP - visN,1)/sNum + conf.params(3)*DVB;
       model.visB = model.visB + DVB;
       DHB  = lr*sum(hidP - hidN,1)/sNum + conf.params(3)*DHB;
       model.hidB = model.hidB + DHB;
       
       % sparsity constraints
       if conf.lambda >0
           hidI = (visP*model.W +  repmat(model.hidB,sNum,1));
           hidN = logistic(hidI);
           pppp = (conf.p - sum(hidN,1)/sNum);
           %model.W    = model.W   + lr*conf.lambda*(repmat(pppp,visNum,1).*(visP'*((hidN.^2).*exp(-hidI))/sNum));
           model.hidB = model.hidB + lr*conf.lambda*(pppp.*(sum((hidN.^2).*exp(-hidI),1)/sNum));                      
       end
        
       if sum(sum(isnan(model.W)))>0 sum(sum(isinf(model.W)))>0
           % if learning something wrong
           return;
       end
    end
   %%               
    fprintf('Epoch %d  : MSE = %f\n',i,mse);   
end

end