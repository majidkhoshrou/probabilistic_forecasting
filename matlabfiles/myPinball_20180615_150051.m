

function  pinballVal  = myPinball_20180615_150051(x1_target,x1_values)

%%

% % % x1_target=tst_y(h) ;
% % % x1_values= tst_Load_hypotheses_estimated(h,:);
% row=982;
% x1_values = tree0utput_test(row,:);

q = .01:.01:.99;

x1_quantile = quantile(x1_values(:),q);
% x1_target = tst_y(row);
 
idx_lower = x1_target < x1_quantile ;
pinballVal_lower = (1-q).*(x1_quantile-x1_target).*idx_lower;

idx_higher = x1_target >= x1_quantile;
pinballVal_higher = q.*(x1_target-x1_quantile).*idx_higher;

pinballVal0 = pinballVal_lower+pinballVal_higher;
pinballVal = mean(pinballVal0);






