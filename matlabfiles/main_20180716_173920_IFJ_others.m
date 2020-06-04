

%% Comments: model I in ...
%% Predictors: $[ M, W, H, L(d-1),L(d-7), T(d) L'(d-1), L"(d-1), ... ].$ 
%% 
%% forecast horizon is one month!
%% LSBoosting method

%%
%%

datetime('now','TimeZone','local','Format','yMMdd_HHmmss ')
datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss')
%%
%%

clc ; clear ; close all;
load matFiles//MyDataSet

MyData = MyDataSet ;
MyData(1:2) = [];

for ii = 1 : length(MyData)
    
    DaysOfYear = MyData(ii).Date ;
    
    [YY,MM,~] = datevec(DaysOfYear) ;
    MM = repmat(MM,24,1) ;
    
    WW = MyData(ii).DayNumber ;
    WW = repmat(WW,24,1) ;

    [ MyHolly , ~ ] = findHolidayIdx_20170910_104757(DaysOfYear) ;
    wknd = MyData(ii).isweekend == 1 ;
    NotWorkingDay = double(or( MyHolly,double(wknd))) ;
    NotWorkingDay(NotWorkingDay==0) = -1;
    NotWorkingDay = repmat(NotWorkingDay,24,1) ;    
%     sum(MyHolly), sum(wknd), sum(NotWorkingDay)
    nDays = length(YY) ;
    HH = (1:24)' ;
    HH = repmat(HH,1,nDays) ;
    
    MyData(ii).M = MM ;
    MyData(ii).W = WW ;
    MyData(ii).MyHolly = MyHolly ;
    MyData(ii).NotWorkingDay = NotWorkingDay ;
    MyData(ii).H = HH ;
    
end

clear ii
clear nDays
clear MyDataSet
clear NotWorkingDay MyHolly
clear  YY MM DD WW HH
clear DaysOfYear
clear wknd

Load_ = [ MyData.Load ];

Load_previousDay = circshift(Load_,[0 1]);
Load_previousDay(:,1) = NaN ;

Load_previousWeek = circshift(Load_,[0 7]);
Load_previousWeek(:,1:7) = NaN ;

Temperature_ = [ MyData.Temp_Ave ];

Temperature_previousDay = circshift(Temperature_,[0 1]);
Temperature_previousWeek = circshift(Temperature_,[0 7]);


Dates_ = [ MyData.Date ];
M_ = [ MyData.M ];
W_ = [ MyData.W ];
NWD_ = [ MyData.NotWorkingDay ];
H_ = [ MyData.H ] ;

clear MyData
%% 
%% Remove NaNs

if 0%isnan(Load_previousDay(1,1))
    Load_(:,1:7)=[];
    Load_previousDay(:,1:7)=[];
    Load_previousWeek(:,1:7)=[];
    Temperature_(:,1:7)=[];
    Temperature_previousDay(:,1:7)=[];
    Temperature_previousWeek(:,1:7)=[];
    Dates_(:,1:7)=[];
    M_(:,1:7)=[];
    W_(:,1:7)=[];
    NWD_(:,1:7)=[];
    H_(:,1:7)=[];
end
%%
%% Gradient

[~ ,  gy_Load_previousDay] = gradient(Load_previousDay);
[~ ,  gyy_Load_previousDay] = gradient(gy_Load_previousDay);

[~ ,  gy_Load_previousWeek] = gradient(Load_previousWeek);
[~ ,  gyy_Load_previousWeek] = gradient(gy_Load_previousWeek);

[~ ,  gy_Temperature_] = gradient(Temperature_);
[~ ,  gyy_Temperature_] = gradient(gy_Temperature_);
%%
%% Diff temperature

Temperature_diff_1 = Temperature_ - Temperature_previousDay ;
Temperature_diff_1 = Temperature_diff_1./Temperature_; % normalized

Temperature_diff_7 = Temperature_ - Temperature_previousWeek ;
Temperature_diff_7 = Temperature_diff_7./Temperature_; % normalized
%%
%% Train, test
%% Rolling window training and testing model
%%

clc;

StartDate_2011 = datetime(2011,01,01,'Format','yyyy-MM-dd') ;
EndDate_2011 = datetime(2011,12,31,'Format','yyyy-MM-dd') ;
% Dates_2011 = StartDate_2011 : EndDate_2011 ;
dummy_test_days = datevec(StartDate_2011 : EndDate_2011);

idx_before2011 = sum(Dates_ < datetime(2011,01,01,'Format','yyyy-MM-dd'));

idx_test = NaN(1,12) ;
idx_train = NaN(1,12) ;
for month0ftest = 1 : 12
    idx_test(month0ftest) = find(dummy_test_days(:,2) == month0ftest & dummy_test_days(:,3) == 1 ) ;
    idx_train(month0ftest) = idx_before2011 + idx_test(month0ftest) -1 ;
end

% clear StartDate_2011 EndDate_2011
% clear dummy_test_days
%%
%% Now we are prepared to have a sliding window train and testing scheme
%%

Pinball_MoY = NaN(1,12);
for month0ftest = 1 : 12
    
    trn_Load_ = Load_(:,1:idx_train(month0ftest));
    
    trn_Load_previousDay = Load_previousDay(:,1:idx_train(month0ftest));
    trn_gy_Load_previousDay = gy_Load_previousDay(:,1:idx_train(month0ftest));
    trn_gyy_Load_previousDay = gyy_Load_previousDay(:,1:idx_train(month0ftest));
    
    trn_Load_previousWeek = Load_previousWeek(:,1:idx_train(month0ftest));
    trn_gy_Load_previousWeek = gy_Load_previousWeek(:,1:idx_train(month0ftest));
    trn_gyy_Load_previousWeek = gyy_Load_previousWeek(:,1:idx_train(month0ftest));
    
    trn_Temperature = Temperature_(:,1:idx_train(month0ftest));
    trn_gy_Temperature = gy_Temperature_(:,1:idx_train(month0ftest));
    trn_gyy_Temperature = gyy_Temperature_(:,1:idx_train(month0ftest));

%     trn_Temperature_diff_1 = Temperature_diff_1(:,1:idx_train(month0ftest));
%     trn_Temperature_diff_7 = Temperature_diff_7(:,1:idx_train(month0ftest));

    trn_M = M_(:,1:idx_train(month0ftest)); 
    trn_W = W_(:,1:idx_train(month0ftest)); 
%     trn_NWD = NWD_(:,1:idx_train(month0ftest)); 
    trn_H = H_(:,1:idx_train(month0ftest)); 
    
    trn_X = [ trn_M(:),  trn_W(:), trn_H(:), trn_Load_previousWeek(:), trn_Load_previousDay(:), trn_Temperature(:) ] ;
    trn_y = trn_Load_(:); 
    
    catidx = 1:3;
    t = templateTree('Surrogate','on','MaxNumSplits',2^7);
    ens = fitrensemble( trn_X,trn_y,'CategoricalPredictors',catidx,'method','LSBoost',...
    'Learners',t,'NumLearningCycles',100,'Nprint',50,'LearnRate',0.1);

%     ens0fmonth(month0ftest).ens = ens;
    clear trn_X trn_y 
    clear trn_M trn_W trn_H
    clear trn_Load_previousWeek trn_Load_previousDay
    clear trn_gy_Load_previousDay trn_gyy_Load_previousDay
    clear trn_gy_Load_previousWeek trn_gyy_Load_previousWeek
    clear trn_Temperature trn_gy_Temperature trn_gyy_Temperature
    clear trn_Temperature_diff_1 trn_Temperature_diff_7
    clear trn_Load_
    clear catidx
    clear t
    %
    %%
    % test
    idx_test2 = [idx_test, 366];
    idx_test_0nemonth = idx_before2011+(idx_test2(month0ftest):idx_test2(month0ftest+1)-1);

    Dates_(idx_test_0nemonth), pause(.2)
    disp("========================================")
    tst_Load_ = Load_(:,idx_test_0nemonth);   
    %
    tst_Load_0neMonth_hat = Load_(:, [idx_test_0nemonth(1)-7:idx_test_0nemonth(1)-1,idx_test_0nemonth] );
%   Or ==>  [idx_test_0nemonth(1)-7:idx_test_0nemonth(end)]
    tst_Temperature_0neMonth = Temperature_(:,idx_test_0nemonth) ;
    nr_replicate = 100 ;
    tst_Temperature_0neMonth_extended = Generate_Hypotheses_20180712_120131( tst_Temperature_0neMonth , nr_replicate );
    
    pinballVal_test = NaN( 24, length(idx_test_0nemonth) );
%     tree0utput_test = NaN( 24, ens.NumTrained, length(idx_test_0nemonth) );
    
    for ii = 1 : length(idx_test_0nemonth)
        
        idx_test_day = idx_test_0nemonth(ii);
        
        tst_Load_previousDay = repmat(tst_Load_0neMonth_hat(:,ii+7-1),nr_replicate,1);
        tst_gy_Load_previousDay = gradient(tst_Load_previousDay);
        tst_gyy_Load_previousDay = gradient(tst_gy_Load_previousDay);
        
        tst_Load_previousWeek = repmat(tst_Load_0neMonth_hat(:,ii+7-7),nr_replicate,1);
        tst_gy_Load_previousWeek = gradient(tst_Load_previousWeek);
        tst_gyy_Load_previousWeek = gradient(tst_gy_Load_previousWeek);
        
        tst_Temperature = tst_Temperature_0neMonth_extended(:,ii);   
    
        tst_gy_Temperature = gradient(tst_Temperature);
        tst_gyy_Temperature = gradient(tst_gy_Temperature);

%         tst_Temperature_diff_1 = tst_Temperature-repmat(Temperature_(:,idx_test_0nemonth(ii)-1),nr_replicate,1);
%         tst_Temperature_diff_1 = tst_Temperature_diff_1./repmat(Temperature_(:,idx_test_0nemonth(ii)-1),nr_replicate,1);
%         
%         tst_Temperature_diff_7 = tst_Temperature-repmat(Temperature_(:,idx_test_0nemonth(ii)-7),nr_replicate,1);
%         tst_Temperature_diff_7 = tst_Temperature_diff_7./ repmat(Temperature_(:,idx_test_0nemonth(ii)-7),nr_replicate,1);
     
        tst_M = repmat(M_(:,idx_test_0nemonth(ii)),nr_replicate,1);
        tst_W = repmat(W_(:,idx_test_0nemonth(ii)),nr_replicate,1);
        tst_H = repmat(H_(:,idx_test_0nemonth(ii)),nr_replicate,1);
         
        tst_X = [ tst_M(:), tst_W(:), tst_H(:), tst_Load_previousWeek(:), tst_Load_previousDay(:), tst_Temperature(:) ] ;
        clear tst_M tst_W tst_H
        clear tst_Load_previousWeek tst_Load_previousDay tst_gy_Load_previousDay
        clear tst_gyy_Load_previousDay tst_gy_Load_previousWeek tst_gyy_Load_previousWeek
        clear tst_Temperature tst_gy_Temperature tst_gyy_Temperature
        clear tst_Temperature_diff_1 tst_Temperature_diff_7
        
        tst_Load_hypotheses_estimated = reshape(predict(ens,tst_X),24,[]);
        tst_Load_0neMonth_hat(:,ii+7) =  mean(tst_Load_hypotheses_estimated,2) ;

% %         figure(1); clf;
% %         plot(reshape(predict(ens,tst_X),24,[]),':')
% %         hold on;
% %         plot(tst_Load_0neMonth_hat(:,ii+7), 'k')
% %         hold off;
% %         waitforbuttonpress
              
        tst_y = tst_Load_(:,ii) ;
%       
        for h = 1 : size(tst_y,1)
            pinballVal_test(h,ii) = myPinball_20180615_150051( tst_y(h) , tst_Load_hypotheses_estimated(h,:));
        end
        clear h
%             
    end
    

         Pinball_MoY(month0ftest) = mean2(pinballVal_test);
         clear tree0utput_test pinballVal_test tst_X 
         clear tst_Load_ tst_Load_hat
         clear ens
         clear treeIdx

end
%%

[1:12;Pinball_MoY]
% save Pinball_MoY_main_20180706_201526_IFJ Pinball_MoY
%%
%%
% 
%        1.0000    2.0000    3.0000    4.0000    5.0000    6.0000    7.0000    8.0000    9.0000   10.0000   11.0000   12.0000
% 2^5    3.8911    3.6660    3.3913    3.2282    4.1448    5.7392    4.2067    9.4381    4.8107    3.5797    4.2449    4.0656
% 2^6    4.0194    3.5408    3.1124    3.0530    3.7938    5.5883    4.2387    9.5208    4.8413    3.3487    3.8991    4.0770
% 2^7    3.9813    3.6025    3.1263    2.9046    3.7059    5.6948    4.3350    9.3347    5.0249    3.1948    3.7625    4.0642
% 2^8    3.9299    3.6352    3.1778    2.8000    3.9421    5.6163    4.4079    9.3278    4.9465    3.2426    3.6052    4.0693