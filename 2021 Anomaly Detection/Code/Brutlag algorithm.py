# brutlag_algorithm (modified)
def brutlag_algorithm(df_diff, GAMMA):    
    
    PERIOD = 7                                        # The given time series has seasonal_period = 7
    M = 3                                            # brutlag scaling factor for the confidence bands
    dt = []                                           # dt = GAMMA(y-y^) + (1-GAMMA)d(t-PERIOD)
    LB_2 = []                                           # y^min(t) = y^ - M*d(t-PERIOD) , brutlag_algorithm
    UB_2 = []                                           # y^max(t) = y^ + M*d(t-PERIOD) , brutlag_algorithm
    
    LB = []                                           # final LB : avg of LB_1 and LB_2
    UB = []                                           # final UB : avg of UB_1 and UB_2
    
    outlier = []                                      # IF actual < LB or actual > UP
    
    # prediction table 
    df_diff['diff'] = df_diff['actual'] - df_diff['predicted']

    ## dt, UB_2, LB_2, UB, LB, outlier
    for i in range(len(df_diff)):
        if i < PERIOD:
            dt.append(GAMMA*abs(df_diff['diff'][i]))
            LB_2.append(df_diff['predicted'][i])
            UB_2.append(df_diff['predicted'][i])
            
            LB.append(df_diff['predicted'][i])
            UB.append(df_diff['predicted'][i])
            outlier.append(bool(False))        
        else:
            dt.append(GAMMA*abs(df_diff['diff'][i]) + (1-GAMMA)*dt[i-PERIOD])
            LB_2.append(df_diff['predicted'][i] - M*dt[i-PERIOD])
            UB_2.append(df_diff['predicted'][i] + M*dt[i-PERIOD])
            
            LB.append((df_diff['LB_1'][i] + (df_diff['predicted'][i] - M*dt[i-PERIOD]))/2)
            UB.append((df_diff['UB_1'][i] + (df_diff['predicted'][i] + M*dt[i-PERIOD]))/2)
        
            # LB.append(min(df_diff['LB_1'][i], (df_diff['predicted'][i] - M*dt[i-PERIOD])))
            # UB.append(max(df_diff['UB_1'][i], (df_diff['predicted'][i] + M*dt[i-PERIOD])))
        
            outlier.append((df_diff['actual'][i] < LB[i]) | (df_diff['actual'][i] > UB[i]))

    df_diff['LB_2'] = LB_2
    df_diff['UB_2'] = UB_2
                      
    df_diff['LB'] = LB
    df_diff['UB'] = UB
                      
    df_diff['outlier_tf'] = outlier
  
    return df_diff   