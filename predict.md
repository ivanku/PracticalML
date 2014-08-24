## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset). 

## Summary

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. 

Exploratory analysis of the data indicated that there is a number of variables with missing values and a number of variable with low variance. all of these variables were excluded from predictive model.

Data set was split into training and cross-validation parts as 60-40%.

Two models were trained - classification tree and random forest. Based on comparison of out of sample error estimates random forest model was selected as providing better results. That model was used for the second part of the assignment to predict 20 different test cases.

## Analysis details


```r
# Load libraries
library(caret)
library(randomForest)
```

### Read the data

```r
d <- read.csv('pml-training.csv')
testing <- read.csv('pml-testing.csv')
```

### exploratory analysis and data cleaning

```r
colnames(d)
```

```
##   [1] "X"                        "user_name"               
##   [3] "raw_timestamp_part_1"     "raw_timestamp_part_2"    
##   [5] "cvtd_timestamp"           "new_window"              
##   [7] "num_window"               "roll_belt"               
##   [9] "pitch_belt"               "yaw_belt"                
##  [11] "total_accel_belt"         "kurtosis_roll_belt"      
##  [13] "kurtosis_picth_belt"      "kurtosis_yaw_belt"       
##  [15] "skewness_roll_belt"       "skewness_roll_belt.1"    
##  [17] "skewness_yaw_belt"        "max_roll_belt"           
##  [19] "max_picth_belt"           "max_yaw_belt"            
##  [21] "min_roll_belt"            "min_pitch_belt"          
##  [23] "min_yaw_belt"             "amplitude_roll_belt"     
##  [25] "amplitude_pitch_belt"     "amplitude_yaw_belt"      
##  [27] "var_total_accel_belt"     "avg_roll_belt"           
##  [29] "stddev_roll_belt"         "var_roll_belt"           
##  [31] "avg_pitch_belt"           "stddev_pitch_belt"       
##  [33] "var_pitch_belt"           "avg_yaw_belt"            
##  [35] "stddev_yaw_belt"          "var_yaw_belt"            
##  [37] "gyros_belt_x"             "gyros_belt_y"            
##  [39] "gyros_belt_z"             "accel_belt_x"            
##  [41] "accel_belt_y"             "accel_belt_z"            
##  [43] "magnet_belt_x"            "magnet_belt_y"           
##  [45] "magnet_belt_z"            "roll_arm"                
##  [47] "pitch_arm"                "yaw_arm"                 
##  [49] "total_accel_arm"          "var_accel_arm"           
##  [51] "avg_roll_arm"             "stddev_roll_arm"         
##  [53] "var_roll_arm"             "avg_pitch_arm"           
##  [55] "stddev_pitch_arm"         "var_pitch_arm"           
##  [57] "avg_yaw_arm"              "stddev_yaw_arm"          
##  [59] "var_yaw_arm"              "gyros_arm_x"             
##  [61] "gyros_arm_y"              "gyros_arm_z"             
##  [63] "accel_arm_x"              "accel_arm_y"             
##  [65] "accel_arm_z"              "magnet_arm_x"            
##  [67] "magnet_arm_y"             "magnet_arm_z"            
##  [69] "kurtosis_roll_arm"        "kurtosis_picth_arm"      
##  [71] "kurtosis_yaw_arm"         "skewness_roll_arm"       
##  [73] "skewness_pitch_arm"       "skewness_yaw_arm"        
##  [75] "max_roll_arm"             "max_picth_arm"           
##  [77] "max_yaw_arm"              "min_roll_arm"            
##  [79] "min_pitch_arm"            "min_yaw_arm"             
##  [81] "amplitude_roll_arm"       "amplitude_pitch_arm"     
##  [83] "amplitude_yaw_arm"        "roll_dumbbell"           
##  [85] "pitch_dumbbell"           "yaw_dumbbell"            
##  [87] "kurtosis_roll_dumbbell"   "kurtosis_picth_dumbbell" 
##  [89] "kurtosis_yaw_dumbbell"    "skewness_roll_dumbbell"  
##  [91] "skewness_pitch_dumbbell"  "skewness_yaw_dumbbell"   
##  [93] "max_roll_dumbbell"        "max_picth_dumbbell"      
##  [95] "max_yaw_dumbbell"         "min_roll_dumbbell"       
##  [97] "min_pitch_dumbbell"       "min_yaw_dumbbell"        
##  [99] "amplitude_roll_dumbbell"  "amplitude_pitch_dumbbell"
## [101] "amplitude_yaw_dumbbell"   "total_accel_dumbbell"    
## [103] "var_accel_dumbbell"       "avg_roll_dumbbell"       
## [105] "stddev_roll_dumbbell"     "var_roll_dumbbell"       
## [107] "avg_pitch_dumbbell"       "stddev_pitch_dumbbell"   
## [109] "var_pitch_dumbbell"       "avg_yaw_dumbbell"        
## [111] "stddev_yaw_dumbbell"      "var_yaw_dumbbell"        
## [113] "gyros_dumbbell_x"         "gyros_dumbbell_y"        
## [115] "gyros_dumbbell_z"         "accel_dumbbell_x"        
## [117] "accel_dumbbell_y"         "accel_dumbbell_z"        
## [119] "magnet_dumbbell_x"        "magnet_dumbbell_y"       
## [121] "magnet_dumbbell_z"        "roll_forearm"            
## [123] "pitch_forearm"            "yaw_forearm"             
## [125] "kurtosis_roll_forearm"    "kurtosis_picth_forearm"  
## [127] "kurtosis_yaw_forearm"     "skewness_roll_forearm"   
## [129] "skewness_pitch_forearm"   "skewness_yaw_forearm"    
## [131] "max_roll_forearm"         "max_picth_forearm"       
## [133] "max_yaw_forearm"          "min_roll_forearm"        
## [135] "min_pitch_forearm"        "min_yaw_forearm"         
## [137] "amplitude_roll_forearm"   "amplitude_pitch_forearm" 
## [139] "amplitude_yaw_forearm"    "total_accel_forearm"     
## [141] "var_accel_forearm"        "avg_roll_forearm"        
## [143] "stddev_roll_forearm"      "var_roll_forearm"        
## [145] "avg_pitch_forearm"        "stddev_pitch_forearm"    
## [147] "var_pitch_forearm"        "avg_yaw_forearm"         
## [149] "stddev_yaw_forearm"       "var_yaw_forearm"         
## [151] "gyros_forearm_x"          "gyros_forearm_y"         
## [153] "gyros_forearm_z"          "accel_forearm_x"         
## [155] "accel_forearm_y"          "accel_forearm_z"         
## [157] "magnet_forearm_x"         "magnet_forearm_y"        
## [159] "magnet_forearm_z"         "classe"
```

```r
dim(d)
```

```
## [1] 19622   160
```

```r
head(d)
```

```
##   X user_name raw_timestamp_part_1 raw_timestamp_part_2   cvtd_timestamp
## 1 1  carlitos           1323084231               788290 05/12/2011 11:23
## 2 2  carlitos           1323084231               808298 05/12/2011 11:23
## 3 3  carlitos           1323084231               820366 05/12/2011 11:23
## 4 4  carlitos           1323084232               120339 05/12/2011 11:23
## 5 5  carlitos           1323084232               196328 05/12/2011 11:23
## 6 6  carlitos           1323084232               304277 05/12/2011 11:23
##   new_window num_window roll_belt pitch_belt yaw_belt total_accel_belt
## 1         no         11      1.41       8.07    -94.4                3
## 2         no         11      1.41       8.07    -94.4                3
## 3         no         11      1.42       8.07    -94.4                3
## 4         no         12      1.48       8.05    -94.4                3
## 5         no         12      1.48       8.07    -94.4                3
## 6         no         12      1.45       8.06    -94.4                3
##   kurtosis_roll_belt kurtosis_picth_belt kurtosis_yaw_belt
## 1                                                         
## 2                                                         
## 3                                                         
## 4                                                         
## 5                                                         
## 6                                                         
##   skewness_roll_belt skewness_roll_belt.1 skewness_yaw_belt max_roll_belt
## 1                                                                      NA
## 2                                                                      NA
## 3                                                                      NA
## 4                                                                      NA
## 5                                                                      NA
## 6                                                                      NA
##   max_picth_belt max_yaw_belt min_roll_belt min_pitch_belt min_yaw_belt
## 1             NA                         NA             NA             
## 2             NA                         NA             NA             
## 3             NA                         NA             NA             
## 4             NA                         NA             NA             
## 5             NA                         NA             NA             
## 6             NA                         NA             NA             
##   amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
## 1                  NA                   NA                   
## 2                  NA                   NA                   
## 3                  NA                   NA                   
## 4                  NA                   NA                   
## 5                  NA                   NA                   
## 6                  NA                   NA                   
##   var_total_accel_belt avg_roll_belt stddev_roll_belt var_roll_belt
## 1                   NA            NA               NA            NA
## 2                   NA            NA               NA            NA
## 3                   NA            NA               NA            NA
## 4                   NA            NA               NA            NA
## 5                   NA            NA               NA            NA
## 6                   NA            NA               NA            NA
##   avg_pitch_belt stddev_pitch_belt var_pitch_belt avg_yaw_belt
## 1             NA                NA             NA           NA
## 2             NA                NA             NA           NA
## 3             NA                NA             NA           NA
## 4             NA                NA             NA           NA
## 5             NA                NA             NA           NA
## 6             NA                NA             NA           NA
##   stddev_yaw_belt var_yaw_belt gyros_belt_x gyros_belt_y gyros_belt_z
## 1              NA           NA         0.00         0.00        -0.02
## 2              NA           NA         0.02         0.00        -0.02
## 3              NA           NA         0.00         0.00        -0.02
## 4              NA           NA         0.02         0.00        -0.03
## 5              NA           NA         0.02         0.02        -0.02
## 6              NA           NA         0.02         0.00        -0.02
##   accel_belt_x accel_belt_y accel_belt_z magnet_belt_x magnet_belt_y
## 1          -21            4           22            -3           599
## 2          -22            4           22            -7           608
## 3          -20            5           23            -2           600
## 4          -22            3           21            -6           604
## 5          -21            2           24            -6           600
## 6          -21            4           21             0           603
##   magnet_belt_z roll_arm pitch_arm yaw_arm total_accel_arm var_accel_arm
## 1          -313     -128      22.5    -161              34            NA
## 2          -311     -128      22.5    -161              34            NA
## 3          -305     -128      22.5    -161              34            NA
## 4          -310     -128      22.1    -161              34            NA
## 5          -302     -128      22.1    -161              34            NA
## 6          -312     -128      22.0    -161              34            NA
##   avg_roll_arm stddev_roll_arm var_roll_arm avg_pitch_arm stddev_pitch_arm
## 1           NA              NA           NA            NA               NA
## 2           NA              NA           NA            NA               NA
## 3           NA              NA           NA            NA               NA
## 4           NA              NA           NA            NA               NA
## 5           NA              NA           NA            NA               NA
## 6           NA              NA           NA            NA               NA
##   var_pitch_arm avg_yaw_arm stddev_yaw_arm var_yaw_arm gyros_arm_x
## 1            NA          NA             NA          NA        0.00
## 2            NA          NA             NA          NA        0.02
## 3            NA          NA             NA          NA        0.02
## 4            NA          NA             NA          NA        0.02
## 5            NA          NA             NA          NA        0.00
## 6            NA          NA             NA          NA        0.02
##   gyros_arm_y gyros_arm_z accel_arm_x accel_arm_y accel_arm_z magnet_arm_x
## 1        0.00       -0.02        -288         109        -123         -368
## 2       -0.02       -0.02        -290         110        -125         -369
## 3       -0.02       -0.02        -289         110        -126         -368
## 4       -0.03        0.02        -289         111        -123         -372
## 5       -0.03        0.00        -289         111        -123         -374
## 6       -0.03        0.00        -289         111        -122         -369
##   magnet_arm_y magnet_arm_z kurtosis_roll_arm kurtosis_picth_arm
## 1          337          516                                     
## 2          337          513                                     
## 3          344          513                                     
## 4          344          512                                     
## 5          337          506                                     
## 6          342          513                                     
##   kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
## 1                                                                       
## 2                                                                       
## 3                                                                       
## 4                                                                       
## 5                                                                       
## 6                                                                       
##   max_roll_arm max_picth_arm max_yaw_arm min_roll_arm min_pitch_arm
## 1           NA            NA          NA           NA            NA
## 2           NA            NA          NA           NA            NA
## 3           NA            NA          NA           NA            NA
## 4           NA            NA          NA           NA            NA
## 5           NA            NA          NA           NA            NA
## 6           NA            NA          NA           NA            NA
##   min_yaw_arm amplitude_roll_arm amplitude_pitch_arm amplitude_yaw_arm
## 1          NA                 NA                  NA                NA
## 2          NA                 NA                  NA                NA
## 3          NA                 NA                  NA                NA
## 4          NA                 NA                  NA                NA
## 5          NA                 NA                  NA                NA
## 6          NA                 NA                  NA                NA
##   roll_dumbbell pitch_dumbbell yaw_dumbbell kurtosis_roll_dumbbell
## 1         13.05         -70.49       -84.87                       
## 2         13.13         -70.64       -84.71                       
## 3         12.85         -70.28       -85.14                       
## 4         13.43         -70.39       -84.87                       
## 5         13.38         -70.43       -84.85                       
## 6         13.38         -70.82       -84.47                       
##   kurtosis_picth_dumbbell kurtosis_yaw_dumbbell skewness_roll_dumbbell
## 1                                                                     
## 2                                                                     
## 3                                                                     
## 4                                                                     
## 5                                                                     
## 6                                                                     
##   skewness_pitch_dumbbell skewness_yaw_dumbbell max_roll_dumbbell
## 1                                                              NA
## 2                                                              NA
## 3                                                              NA
## 4                                                              NA
## 5                                                              NA
## 6                                                              NA
##   max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell min_pitch_dumbbell
## 1                 NA                                 NA                 NA
## 2                 NA                                 NA                 NA
## 3                 NA                                 NA                 NA
## 4                 NA                                 NA                 NA
## 5                 NA                                 NA                 NA
## 6                 NA                                 NA                 NA
##   min_yaw_dumbbell amplitude_roll_dumbbell amplitude_pitch_dumbbell
## 1                                       NA                       NA
## 2                                       NA                       NA
## 3                                       NA                       NA
## 4                                       NA                       NA
## 5                                       NA                       NA
## 6                                       NA                       NA
##   amplitude_yaw_dumbbell total_accel_dumbbell var_accel_dumbbell
## 1                                          37                 NA
## 2                                          37                 NA
## 3                                          37                 NA
## 4                                          37                 NA
## 5                                          37                 NA
## 6                                          37                 NA
##   avg_roll_dumbbell stddev_roll_dumbbell var_roll_dumbbell
## 1                NA                   NA                NA
## 2                NA                   NA                NA
## 3                NA                   NA                NA
## 4                NA                   NA                NA
## 5                NA                   NA                NA
## 6                NA                   NA                NA
##   avg_pitch_dumbbell stddev_pitch_dumbbell var_pitch_dumbbell
## 1                 NA                    NA                 NA
## 2                 NA                    NA                 NA
## 3                 NA                    NA                 NA
## 4                 NA                    NA                 NA
## 5                 NA                    NA                 NA
## 6                 NA                    NA                 NA
##   avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell gyros_dumbbell_x
## 1               NA                  NA               NA                0
## 2               NA                  NA               NA                0
## 3               NA                  NA               NA                0
## 4               NA                  NA               NA                0
## 5               NA                  NA               NA                0
## 6               NA                  NA               NA                0
##   gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x accel_dumbbell_y
## 1            -0.02             0.00             -234               47
## 2            -0.02             0.00             -233               47
## 3            -0.02             0.00             -232               46
## 4            -0.02            -0.02             -232               48
## 5            -0.02             0.00             -233               48
## 6            -0.02             0.00             -234               48
##   accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y magnet_dumbbell_z
## 1             -271              -559               293               -65
## 2             -269              -555               296               -64
## 3             -270              -561               298               -63
## 4             -269              -552               303               -60
## 5             -270              -554               292               -68
## 6             -269              -558               294               -66
##   roll_forearm pitch_forearm yaw_forearm kurtosis_roll_forearm
## 1         28.4         -63.9        -153                      
## 2         28.3         -63.9        -153                      
## 3         28.3         -63.9        -152                      
## 4         28.1         -63.9        -152                      
## 5         28.0         -63.9        -152                      
## 6         27.9         -63.9        -152                      
##   kurtosis_picth_forearm kurtosis_yaw_forearm skewness_roll_forearm
## 1                                                                  
## 2                                                                  
## 3                                                                  
## 4                                                                  
## 5                                                                  
## 6                                                                  
##   skewness_pitch_forearm skewness_yaw_forearm max_roll_forearm
## 1                                                           NA
## 2                                                           NA
## 3                                                           NA
## 4                                                           NA
## 5                                                           NA
## 6                                                           NA
##   max_picth_forearm max_yaw_forearm min_roll_forearm min_pitch_forearm
## 1                NA                               NA                NA
## 2                NA                               NA                NA
## 3                NA                               NA                NA
## 4                NA                               NA                NA
## 5                NA                               NA                NA
## 6                NA                               NA                NA
##   min_yaw_forearm amplitude_roll_forearm amplitude_pitch_forearm
## 1                                     NA                      NA
## 2                                     NA                      NA
## 3                                     NA                      NA
## 4                                     NA                      NA
## 5                                     NA                      NA
## 6                                     NA                      NA
##   amplitude_yaw_forearm total_accel_forearm var_accel_forearm
## 1                                        36                NA
## 2                                        36                NA
## 3                                        36                NA
## 4                                        36                NA
## 5                                        36                NA
## 6                                        36                NA
##   avg_roll_forearm stddev_roll_forearm var_roll_forearm avg_pitch_forearm
## 1               NA                  NA               NA                NA
## 2               NA                  NA               NA                NA
## 3               NA                  NA               NA                NA
## 4               NA                  NA               NA                NA
## 5               NA                  NA               NA                NA
## 6               NA                  NA               NA                NA
##   stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
## 1                   NA                NA              NA
## 2                   NA                NA              NA
## 3                   NA                NA              NA
## 4                   NA                NA              NA
## 5                   NA                NA              NA
## 6                   NA                NA              NA
##   stddev_yaw_forearm var_yaw_forearm gyros_forearm_x gyros_forearm_y
## 1                 NA              NA            0.03            0.00
## 2                 NA              NA            0.02            0.00
## 3                 NA              NA            0.03           -0.02
## 4                 NA              NA            0.02           -0.02
## 5                 NA              NA            0.02            0.00
## 6                 NA              NA            0.02           -0.02
##   gyros_forearm_z accel_forearm_x accel_forearm_y accel_forearm_z
## 1           -0.02             192             203            -215
## 2           -0.02             192             203            -216
## 3            0.00             196             204            -213
## 4            0.00             189             206            -214
## 5           -0.02             189             206            -214
## 6           -0.03             193             203            -215
##   magnet_forearm_x magnet_forearm_y magnet_forearm_z classe
## 1              -17              654              476      A
## 2              -18              661              473      A
## 3              -18              658              469      A
## 4              -16              658              469      A
## 5              -17              655              473      A
## 6               -9              660              478      A
```

```r
sum(complete.cases(d))
```

```
## [1] 406
```

We see that there's only a small number of rows that have all variables available.
Let's see what % of NAs columns have.


```r
apply(d, 2, function(col)sum(is.na(col))/length(col))
```

```
##                        X                user_name     raw_timestamp_part_1 
##                   0.0000                   0.0000                   0.0000 
##     raw_timestamp_part_2           cvtd_timestamp               new_window 
##                   0.0000                   0.0000                   0.0000 
##               num_window                roll_belt               pitch_belt 
##                   0.0000                   0.0000                   0.0000 
##                 yaw_belt         total_accel_belt       kurtosis_roll_belt 
##                   0.0000                   0.0000                   0.0000 
##      kurtosis_picth_belt        kurtosis_yaw_belt       skewness_roll_belt 
##                   0.0000                   0.0000                   0.0000 
##     skewness_roll_belt.1        skewness_yaw_belt            max_roll_belt 
##                   0.0000                   0.0000                   0.9793 
##           max_picth_belt             max_yaw_belt            min_roll_belt 
##                   0.9793                   0.0000                   0.9793 
##           min_pitch_belt             min_yaw_belt      amplitude_roll_belt 
##                   0.9793                   0.0000                   0.9793 
##     amplitude_pitch_belt       amplitude_yaw_belt     var_total_accel_belt 
##                   0.9793                   0.0000                   0.9793 
##            avg_roll_belt         stddev_roll_belt            var_roll_belt 
##                   0.9793                   0.9793                   0.9793 
##           avg_pitch_belt        stddev_pitch_belt           var_pitch_belt 
##                   0.9793                   0.9793                   0.9793 
##             avg_yaw_belt          stddev_yaw_belt             var_yaw_belt 
##                   0.9793                   0.9793                   0.9793 
##             gyros_belt_x             gyros_belt_y             gyros_belt_z 
##                   0.0000                   0.0000                   0.0000 
##             accel_belt_x             accel_belt_y             accel_belt_z 
##                   0.0000                   0.0000                   0.0000 
##            magnet_belt_x            magnet_belt_y            magnet_belt_z 
##                   0.0000                   0.0000                   0.0000 
##                 roll_arm                pitch_arm                  yaw_arm 
##                   0.0000                   0.0000                   0.0000 
##          total_accel_arm            var_accel_arm             avg_roll_arm 
##                   0.0000                   0.9793                   0.9793 
##          stddev_roll_arm             var_roll_arm            avg_pitch_arm 
##                   0.9793                   0.9793                   0.9793 
##         stddev_pitch_arm            var_pitch_arm              avg_yaw_arm 
##                   0.9793                   0.9793                   0.9793 
##           stddev_yaw_arm              var_yaw_arm              gyros_arm_x 
##                   0.9793                   0.9793                   0.0000 
##              gyros_arm_y              gyros_arm_z              accel_arm_x 
##                   0.0000                   0.0000                   0.0000 
##              accel_arm_y              accel_arm_z             magnet_arm_x 
##                   0.0000                   0.0000                   0.0000 
##             magnet_arm_y             magnet_arm_z        kurtosis_roll_arm 
##                   0.0000                   0.0000                   0.0000 
##       kurtosis_picth_arm         kurtosis_yaw_arm        skewness_roll_arm 
##                   0.0000                   0.0000                   0.0000 
##       skewness_pitch_arm         skewness_yaw_arm             max_roll_arm 
##                   0.0000                   0.0000                   0.9793 
##            max_picth_arm              max_yaw_arm             min_roll_arm 
##                   0.9793                   0.9793                   0.9793 
##            min_pitch_arm              min_yaw_arm       amplitude_roll_arm 
##                   0.9793                   0.9793                   0.9793 
##      amplitude_pitch_arm        amplitude_yaw_arm            roll_dumbbell 
##                   0.9793                   0.9793                   0.0000 
##           pitch_dumbbell             yaw_dumbbell   kurtosis_roll_dumbbell 
##                   0.0000                   0.0000                   0.0000 
##  kurtosis_picth_dumbbell    kurtosis_yaw_dumbbell   skewness_roll_dumbbell 
##                   0.0000                   0.0000                   0.0000 
##  skewness_pitch_dumbbell    skewness_yaw_dumbbell        max_roll_dumbbell 
##                   0.0000                   0.0000                   0.9793 
##       max_picth_dumbbell         max_yaw_dumbbell        min_roll_dumbbell 
##                   0.9793                   0.0000                   0.9793 
##       min_pitch_dumbbell         min_yaw_dumbbell  amplitude_roll_dumbbell 
##                   0.9793                   0.0000                   0.9793 
## amplitude_pitch_dumbbell   amplitude_yaw_dumbbell     total_accel_dumbbell 
##                   0.9793                   0.0000                   0.0000 
##       var_accel_dumbbell        avg_roll_dumbbell     stddev_roll_dumbbell 
##                   0.9793                   0.9793                   0.9793 
##        var_roll_dumbbell       avg_pitch_dumbbell    stddev_pitch_dumbbell 
##                   0.9793                   0.9793                   0.9793 
##       var_pitch_dumbbell         avg_yaw_dumbbell      stddev_yaw_dumbbell 
##                   0.9793                   0.9793                   0.9793 
##         var_yaw_dumbbell         gyros_dumbbell_x         gyros_dumbbell_y 
##                   0.9793                   0.0000                   0.0000 
##         gyros_dumbbell_z         accel_dumbbell_x         accel_dumbbell_y 
##                   0.0000                   0.0000                   0.0000 
##         accel_dumbbell_z        magnet_dumbbell_x        magnet_dumbbell_y 
##                   0.0000                   0.0000                   0.0000 
##        magnet_dumbbell_z             roll_forearm            pitch_forearm 
##                   0.0000                   0.0000                   0.0000 
##              yaw_forearm    kurtosis_roll_forearm   kurtosis_picth_forearm 
##                   0.0000                   0.0000                   0.0000 
##     kurtosis_yaw_forearm    skewness_roll_forearm   skewness_pitch_forearm 
##                   0.0000                   0.0000                   0.0000 
##     skewness_yaw_forearm         max_roll_forearm        max_picth_forearm 
##                   0.0000                   0.9793                   0.9793 
##          max_yaw_forearm         min_roll_forearm        min_pitch_forearm 
##                   0.0000                   0.9793                   0.9793 
##          min_yaw_forearm   amplitude_roll_forearm  amplitude_pitch_forearm 
##                   0.0000                   0.9793                   0.9793 
##    amplitude_yaw_forearm      total_accel_forearm        var_accel_forearm 
##                   0.0000                   0.0000                   0.9793 
##         avg_roll_forearm      stddev_roll_forearm         var_roll_forearm 
##                   0.9793                   0.9793                   0.9793 
##        avg_pitch_forearm     stddev_pitch_forearm        var_pitch_forearm 
##                   0.9793                   0.9793                   0.9793 
##          avg_yaw_forearm       stddev_yaw_forearm          var_yaw_forearm 
##                   0.9793                   0.9793                   0.9793 
##          gyros_forearm_x          gyros_forearm_y          gyros_forearm_z 
##                   0.0000                   0.0000                   0.0000 
##          accel_forearm_x          accel_forearm_y          accel_forearm_z 
##                   0.0000                   0.0000                   0.0000 
##         magnet_forearm_x         magnet_forearm_y         magnet_forearm_z 
##                   0.0000                   0.0000                   0.0000 
##                   classe 
##                   0.0000
```

We can either impute them or prune them. We see that if column contains NAs, then 98% of the values are NAs.
In this case we made a decision to prune the columns with a large number of NAs.


```r
naVars <- apply(d, 2, function(col)sum(is.na(col))/length(col)) > 0.95
d <- d[,names(naVars[naVars == F])]
```

There are also columns that identify the user and have a timestamp - we exclude these from the model building, as 
we are looking to pridict the result based on accelerometer data, not the time when measurements occured or user who
executed the exercise.


```r
d <- d[,!(names(d) %in% c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window"))]
```

Looking at the number of variables it is pretty big - 84. Let's remove columns with low variance:


```r
zeroVarCols <- nearZeroVar(d)
d <- d[,-zeroVarCols]
```

Split into training and test sets for cross-validation


```r
inTrain <- createDataPartition(d$classe, p=0.6, list=FALSE)
training <- d[inTrain, ]
cv <- d[-inTrain,]
```

Train classification tree

```r
set.seed(12345)
rpartFit <- train(classe ~ ., method="rpart", data=training)
```

Do a sanity check of the model - is it capable of producing all result levels A-E

```r
print(rpartFit$finalModel)
```

```
## n= 11776 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 11776 8428 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 10780 7442 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -33.95 982    7 A (0.99 0.0071 0 0 0) *
##      5) pitch_forearm>=-33.95 9798 7435 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 436.5 8250 5939 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 124.5 5185 3097 A (0.4 0.18 0.19 0.17 0.059) *
##         21) roll_forearm>=124.5 3065 2035 C (0.073 0.18 0.34 0.23 0.18) *
##       11) magnet_dumbbell_y>=436.5 1548  777 B (0.034 0.5 0.041 0.23 0.2) *
##    3) roll_belt>=130.5 996   10 E (0.01 0 0 0 0.99) *
```

Estimate out of sample error using cross validation set

```r
cm <- confusionMatrix(cv$classe, predict(rpartFit, cv))
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2010   43  175    0    4
##          B  624  531  363    0    0
##          C  632   45  691    0    0
##          D  567  224  495    0    0
##          E  214  190  393    0  645
## 
## Overall Statistics
##                                         
##                Accuracy : 0.494         
##                  95% CI : (0.483, 0.505)
##     No Information Rate : 0.516         
##     P-Value [Acc > NIR] : 1             
##                                         
##                   Kappa : 0.339         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.497   0.5140   0.3264       NA   0.9938
## Specificity             0.942   0.8551   0.8818    0.836   0.8893
## Pos Pred Value          0.901   0.3498   0.5051       NA   0.4473
## Neg Pred Value          0.637   0.9207   0.7799       NA   0.9994
## Prevalence              0.516   0.1317   0.2698    0.000   0.0827
## Detection Rate          0.256   0.0677   0.0881    0.000   0.0822
## Detection Prevalence    0.284   0.1935   0.1744    0.164   0.1838
## Balanced Accuracy       0.719   0.6846   0.6041       NA   0.9415
```

This model is not performing great with 49.4137% accuracy.

Train random forest and compare its performance to classification tree

```r
rfFit <- randomForest(classe ~ ., data=training)
cm <- confusionMatrix(cv$classe, predict(rfFit, cv))
cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    0    0    0    1
##          B    2 1512    4    0    0
##          C    0   12 1355    1    0
##          D    0    0   14 1270    2
##          E    0    0    2    7 1433
## 
## Overall Statistics
##                                         
##                Accuracy : 0.994         
##                  95% CI : (0.992, 0.996)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.993         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.999    0.992    0.985    0.994    0.998
## Specificity             1.000    0.999    0.998    0.998    0.999
## Pos Pred Value          1.000    0.996    0.990    0.988    0.994
## Neg Pred Value          1.000    0.998    0.997    0.999    1.000
## Prevalence              0.285    0.194    0.175    0.163    0.183
## Detection Rate          0.284    0.193    0.173    0.162    0.183
## Detection Prevalence    0.284    0.193    0.174    0.164    0.184
## Balanced Accuracy       0.999    0.996    0.992    0.996    0.998
```

Random forest gives a much better model with 99.4265% accuracy.

Remove variables from testing set that we didn't use for training

```r
testing <- testing[,colnames(d)[-length(colnames(d))]]
```

Now let's predict data for the testing set

```r
answers <- predict(rfFit, testing)
```

And save data into files for submission

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files(answers)
```

