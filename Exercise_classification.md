Summary
-------

Six participants were asked to perform ten repetitions of the Unilateral
Dumbbell Biceps Curl in five different fashions, each measured using an
accelerometer. Original dataset is credited to the [Human Activity
Recognition](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har)
project. Five classes of activity were monitored using a pair of
dumbells:

-   Class A: Performed exactly according to instructions
-   Class B: Throwing the elbows to the front
-   Class C: Lifting the dumbbell only halfway
-   Class D: Lowering the dumbbell only halfway
-   Class E: Throwing the hips to the front

Our task was to categorically assign which of the five activities were
be performed by participants based on the available accelerometer
activity data. It was found that the Random Forest model was the best
classifier.

### Methodology

We compared the accuracy of Support Vector Machine
[SVM](https://cran.r-project.org/web/packages/e1071/vignettes/svmdoc.pdf)
and Ramdom Forest
[RF](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf)
machine learning packages, for predicting exercise classes A through E.
Evaluation was done through cross validating, subsetting 80% of our
data, then testing on the remaining 20%. Using our best predictor (RF),
we estimated the robustness of our predictors through principal
component analysis (PCA) taking the top PCs and top-ranked RF variables
found in our accelerometer training set. Both subsets performed better
than SVM model.

    #loading libraries
    library(reshape)
    library(dplyr)
    library(AppliedPredictiveModeling)
    library(e1071)
    library(randomForest)
    library(ElemStatLearn)
    library(caret)
    library(corrplot)
    library(knitr)

    #reproducible results
    set.seed(999)

    #downloading data
    if (!file.exists("pml-training.csv")){
            url1 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
            download.file(url1,"pml-training.csv")
            
            url2 <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
            download.file(url2,"pml-testing.csv")
    }

    #loading test/train, defining NAs
    trainset <- read.csv("pml-training.csv",stringsAsFactors = T, na.strings = c("NA","","#DIV/0!"))
    validation <- read.csv("pml-testing.csv",stringsAsFactors = T, na.strings = c("NA","","#DIV/0!"))

### Data preparation

We the training dataset has 19622 rows and 160 columns from which to
subset for cross validation. The first 7 columns do not contain
accelerometer data, but rather names and timestamps, which we do not
want to include. Furthermore, several columns within appear to have NA
values. We will select only numeric columns for our analysis.

    #subset all nonzero, non-relvant values
    trainset <- trainset[,!is.na(trainset[3,])] %>% select(roll_belt:classe)

We are left with 53 variables. Plotting a subset of the data:

![](Exercise_classification_files/figure-markdown_strict/plotting%20subset-1.png)

The data we have selected appears to show enough variance (and few
outliers) to merit including in our model. Next we split our data into
testing and training, and test the accuracy of the SVM model.

### Machine learning evaluation: RF vs SVM

Start with data preparation, dividing into our test and train groups in
80/20 ratio.

    #split train set for cross validation
    trainIndex <- createDataPartition(trainset$classe, p = 0.8, list=F)
    training <- trainset[trainIndex,]
    testing <- trainset[-trainIndex,]

    ## SVM package prediction ##
    modSVM <- e1071::svm(classe ~ ., data=training)
    predSVM <- predict(modSVM,testing)

    #SVM prediction accurary
    cfSVM <- confusionMatrix(predSVM,testing$classe)

    cfSVM$table

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1114   50    0    1    0
    ##          B    0  698   16    0    5
    ##          C    1   11  657   44   12
    ##          D    0    0    5  598   17
    ##          E    1    0    6    0  687

Accuracy for SVM is **0.957**. Out of sample error rate is then
**4.3**%. Next we test the predictive power of the Random Forest
package.

    ## RF package prediction ##
    control <- trainControl(method = "cv", number = 5)
    modRF <- randomForest(classe ~., data = training, trControl = control)
    predRF <- predict(modRF,testing)

    print(modRF,digits = 4)

    ## 
    ## Call:
    ##  randomForest(formula = classe ~ ., data = training, trControl = control) 
    ##                Type of random forest: classification
    ##                      Number of trees: 500
    ## No. of variables tried at each split: 7
    ## 
    ##         OOB estimate of  error rate: 0.36%
    ## Confusion matrix:
    ##      A    B    C    D    E  class.error
    ## A 4462    2    0    0    0 0.0004480287
    ## B    9 3027    2    0    0 0.0036208032
    ## C    0   13 2722    3    0 0.0058436815
    ## D    0    0   20 2552    1 0.0081616790
    ## E    0    0    1    5 2880 0.0020790021

    # prediction accurary
    cfRF <- confusionMatrix(predRF,testing$classe)

Accuracy for RF is **0.997**. Out of sample error rate is then **0.3**%.
Random Forest is clearly the superior of the two methods. We expect it
will predict 100% of the 20 Quiz test cases, which was verified.
Subsetting to the 13 most important variabbles (25% of total) we can see
how much accuracy comes from these few variables:

<table>
<thead>
<tr class="header">
<th align="right">Overall</th>
<th align="left">names</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td align="right">982</td>
<td align="left">roll_belt</td>
</tr>
<tr class="even">
<td align="right">729</td>
<td align="left">yaw_belt</td>
</tr>
<tr class="odd">
<td align="right">665</td>
<td align="left">pitch_forearm</td>
</tr>
<tr class="even">
<td align="right">611</td>
<td align="left">magnet_dumbbell_z</td>
</tr>
<tr class="odd">
<td align="right">571</td>
<td align="left">pitch_belt</td>
</tr>
<tr class="even">
<td align="right">543</td>
<td align="left">magnet_dumbbell_y</td>
</tr>
<tr class="odd">
<td align="right">477</td>
<td align="left">roll_forearm</td>
</tr>
<tr class="even">
<td align="right">381</td>
<td align="left">magnet_dumbbell_x</td>
</tr>
<tr class="odd">
<td align="right">333</td>
<td align="left">accel_dumbbell_y</td>
</tr>
<tr class="even">
<td align="right">321</td>
<td align="left">roll_dumbbell</td>
</tr>
<tr class="odd">
<td align="right">320</td>
<td align="left">magnet_belt_z</td>
</tr>
<tr class="even">
<td align="right">316</td>
<td align="left">accel_belt_z</td>
</tr>
<tr class="odd">
<td align="right">311</td>
<td align="left">magnet_belt_y</td>
</tr>
</tbody>
</table>

![](Exercise_classification_files/figure-markdown_strict/ranking%20RF-1.png)

Accuracy for RF subset is **0.99**, including the top 3 roll belt, yaw
belt, and pitch forearm. We now compare this RF subset with a PCA subset
approach (same size of subset).

### Principle component analysis

Rescaling and subsetting PCA data.

    #determine PCA fraction
    HAR.pca <- prcomp(training[,-ncol(training)], center = TRUE, scale. = TRUE) 
    PCArank <- summary(HAR.pca)$importance %>% t() %>% as.data.frame()

    #list only variables that explain up to 90% of variance
    topPredict <- sum(PCArank$`Cumulative Proportion` < 0.9)

Note that 19 variables explain 90% of total variance in activity data.

    #preprocess with the desired number of PC's
    preProc <- preProcess(training[,-ncol(training)],method = "pca", pcaComp = 13)

    #create PC matrices for test and training data
    trainPC <- predict(preProc,training[,-ncol(training)])
    testPC <- predict(preProc,testing[,-ncol(testing)])

    #fit the RF model with PCA variables
    modelFitPC <- randomForest(classe ~., data = data.frame(trainPC,classe = training$classe))

    #see how well PCA worked compared to original
    cfPCA <- confusionMatrix(testing$classe, predict(modelFitPC,testPC))

    cfPCA$table

    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 1095    7    6    6    2
    ##          B   16  727   15    0    1
    ##          C    4    8  657   10    5
    ##          D    6    0   11  623    3
    ##          E    2    6    5    5  703

Accuracy for PCA subset variables is **0.97** using 13 variables.

### Conclusion

Random Forest and SVM are both capable classifiers, however the RF
approach is superior, having correctly predicted all 20 results in test
set (not shown for honor code purposes). Distinguishing between Class A
and B appears to be the most challenging. Furthermore, subsetting RF via
the 13 most important variables fared better than principle component
analysis, hence both are viable compression options.

### APPENDIX

Complete list of variables

    str(trainset)

    ## 'data.frame':    19622 obs. of  53 variables:
    ##  $ roll_belt           : num  1.41 1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 ...
    ##  $ pitch_belt          : num  8.07 8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.17 ...
    ##  $ yaw_belt            : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
    ##  $ total_accel_belt    : int  3 3 3 3 3 3 3 3 3 3 ...
    ##  $ gyros_belt_x        : num  0 0.02 0 0.02 0.02 0.02 0.02 0.02 0.02 0.03 ...
    ##  $ gyros_belt_y        : num  0 0 0 0 0.02 0 0 0 0 0 ...
    ##  $ gyros_belt_z        : num  -0.02 -0.02 -0.02 -0.03 -0.02 -0.02 -0.02 -0.02 -0.02 0 ...
    ##  $ accel_belt_x        : int  -21 -22 -20 -22 -21 -21 -22 -22 -20 -21 ...
    ##  $ accel_belt_y        : int  4 4 5 3 2 4 3 4 2 4 ...
    ##  $ accel_belt_z        : int  22 22 23 21 24 21 21 21 24 22 ...
    ##  $ magnet_belt_x       : int  -3 -7 -2 -6 -6 0 -4 -2 1 -3 ...
    ##  $ magnet_belt_y       : int  599 608 600 604 600 603 599 603 602 609 ...
    ##  $ magnet_belt_z       : int  -313 -311 -305 -310 -302 -312 -311 -313 -312 -308 ...
    ##  $ roll_arm            : num  -128 -128 -128 -128 -128 -128 -128 -128 -128 -128 ...
    ##  $ pitch_arm           : num  22.5 22.5 22.5 22.1 22.1 22 21.9 21.8 21.7 21.6 ...
    ##  $ yaw_arm             : num  -161 -161 -161 -161 -161 -161 -161 -161 -161 -161 ...
    ##  $ total_accel_arm     : int  34 34 34 34 34 34 34 34 34 34 ...
    ##  $ gyros_arm_x         : num  0 0.02 0.02 0.02 0 0.02 0 0.02 0.02 0.02 ...
    ##  $ gyros_arm_y         : num  0 -0.02 -0.02 -0.03 -0.03 -0.03 -0.03 -0.02 -0.03 -0.03 ...
    ##  $ gyros_arm_z         : num  -0.02 -0.02 -0.02 0.02 0 0 0 0 -0.02 -0.02 ...
    ##  $ accel_arm_x         : int  -288 -290 -289 -289 -289 -289 -289 -289 -288 -288 ...
    ##  $ accel_arm_y         : int  109 110 110 111 111 111 111 111 109 110 ...
    ##  $ accel_arm_z         : int  -123 -125 -126 -123 -123 -122 -125 -124 -122 -124 ...
    ##  $ magnet_arm_x        : int  -368 -369 -368 -372 -374 -369 -373 -372 -369 -376 ...
    ##  $ magnet_arm_y        : int  337 337 344 344 337 342 336 338 341 334 ...
    ##  $ magnet_arm_z        : int  516 513 513 512 506 513 509 510 518 516 ...
    ##  $ roll_dumbbell       : num  13.1 13.1 12.9 13.4 13.4 ...
    ##  $ pitch_dumbbell      : num  -70.5 -70.6 -70.3 -70.4 -70.4 ...
    ##  $ yaw_dumbbell        : num  -84.9 -84.7 -85.1 -84.9 -84.9 ...
    ##  $ total_accel_dumbbell: int  37 37 37 37 37 37 37 37 37 37 ...
    ##  $ gyros_dumbbell_x    : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ gyros_dumbbell_y    : num  -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 -0.02 ...
    ##  $ gyros_dumbbell_z    : num  0 0 0 -0.02 0 0 0 0 0 0 ...
    ##  $ accel_dumbbell_x    : int  -234 -233 -232 -232 -233 -234 -232 -234 -232 -235 ...
    ##  $ accel_dumbbell_y    : int  47 47 46 48 48 48 47 46 47 48 ...
    ##  $ accel_dumbbell_z    : int  -271 -269 -270 -269 -270 -269 -270 -272 -269 -270 ...
    ##  $ magnet_dumbbell_x   : int  -559 -555 -561 -552 -554 -558 -551 -555 -549 -558 ...
    ##  $ magnet_dumbbell_y   : int  293 296 298 303 292 294 295 300 292 291 ...
    ##  $ magnet_dumbbell_z   : num  -65 -64 -63 -60 -68 -66 -70 -74 -65 -69 ...
    ##  $ roll_forearm        : num  28.4 28.3 28.3 28.1 28 27.9 27.9 27.8 27.7 27.7 ...
    ##  $ pitch_forearm       : num  -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.9 -63.8 -63.8 -63.8 ...
    ##  $ yaw_forearm         : num  -153 -153 -152 -152 -152 -152 -152 -152 -152 -152 ...
    ##  $ total_accel_forearm : int  36 36 36 36 36 36 36 36 36 36 ...
    ##  $ gyros_forearm_x     : num  0.03 0.02 0.03 0.02 0.02 0.02 0.02 0.02 0.03 0.02 ...
    ##  $ gyros_forearm_y     : num  0 0 -0.02 -0.02 0 -0.02 0 -0.02 0 0 ...
    ##  $ gyros_forearm_z     : num  -0.02 -0.02 0 0 -0.02 -0.03 -0.02 0 -0.02 -0.02 ...
    ##  $ accel_forearm_x     : int  192 192 196 189 189 193 195 193 193 190 ...
    ##  $ accel_forearm_y     : int  203 203 204 206 206 203 205 205 204 205 ...
    ##  $ accel_forearm_z     : int  -215 -216 -213 -214 -214 -215 -215 -213 -214 -215 ...
    ##  $ magnet_forearm_x    : int  -17 -18 -18 -16 -17 -9 -18 -9 -16 -22 ...
    ##  $ magnet_forearm_y    : num  654 661 658 658 655 660 659 660 653 656 ...
    ##  $ magnet_forearm_z    : num  476 473 469 469 473 478 470 474 476 473 ...
    ##  $ classe              : Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
