## LOAD LIBRARIES

library(mlr3verse)
library(mlr3)
library(mlr3viz) 
library(mlr3learners) 
library(mlr3tuning) 
library(readxl) 
library(precrec)
library(apcluster)
library(ranger)
library(randomForest)
library(performanceEstimation)
library(caTools)
library(bbotk)
library(data.table)
library(DiceKriging)
library(mlr3mbo)
library(nloptr)
library(patchwork)
library(pROC)

max_n_evals <- 10
n.rep <- 5
CV_results_all_learners <- matrix(NA, ncol = 5, nrow = n.rep)

####################################
#
# DATA PREPARATION
#
####################################

# READ DATA
handpump <- read.csv("handpump_data.csv", header = TRUE)
handpump_narm <- na.omit(handpump) # Remove rows with missing values.

# Remove irrelevant predictors.
preprocessed_hp <- subset(handpump_narm, 
                          select = -c(site_barcode, local_date, pf_overall_stl_seasonal_percent_log, nevents_overall_stl_seasonal_percent_log, nevents_wd_simple_percent_log, nevents_wd_cluster_percent_log, nevents_overall_cluster_percent_log, naive_failure,pred_currently_failed_glm, pred_fail_next_week_glm, pred_currently_failed_SL, pred_fail_next_week_SL, 
                                      currently_failed))

preprocessed_hp$fail_next_week <- factor(preprocessed_hp$fail_next_week + 0)

# Define Classification Task

hp_task = as_task_classif(preprocessed_hp, target = "fail_next_week", positive = "1")
hp_task$positive # Show positive class.

# Create data partition. 70% for training and the rest for testing.

set.seed(4285452) # Add seed for reproducibility.
splits = partition(hp_task, ratio = 0.7)
sample <- sample.split(preprocessed_hp, SplitRatio = 0.7)
train  <- subset(preprocessed_hp, sample == TRUE)
test   <- subset(preprocessed_hp, sample == FALSE)

## Created Balanced Data using SMOTE

tibble::rowid_to_column(train, "ID")
tibble::rowid_to_column(test, "ID")

train$fail_next_week <- as.factor(train$fail_next_week) # make target variable a factor

# originally 0 had 3762 obs. while 1 had 253.
# We use the default values of the function.
set.seed(4285452) # Add seed for reproducibility.
train <- smote(fail_next_week ~ ., train, perc.over = 1, perc.under = 2, k = 5)
table(train$fail_next_week)

smote_hp_task = as_task_classif(train, target = "fail_next_week", positive = "1")
smote_hp_task$positive # Show positive class
# References: https://youtu.be/1Mt7EuVJf1A and https://www.rdocumentation.org/packages/DMwR/versions/0.4.1/topics/SMOTE

######################################
#
# STANDARD RANDOM FOREST
#
######################################

basic_learner_classif = lrn("classif.ranger", predict_type = "prob")
default.parameters <- list()
default.parameters$num.threads <- 1
default.parameters$num.trees <- 500
default.parameters$mtry <- 2
default.parameters$min.node.size <- 1
default.parameters$replace <- FALSE
default.parameters$splitrule <- "gini"
default.parameters$sample.fraction <- 1

basic_learner_classif$param_set$values = default.parameters
print(basic_learner_classif$param_set$values) # Print default parameters

# Define type of resampling technique
cvFive = rsmp("cv", folds = 5)
# Initiate resampling technique
cvFive$instantiate(smote_hp_task)
# Train model
rr = resample(smote_hp_task, basic_learner_classif, cvFive)
# Print compiled score. Overall, cross-validation estimate.
cv_basic = rr$aggregate(msr("classif.recall"))

######################################
#
# HYPERPARAMETER OPTIMIZATION
#
######################################

learner_classif = lrn("classif.ranger", predict_type = "prob",
                      mtry = to_tune(1, 5),
                      min.node.size = to_tune(1, 500),
                      replace = to_tune(),
                      splitrule = to_tune(c("gini", "extratrees")),
                      sample.fraction = to_tune(0.1, 1),
                      num.trees = 500) 

## Random search
set.seed(4285452) # Add seed for reproducibility.

for (i in 1:n.rep){
  RS_instance = tune(tuner = tnr("random_search"), task = smote_hp_task,
                     learner = learner_classif, resampling = rsmp("cv", folds = 5),
                     measure = msr("classif.recall"), 
                     terminator = trm("evals", n_evals = max_n_evals))
  
  
  ## Grid search
  GS_instance = tune(tuner = tnr("grid_search", resolution = 5), 
                     task = smote_hp_task, learner = learner_classif, 
                     resampling = rsmp("cv", folds = 5), 
                     measures = msr("classif.recall"), 
                     terminator = trm("evals", n_evals = max_n_evals))
  
  ## Bayesian Optimization
  bayesopt_ego = mlr_loop_functions$get("bayesopt_ego")
  surrogate = srlrn(lrn("regr.km", covtype = "matern5_2",
                        optim.method = "BFGS", control = list(trace = FALSE)))
  acq_function = acqf("ei")
  acq_optimizer = acqo(opt("nloptr", algorithm = "NLOPT_GN_ORIG_DIRECT"),
                       terminator = trm("stagnation", iters = 100, threshold = 1e-5))
  
  tuner = tnr("mbo", loop_function = bayesopt_ego, surrogate = surrogate, 
              acq_function = acq_function, acq_optimizer = acq_optimizer)
  
  #set.seed(4285452) # Add seed for reproducibility.
  BO_instance = tune(tuner, smote_hp_task, learner_classif, rsmp("cv", folds = 5),
                     msr("classif.recall"), 
                     terminator = trm("evals", n_evals = max_n_evals))
  
  ######################################
  #
  # TRAIN OPTIMIZED LEARNERS 
  #
  ######################################
  
  RS_learner_classif = lrn("classif.ranger", predict_type = "prob")
  GS_learner_classif = lrn("classif.ranger", predict_type = "prob")
  BO_learner_classif = lrn("classif.ranger", predict_type = "prob")
  
  RS_learner_classif$param_set$values = RS_instance$result$learner_param_vals[[1]]
  GS_learner_classif$param_set$values = GS_instance$result$learner_param_vals[[1]]
  BO_learner_classif$param_set$values = BO_instance$result$learner_param_vals[[1]]
  
  # Comparisons using 5-fold cross-validation estimates
  
  # Define type of resampling technique
  cvFive = rsmp("cv", folds = 5)
  # Initiate resampling technique
  cvFive$instantiate(smote_hp_task)
  
  # Train models and evaluate using cross validation
  RS_rr = resample(smote_hp_task, RS_learner_classif, cvFive)
  GS_rr = resample(smote_hp_task, GS_learner_classif, cvFive)
  BO_rr = resample(smote_hp_task, BO_learner_classif, cvFive)
  
  # Print compiled score. Overall, cross-validation estimate.
  cv_rs = RS_rr$aggregate(msr("classif.recall"))
  cv_gs = GS_rr$aggregate(msr("classif.recall"))
  cv_BO = BO_rr$aggregate(msr("classif.recall"))
  
  CV_results_all_learners[i,] <- c(max_n_evals, cv_rs, cv_gs, cv_BO, cv_basic)
  
}

file_to_save <- paste("Results_", max_n_evals, "_nRepetition_", n.rep, ".csv",
                      sep = "")
write.csv(CV_results_all_learners, file_to_save)
