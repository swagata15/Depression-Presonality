# libraries
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.model_selection import LeaveOneOut
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RandomizedLogisticRegression
import time
import sys
# my code
from outcomeHandler import get_relevant_outcomes
from storeFeatMatsInCSV import genSuffixList
from utils.preprocessingHelper import *
import os
from nested_feature_selection_CLEAN_general_config import getConfig, getBestConfig

# VERY IMPORTANT FOR REPRODUCIBLE RESULTS!!
np.random.seed(0)

serverDataPath = "/Users/swagataashwani/Desktop/Feature-Analysis-SUP-Swagata"

import warnings
warnings.filterwarnings("ignore")


#utils
def printAllRows(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', 3):
        print(df)

# settings
THRES_PERCENT_NANS_PER_SUBJECT = 0.2 # must be less than this
THRES_NUM_NANS_PER_COL = 30 # if a column is null in more than these many people, it'll be removed.

# timer
startt = time.time()
n_jobs = 1

# DOES REGULAR, EXC, PRE-BDI-2
#  filename and sensor settings

toThresholdBCBV = True
print (toThresholdBCBV)
sen_epochs = ["", "_mo", "_af", "_ev", "_ni"]
sen_weekdays = ["", "_wkdy", "_wkend"]
sen_grps = ["_sem", "_half_sem", "_week"] # will be changed to _week if _BC or _BV

def getExcldSemHalfsemFromWks(wklist):
    excfirsthalf = 0
    excsecondhalf = 0
    for w in wklist:
        if w>=1 and w<=6:
            excfirsthalf = 1
        if w>=7 and w<=16:
            excsecondhalf = 1
    if excfirsthalf == 1 or excsecondhalf == 1:
        excsem = range(1,17)
    else:
        excsem = []
    if excfirsthalf == 1 and excsecondhalf == 1:
        exchalfsem = range(1,17)
    elif excfirsthalf == 1 and excsecondhalf == 0:
        exchalfsem = range(1, 7)
    elif excfirsthalf == 0 and excsecondhalf == 1:
        exchalfsem = range(7, 17)
    else:
        exchalfsem = []
    return (excsem, exchalfsem)

## TO CONFIGURE
# excluded weeks
excwks = []
excsem, exchalfsem = getExcldSemHalfsemFromWks(excwks)
excld_weeks_for_grp = { # can be from 1 to 16
    "_sem" : excsem, # always range(1,17) while excluding any week
    "_half_sem" : exchalfsem, # to exclude 1st half use range(1,7). To exclude 2nd half use range(7,17). To exclude both, use (1,17)
    "_week" : excwks
}
## 6 parameters:
## 1) sensorname = {f_batt, f_blue, f_call, f_calor, f_loc, f_locMap, f_screen, f_sleep, f_steps, f_wifi}
## 2) outcome_status = {post_bdi_2, change_bdi_2, post_bdi_2_levelsC, change_bdi_2_levelsC}
## 3) prebdi_status = {NO, SC, LBL}, # selParamsDict does not dependent on prebdi_status
## 4) modelname = {GBC, LOGR} # selParamsDict does not dependent on modelname
## 5) suffix_foldername = NUMBER >=1, see nested_feature_selection_CLEAN_general_config.py for available configurations
## 6) selParamsDict = getConfig(sensorname, outcome_status, suffix_foldername, n_jobs)

## outcome_status
## post_bdi_2: depressed vs not depressed
## change_bdi_2: getting worse vs staying the same (only 3 get better which we throw away)
## post_bdi_2_levelsC (For N=138, 0: 82, 1: 25, 2: 19, 3: 12): classifying into severity levels (not regression)
## change_bdi_2_levelsC (For N=138, 0: 88, -1: 26, -2: 16, -3: 5, 1:3): classifying into severity levels (not regression)

## To run independent sensor models and tune model parameters, use this
sensorname = "f_slp_BC" # change this
bc = True
bv = False
outcome_status = "change_bdi_2" # post_bdi_2 or change_bdi_2 or post_bdi_2_levelsC or change_bdi_2_levelsC
prebdi_status = "NO" #2 # "NO": Do not add, "SC" : add score, "LBL" : add label
modelname = "GBC" #3 # GBC or LOGR
suffix_foldername = 41
selParamsDict = getConfig(sensorname, outcome_status, suffix_foldername, n_jobs)

## Using Best sensor models instead of above: Eg: for limiting weeks, etc we cannot tune model params
##sensorname = "f_call_BC" # change this
##bc = True
##bv = False
##outcome_status = "change_bdi_2"
##prebdi_status = "NO"
##modelname, selParamsDict, suffix_foldername = getBestConfig(sensorname, outcome_status, n_jobs)

## END CONFIGURATION
limflag = ""
for k in excld_weeks_for_grp.keys():
    if len(excld_weeks_for_grp[k])>0:
        limflag = "EXC"
        break
if limflag == "EXC":
    for k in excld_weeks_for_grp.keys():
        if len(excld_weeks_for_grp[k])>0:
            excldwksfork = np.array(excld_weeks_for_grp[k])
            excldwksmin = np.min(excldwksfork)
            excldwksmax = np.max(excldwksfork)
            limflag = limflag + "{2}Wk{0}to{1}".format(excldwksmin, excldwksmax, k[1])

if sensorname[-3:] == "_BC" or sensorname[-3:] == "_BV":
    sen_grps = ["_week"]

if modelname == "GBC":
    modelnameinfname = ""
elif modelname == "LOGR":
    modelnameinfname = "NOGBC"
else:
    raise Exception ("Unrecognized modelname")

if prebdi_status not in ["NO", "SC", "LBL"]:
    raise Exception("Pre BDI status type unrecognized!")

sen = [sensorname]
if outcome_status == "post_bdi_2":
    outsuffix = ""
elif outcome_status == "change_bdi_2":
    outsuffix = "outChange_"
elif outcome_status == "change_bdi_2_levelsC":
    outsuffix = "outChangeLvlsC_"
elif outcome_status == "post_bdi_2_levelsC":
    outsuffix = "outSeverityC_"
else:
    raise Exception("Unrecognized outcome_status {0}".format(outcome_status))

folderpath = serverDataPath+"/models/{0}".format(sensorname)+"_{3}fsresults{0}{2}prebdi{1}".format(limflag, prebdi_status, modelnameinfname, outsuffix)+"_rlog_{0}/"
folderpath = folderpath.format(suffix_foldername)
print ("Created folder {0}".format(folderpath))
if os.path.isdir(folderpath):
    raise Exception("{0} folder already exists".format(folderpath))
else:
    os.makedirs(folderpath)


suffix_list = genSuffixList(sen, sen_epochs, sen_weekdays, sen_grps)
print (len(suffix_list))

datestr_to_week = {
    "2017-01-18" : 1,
    "2017-01-25" : 2,
    "2017-02-01" : 3,
    "2017-02-08" : 4,
    "2017-02-15" : 5,
    "2017-02-22" : 6,
    "2017-03-01" : 7,
    "2017-03-08" : 8,
    "2017-03-15" : 9,
    "2017-03-22" : 10,
    "2017-03-29" : 11,
    "2017-04-05" : 12,
    "2017-04-12" : 13,
    "2017-04-19" : 14,
    "2017-04-26" : 15,
    "2017-05-03" : 16,
}
week_to_datestr = {
    1 : "2017-01-18",
    2 : "2017-01-25",
    3 : "2017-02-01",
    4 : "2017-02-08",
    5 : "2017-02-15",
    6 : "2017-02-22",
    7 : "2017-03-01",
    8 : "2017-03-08",
    9 : "2017-03-15",
    10 : "2017-03-22",
    11 : "2017-03-29",
    12 : "2017-04-05",
    13 : "2017-04-12",
    14 : "2017-04-19",
    15 : "2017-04-26",
    16 : "2017-05-03",
}
def excludeDates(df, wknumsToRm):
    if len(wknumsToRm) == 0:
        return (df)
    colslikelst = []
    for wknum in wknumsToRm:
        wkstr = week_to_datestr[wknum]
        colslikelst.append(wkstr)
    #print (colslikelst)
    colsToRm = getColsLikeInList(df, colslikelst)
    #print (colsToRm)
    df = df.drop(colsToRm, axis=1)
    return (df)


## Get Outcomes
def cleanOutcome(x):
    if x in [np.nan, None, 999, 999.0, "999", "999.0"] or x.strip() == "":
        return np.nan
    else:
        return int(x.strip())
demo_sel, pre_sel, post_sel = get_relevant_outcomes(outputObj = False, fetchFromFile = True)
#POST
print ("Retrieved outcomes:")
print (post_sel.columns)
postbdi_col = "Post_BDI_II_Score"
postbdi = post_sel[postbdi_col].apply(cleanOutcome).dropna()
print ("We have post-bdi-ii scores for {0} people".format(len(postbdi)))
# printAllRows(postbdi)
#PRE
print ("Retrieved outcomes:")
print(pre_sel.columns)
prebdi_col = "Pre_BDI_II_Score"
prebdi = pre_sel[prebdi_col].apply(cleanOutcome).dropna()
print ("We have pre-bdi-ii scores for {0} people".format(len(prebdi)))
# printAllRows(prebdi)

## Function to load from CSVs, remove num samples!
featMatPath = serverDataPath+"/featureMats/"
def fetchFromCSVS(suffix_list): # ALSO PREPROCESSES
    LoadedDict = {}
    LoadedList = []
    cnt = 0
    stotal = len(suffix_list)
    currfeat = None
    for s in suffix_list:
        if currfeat != s[0]:
            currfeat = s[0]
            print (currfeat)
#         print (s)
        #print ("Fetching {0} of {1}".format(cnt, stotal-1))
        #print (s)
        suffix_list_str = ",".join(s)
        #print (suffix_list_str)
        fpath = featMatPath+suffix_list_str+".csv"
        df = pd.read_csv(fpath)
        df = df.set_index('device_id')
        # pre-processing
        df, df_numsamps = separateNumSamples(df)
        df = excludeFeaturesToExclude(df)
        excld_weeks = excld_weeks_for_grp[s[-1]]
        df = excludeDates(df, excld_weeks)
        df = encodeCategoricalFeatures(df)
        df = cleanDataTypes(df)
        LoadedDict[suffix_list_str] = cnt
        LoadedList.append(df)
        cnt = cnt + 1
    return (LoadedDict, LoadedList)

## Load everything
LoadedDict, LoadedList = fetchFromCSVS(suffix_list)
print ("Loaded {0} dataframes".format(len(LoadedList)))

# functions for thresholding outcome scores
def thresholdChange(x):
    if x in [np.nan, None]:
        print ("None because value is {0}".format(x))
        return np.nan
    elif x > 0: # pre>post meaning depression got better
        return None # we should never actually get x>0 as all x>0 have been removed
    elif x < 0: # pre<post meaning depression got worse
        return 1 # label = 1 means getting worse
    else:
        return 0

def thresholdScore(x):
    if x in [np.nan, None]:
        print ("None because value is {0}".format(x))
        return np.nan
    elif x < 14:
        return 0
    else:
        return 1

def thresholdScoreForLevels(x):
    if x in [np.nan, None]:
        print ("None because value is {0}".format(x))
        return np.nan
    elif x < 14:
        return 0
    elif x >= 14 and x < 20:
        return 1
    elif x >= 20 and x < 29:
        return 2
    # elif x>=29 and x<64:
    elif x >= 29:
        return 3
    else:
        print ("None because value is {0}".format(x))
        return np.nan


# ## Pre-process all loaded featmats
def getXIdxsForFG(suffix_list_str, fgColNames, cols2matidx):
    if suffix_list_str not in fgColNames:
        return (None)
    fgcols = fgColNames[suffix_list_str]
    fgidxs = []
    fgidxs_names = []
#     print (cols2matidx.keys())
    for c in fgcols:
        if c in cols2matidx.keys():
            i = cols2matidx[c]
            fgidxs.append(i)
            fgidxs_names.append(c)
    return (fgidxs, fgidxs_names)

def concatAndPreprocessFGs(suffix_list_sel):
    ## assign a number to each FG for each reference
    fgIdxs = {}
    fgnum = 0
    for s in suffix_list_sel:
        suffix_list_str = ",".join(s)
        fgnum = fgnum + 1
        fgIdxs[suffix_list_str] = fgnum
    ## Concatenation
    fgColNames = {}
    print ("Begin concatenation")
#     print (suffix_list_sel)
    df_list = []
    for s in suffix_list_sel:
        suffix_list_str = ",".join(s)
        df = LoadedList[LoadedDict[suffix_list_str]]
#         print ("Found df of length = {0}".format(len(df)))
        if len(df) == 0:
            continue
        ## suffix FG number to column names
        fgnum = fgIdxs[suffix_list_str]
        df.columns = ['FG{0}_'.format(fgnum) + str(col) for col in df.columns]
        ## store feature names for each fg
        fgColNames[suffix_list_str] = df.columns
#         print (suffix_list_str)
#         print (df.columns)
        ## append df
        df_list.append(df)
    featmat = pd.concat(df_list, axis=1)
    print ("Final featmat length = {0}".format(len(featmat)))
    ## Preprocessing - nan handling (same as old)
    featmat = removeColsAndSubjectsWithThresNans(featmat, THRES_NUM_NANS_PER_COL, THRES_PERCENT_NANS_PER_SUBJECT)
    featmat = imputeForRemainingNans(featmat, -1)
    print ("Num features after nan removal = {0}".format(len(featmat.columns)))
    devices_to_keep = featmat.index
    if prebdi_status == "LBL":
        prebdi_local = prebdi.apply(thresholdScore)
        prebdi_local = prebdi_local.loc[devices_to_keep]
    elif prebdi_status == "SC":
        prebdi_local = prebdi.loc[devices_to_keep]
    if prebdi_status == "SC" or prebdi_status == "LBL":
        featmat = pd.concat([prebdi_local, featmat], axis=1)
        print ("Num features after adding prebdi = {0}".format(len(featmat.columns)))
    ## Map column names to index
    cols2matidx = {}
    cols = featmat.columns
    for i in range(0, len(cols)):
        coliname = cols[i]
        cols2matidx[coliname] = i
#     print (fgIdxs.keys())
#     print (fgColNames.keys())
#     print (cols2matidx.keys())
    ## fgColIdxs in featmat.values
    fgColIdxs = {}
    fgColNames_New = {}
    for s in suffix_list_sel:
        suffix_list_str = ",".join(s)
        fgColIdxs[suffix_list_str], fgColNames_New[suffix_list_str] = getXIdxsForFG(suffix_list_str, fgColNames, cols2matidx)
    return (featmat, fgIdxs, fgColNames_New, fgColIdxs)

def getChangeRelated(featmat_train, postbdi, prebdi):
    # add postbdi and prebdi columns to featmat
    featmat_train_with_postbdi_prebdi = pd.merge(featmat_train, pd.DataFrame(postbdi), left_index=True, right_index=True, how='inner')
    featmat_train_with_postbdi_prebdi = pd.merge(featmat_train_with_postbdi_prebdi, pd.DataFrame(prebdi), left_index=True, right_index=True, how='inner')
    # fetch postbdi column and threshold into levels
    postbdi_train = featmat_train_with_postbdi_prebdi[postbdi_col]
    postbdi_train_levels = postbdi_train.apply(thresholdScoreForLevels)
    # fetch prebdi column and threshold into levels
    prebdi_train = featmat_train_with_postbdi_prebdi[prebdi_col]
    prebdi_train_levels = prebdi_train.apply(thresholdScoreForLevels)
    # calculate changebdi_levels
    changebdi_train_levels = prebdi_train_levels - postbdi_train_levels
    # drop changebdi_levels >0 i.e. person is getting better because only 3 out of 138 people get better
    changebdi_train_levels_for_lbl = changebdi_train_levels.drop(changebdi_train_levels[(changebdi_train_levels > 0)].index)
    # threshold remaining
    changebdi_train_lbl = changebdi_train_levels_for_lbl.apply(thresholdChange)
    # drop postbdi and prebdi columns from featmat
    featmat_train = featmat_train_with_postbdi_prebdi.drop(postbdi_col, axis=1)
    featmat_train = featmat_train.drop(prebdi_col, axis=1)
    return (featmat_train, changebdi_train_levels, changebdi_train_lbl)

def getTrainTestSplit(featmat):
    if TEST_PERSON_DEVICE_ID not in list(featmat.index):
        return (None, None, None, None, 0)
    if len(featmat.columns) == 0:
        return (None, None, None, None, 0)
    # train
    featmat_train = featmat[~featmat.index.isin([TEST_PERSON_DEVICE_ID])]
    # test
    featmat_test = featmat[featmat.index.isin([TEST_PERSON_DEVICE_ID])]
    if outcome_status == "post_bdi_2":
        # train
        featmat_train_with_postbdi = pd.merge(featmat_train, pd.DataFrame(postbdi), left_index=True, right_index=True, how='inner')
        postbdi_train = featmat_train_with_postbdi[postbdi_col]
        postbdi_train_lbl = postbdi_train.apply(thresholdScore)
        featmat_train = featmat_train_with_postbdi.drop(postbdi_col, axis=1)
        # test
        featmat_test_with_postbdi = pd.merge(featmat_test, pd.DataFrame(postbdi), left_index=True, right_index=True, how='inner')
        postbdi_test = featmat_test_with_postbdi[postbdi_col]
        postbdi_test_lbl = postbdi_test.apply(thresholdScore)
        featmat_test = featmat_test_with_postbdi.drop(postbdi_col, axis=1)
        return (featmat_train, postbdi_train_lbl, featmat_test, postbdi_test_lbl, 1)
    elif outcome_status == "change_bdi_2":
        # train
        featmat_train, changebdi_train_levels_unused, changebdi_train_lbl = getChangeRelated(featmat_train, postbdi, prebdi)
        featmat_train = featmat_train.loc[changebdi_train_lbl.index]
        # test
        featmat_test, changebdi_test_levels_unused, changebdi_test_lbl = getChangeRelated(featmat_test, postbdi, prebdi)
        featmat_test = featmat_test.loc[changebdi_test_lbl.index]
        if (not featmat_train.index.equals(changebdi_train_lbl.index)):
            raise Exception ("train indexes don't match")
        if (not featmat_test.index.equals(changebdi_test_lbl.index)):
            raise Exception ("test indexes don't match")
        # the below step is needed as we are removing some subjects here to make it binary (subjects who improve)
        if len(featmat_test) == 0:
            return (None, None, None, None, 0)
        return (featmat_train, changebdi_train_lbl, featmat_test, changebdi_test_lbl, 1)
    elif outcome_status == "change_bdi_2_levelsC":
        # train
        featmat_train, changebdi_train_levels, changebdi_train_lbl_unused = getChangeRelated(featmat_train, postbdi, prebdi)
        # test
        featmat_test, changebdi_test_levels, changebdi_test_lbl_unused = getChangeRelated(featmat_test, postbdi, prebdi)
        if (not featmat_train.index.equals(changebdi_train_levels.index)):
            raise Exception ("train indexes don't match")
        if (not featmat_test.index.equals(changebdi_test_levels.index)):
            raise Exception ("test indexes don't match")
        return (featmat_train, changebdi_train_levels, featmat_test, changebdi_test_levels, 1)
    elif outcome_status == "post_bdi_2_levelsC":
        # train
        featmat_train_with_postbdi = pd.merge(featmat_train, pd.DataFrame(postbdi), left_index=True, right_index=True, how='inner')
        postbdi_train = featmat_train_with_postbdi[postbdi_col]
        postbdi_train_levels = postbdi_train.apply(thresholdScoreForLevels)
        featmat_train = featmat_train_with_postbdi.drop(postbdi_col, axis=1)
        # test
        featmat_test_with_postbdi = pd.merge(featmat_test, pd.DataFrame(postbdi), left_index=True, right_index=True, how='inner')
        postbdi_test = featmat_test_with_postbdi[postbdi_col]
        postbdi_test_levels = postbdi_test.apply(thresholdScoreForLevels)
        featmat_test = featmat_test_with_postbdi.drop(postbdi_col, axis=1)
        return (featmat_train, postbdi_train_levels, featmat_test, postbdi_test_levels, 1)
    else:
        raise Exception("outcome status {0} not recognized".format(outcome_status))

## Get Featmat
featmat_all, fgIdxs, fgColNames, fgColIdxs = concatAndPreprocessFGs(suffix_list)
print ("Final featmat_all {0} subjects, {1} features\n".format(len(featmat_all), len(featmat_all.columns)))

def getValueCounts(outcome_train_lbl, outcome_test_lbl):
    outcome_train_lbl_cnts = outcome_train_lbl.value_counts()
    outcome_test_lbl_cnts = outcome_test_lbl.value_counts()
    added_cnts = outcome_train_lbl_cnts.add(outcome_test_lbl_cnts, fill_value=0)
    added_cnts = added_cnts.to_dict()
    return (added_cnts)

def runTest(featmat_train, outcome_train_lbl, featmat_test, outcome_test_lbl, sel, paramsDict, bestmodelnum):
    print ("Running Test for #{0} ({1})".format(TEST_PERSON_NUM, TEST_PERSON_DEVICE_ID))
    X_train_allfg = featmat_train.values
    Y_train = outcome_train_lbl.values
    #     Y_train = Y_train.reshape(Y_train.size, 1)# does this help?
    featnames_allfg = featmat_train.columns
    X_test_allfg = featmat_test.values
    Y_test = outcome_test_lbl.values
    Y_true = Y_test[0]
    sel_featnames_per_fg = {}
    sel_featnames_list_ordered = []
    sel_X_train = []
    sel_X_test = []
    countNumSel = 0
    fgi = 0
    for s in suffix_list:
        fgi = fgi + 1
    #    print fgi,
        suffix_list_str = ",".join(s)
        fgidxs = fgColIdxs[suffix_list_str]
        X_train = X_train_allfg[:, fgidxs]
        X_test = X_test_allfg[:, fgidxs]
        featnames_fg = featnames_allfg[fgidxs]
        # continue if empty
        if X_train.shape[1] == 0:
            continue
        ## scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # variance thresholding
        vartransform = VarianceThreshold()
        X_train = vartransform.fit_transform(X_train)
        X_test = vartransform.transform(X_test)
        varthres_support = vartransform.get_support()
        featnames_fg = featnames_fg[varthres_support]
        ## feature selection
        if sel == "rlog":
            #print (X_train.shape)
            randomized_rlog = RandomizedLogisticRegression(**paramsDict)
            X_train = randomized_rlog.fit_transform(X_train, Y_train)
            X_test = randomized_rlog.transform(X_test)
            chosen_col_idxs = randomized_rlog.get_support()
            #print (len(featnames_fg))
            #print (len(chosen_col_idxs))

            if len(chosen_col_idxs) > 0:
                featnames_fg_chosen = list(featnames_fg[chosen_col_idxs])
                sel_featnames_per_fg[suffix_list_str] = featnames_fg_chosen
                sel_featnames_list_ordered = sel_featnames_list_ordered + featnames_fg_chosen
                sel_X_train.append(X_train)
                sel_X_test.append(X_test)
                countNumSel = countNumSel + len(featnames_fg_chosen)
        else:
            raise ("Unrecognized sel (feature selection algorithm)")
    ## feature selection:  sel{sel{fg1}.....sel{fg45}}
    X_train_concat = np.hstack(sel_X_train)
    X_test_concat = np.hstack(sel_X_test)
    print ("\nSum of number of features selected from all fgs = {0}".format(countNumSel))
    print ("Concatenated X_train has {0} features".format(X_train_concat.shape[1]))
    print ("Concatenated X_test has {0} features".format(X_test_concat.shape[1]))
    if sel == "rlog":
        randomized_rlog = RandomizedLogisticRegression(**paramsDict)
        X_train_concat = randomized_rlog.fit_transform(X_train_concat, Y_train)
        X_test_concat = randomized_rlog.transform(X_test_concat)
        chosen_col_idxs = randomized_rlog.get_support()
        sel_featnames_list_ordered = np.array(sel_featnames_list_ordered)
        chosen_col_idxs = np.array(chosen_col_idxs)
        chosen_cols_final = sel_featnames_list_ordered[chosen_col_idxs]
    else:
        raise ("Unrecognized sel (feature selection algorithm)")
    print ("Final number of features in model = {0}".format(X_train_concat.shape[1]))
    # GBCT
    if modelname == "GBC":
        clf = GradientBoostingClassifier(random_state=0)
    elif modelname == "LOGR":
        clf = LogisticRegression(random_state=0, C=paramsDict["C"], tol=1e-3, penalty="l1", n_jobs=paramsDict["n_jobs"], intercept_scaling=1, class_weight="balanced")
    else:
        raise ("Unrecognized model name")
    clf.fit(X_train_concat, Y_train)
    pred = clf.predict(X_test_concat)
    pred_proba = clf.predict_proba(X_test_concat)
    Y_pred = pred[0]
    Y_pred_proba = pred_proba[0][1]
    ## Logging test_person_test.csv - outputs 1 line only
    ## did, sel, selParams, Y_pred, Y_pred_proba, Y_true, chosen_cols_final, suffix_list_str : sel_featnames_per_fg[suffix_list_str] in separate columns
    chosen_cols_final_str = ",".join(chosen_cols_final)
    paramsDict_str = ','.join("%s:%r" % (key, val) for (key, val) in paramsDict.iteritems())
    fgIdxs_str = ','.join("%s:%r" % (key, val) for (key, val) in fgIdxs.iteritems())
    cnts_per_lbl_dict = getValueCounts(outcome_train_lbl, outcome_test_lbl)
    cnts_per_lbl_str = ','.join("%s:%r" % (key, val) for (key, val) in cnts_per_lbl_dict.iteritems())
    dfout = pd.DataFrame({"did": [TEST_PERSON_DEVICE_ID], "cnts_per_lbl": [cnts_per_lbl_str], "sel": [sel], "selParams": [paramsDict_str], "Y_pred": [Y_pred], "Y_pred_proba": [Y_pred_proba], "Y_true": [Y_true], "fgIdxs": [fgIdxs_str], "sel_final": [chosen_cols_final_str]})
    dfout = dfout.set_index("did")
    cols = ["cnts_per_lbl", "sel", "selParams", "Y_pred", "Y_pred_proba", "Y_true", "fgIdxs", "sel_final"]
    for s in suffix_list:
        suffix_list_str = ",".join(s)
        if suffix_list_str in sel_featnames_per_fg:
            sel_feats_fg_str = ",".join(sel_featnames_per_fg[suffix_list_str])
        else:
            sel_feats_fg_str = ""
        dfcol = pd.DataFrame({"did": [TEST_PERSON_DEVICE_ID], "sel_{0}".format(suffix_list_str): [sel_feats_fg_str]})
        dfcol = dfcol.set_index("did")
        dfout = pd.concat([dfout, dfcol], axis=1)
        cols.append("sel_{0}".format(suffix_list_str))
    dfout.to_csv(folderpath + "{0}_test_model{1}.csv".format(TEST_PERSON_DEVICE_ID, bestmodelnum), columns=cols, header=True)
    print ("{0} minutes elapsed since start of program ".format((time.time() - STARTTIME) / 60.0))
    return (Y_pred, Y_pred_proba)


def checkFeatMat():
    print ("MAIN: Final featmat_all {0} subjects, {1} features\n".format(len(featmat_all), len(featmat_all.columns)))
    print ("fgIdxs")
    print (len(fgIdxs.keys()))
    print ("fgColNames")
    print (len(fgColNames.keys()))
    vallensum = 0
    for k in fgColNames.keys():
        vallen = len(fgColNames[k])
        vallensum = vallensum + vallen
    print (vallensum)
    print ("fgColIdxs")
    print (len(fgColIdxs.keys()))
    vallensum = 0
    for k in fgColIdxs.keys():
        vallen = len(fgColNames[k])
        vallensum = vallensum + vallen
    print (vallensum)


# def main():
#     global n_jobs
#     n_jobs = 1
#     global STARTTIME
#     STARTTIME = time.time()
# #     checkFeatMat()
#     ## Code runs only for test person
#     global TEST_PERSON_NUM
#     global TEST_PERSON_DEVICE_ID
# #     TEST_PERSON_NUM = int(sys.argv[1])
#     TEST_PERSON_NUM = int(0)
#     print ("test num arg {0}".format(TEST_PERSON_NUM))
#     DEVICE_IDS_ALL = postbdi.index
#     TEST_PERSON_DEVICE_ID = DEVICE_IDS_ALL[TEST_PERSON_NUM]
#     ## Get train and test featmats
#     featmat_train, postbdi_train_lbl, featmat_test, postbdi_test_lbl, exists = getTrainTestSplit(featmat_all)
#     print ("Train has {0} subjects and {1} features".format(len(featmat_train), len(featmat_train.columns)))
#     print ("Test has {0} subjects and {1} features".format(len(featmat_test), len(featmat_test.columns)))
#     if exists == 1:
# #         sel, selParamsDict, bestmodelnum = runCv(featmat_train, postbdi_train_lbl)
#         sel = "rlasso"
#         selParamsDict = {"alpha": 'aic', "scaling": 0.5, "sample_fraction": 0.75, "n_resampling": 200, "selection_threshold": 0.25, "fit_intercept": True, "normalize": False, "random_state": 0, "n_jobs": n_jobs}
#         bestmodelnum = 1
#         Y_pred, Y_pred_proba = runTest(featmat_train, postbdi_train_lbl, featmat_test, postbdi_test_lbl, sel, selParamsDict, bestmodelnum)

def main():
    global STARTTIME
    STARTTIME = time.time()
    #     checkFeatMat()
    ## Code runs only for test person
    global TEST_PERSON_NUM
    global TEST_PERSON_DEVICE_ID
    for ti in range(0, 138):
        TEST_PERSON_NUM = int(ti)
        print ("test num arg {0}".format(TEST_PERSON_NUM))
        DEVICE_IDS_ALL = postbdi.index
        TEST_PERSON_DEVICE_ID = DEVICE_IDS_ALL[TEST_PERSON_NUM]
        ## Get train and test featmats
        featmat_train, outcome_train_lbl, featmat_test, outcome_test_lbl, exists = getTrainTestSplit(featmat_all)
        if exists == 1:
            print ("Train has {0} subjects and {1} features".format(len(featmat_train), len(featmat_train.columns)))
            print ("Test has {0} subjects and {1} features".format(len(featmat_test), len(featmat_test.columns)))
            sel = "rlog"
            print (sel)
            print (selParamsDict)
            bestmodelnum = 1
            Y_pred, Y_pred_proba = runTest(featmat_train, outcome_train_lbl, featmat_test, outcome_test_lbl, sel, selParamsDict, bestmodelnum)

# call main
if __name__ == '__main__':
    main()