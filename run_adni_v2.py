'''
run_uni_h2o.py v1.0
Rafi Haque rafihaq90@gmail.com
--------------------
Description:
--------------------
Instructions:
'''

from ADNIData import ADNIData
from Subtyper import Subtyper
from statsmodels.stats.multitest import fdrcorrection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import f_oneway, ttest_ind
from Plotter import Plotter
from ML import ML
from datasets import load_data
from bayes_opt import BayesianOptimization
import xgboost as xgb
import shap
import seaborn as sns
import scanpy as sc
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import anndata as ad
from os.path import join
from itertools import compress
import time
import pickle
import imp
import csv
import argparse
import os
import pdb

import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 100)

parser = argparse.ArgumentParser(
    description='compute univariate auc using automl framework')
parser.add_argument('--data_path', nargs='?', default='FNIH_CSF_ALL')
parser.add_argument('--anal', nargs='?', default='STATS-ConvsAD-Diversity')
parser.add_argument('--load', nargs='?', default='emory')
parser.add_argument('--dirs_file', nargs='?', default='dirs.py')
args = parser.parse_args()

# paths
prot_path = 'ADNI/JL_ADNIMERGE_64peptides_APOEpeptides_rm4SDoutliers_LastVisit.csv'
prot_path = 'ADNI/64peptides_pheno_lastVisit_rmOutlier.csv'
prot_path = 'ADNI/Final_peptides_after_QC.csv'
prot_path = 'ADNI/Final_peptides_after_QC_regress_ageSexRaceBatch.csv'

merge_path = 'ADNI/Final_Clinical_Data.csv'

obj_path = 'ADNI/ADNIDATA.obj'
obj_path1 = 'ADNI/ZADNIDATA.obj'
#obj_path  = 'ADNI/ADNIDATAZ.obj'
rawfig_path = 'ADNI/rawfig/'

# filters
or_columns = ['STD_DIAG']*2
or_conds = ['Control', 'AD']
alpha = 1e-15


# predictors


# #cog_bl = ['CDRSB_bl','ADAS13_bl','MMSE_bl','MOCA_bl','RAVLT_immediate_bl','FAQ_bl','mPACCtrailsB_bl']
# cog_bl = ['CDRSB_bl','MOCA_bl']
# mri_bl =
# pet_bl = ['FDG_bl']

# decliners
cog = ['CDRSB', 'MOCA']
mri = ['Ventricles', 'Hippocampus', 'WholeBrain',
       'Entorhinal', 'Fusiform', 'MidTemp']

# filters
or_cols = ['STD_DIAG']*2
or_conds = ['Control', 'AD']
alpha = .01
split = 5
repeats = 1
seed = 1
models = ['LGR', 'XGBC']
models = ['LGRCV']
nperm = 100
clip = 3


def main():

    # get features of interest
    preds = {
        'age_bl': ['AGE'],
        'sex_bl': ['PTGENDER'],
        'bat_bl': ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'],
        'gen_bl': ['APOE_geno1', 'APOE_geno2'],
        'amy_bl': ['ABETA_bl'],
        'av45_bl': ['AV45'],
        'ptau_bl': ['PTAU_bl'],
        'rat_bl': ['PTAU_bl/ABETA_bl'],
        'tau_bl': ['TAU_bl'],
        'fdg_bl': ['FDG_bl'],
        'hipp_bl': ['Hippocampus_bl'],
        'moca_bl': ['MOCA_bl'],
        'cdrsb_bl': ['CDRSB_bl'],
        'cli_bl': ['AGE', 'PTGENDER'],
        'amy_all_bl': ['ABETA_bl', 'AV45'],
        'tau_all_bl': ['PTAU_bl', 'TAU_bl'],
        'deg_all_bl': ['Hippocampus_bl', 'FDG_bl'],
        'cog_all_bl': ['CDRSB_bl', 'MOCA_bl'],
        'tra_bl': ['DX_bl_new', 'change_CN_MCIdementia', 'DX_last'],
        'mol_bl': ['ABETA_bl', 'TAU_bl', 'PTAU_bl'],
    }

    merge_bls = preds['amy_all_bl'] + preds['tau_all_bl'] + \
        preds['cog_all_bl'] + preds['cli_bl'] + \
        preds['deg_all_bl'] + preds['gen_bl']

    # save adni data
    #data = ADNIData(merge_path = merge_path, feats_path=prot_path,trajs=cog+mri,bls=merge_bls,clis=preds['cli_bl'],tras=preds['tra_bl']).get_base().get_trajs().save(obj_path)
    # load adni data
    data = ADNIData(merge_path=merge_path, feats_path=prot_path,
                    trajs=cog+mri, bls=merge_bls).load(obj_path)

    # update dictionary with feature combinations features
    preds.update({'cli_bl+bat_bl': preds['cli_bl']+preds['bat_bl']})
    preds.update({'prot_bl': data.new_prots.tolist()})
    preds.update({'prot_bl+tau_all_bl': preds['prot_bl']+preds['tau_all_bl']})
    preds.update({'prot_bl+deg_all_bl': preds['prot_bl']+preds['deg_all_bl']})
    preds.update({'prot_bl+amy_all_bl': preds['prot_bl']+preds['amy_all_bl']})
    preds.update({'prot_bl+cog_all_bl': preds['prot_bl']+preds['cog_all_bl']})
    preds.update({'prot_bl+fdg_bl': preds['prot_bl']+preds['fdg_bl']})
    preds.update({'prot_bl+hipp_bl': preds['prot_bl']+preds['hipp_bl']})
    preds.update({'prot_bl+tau_bl': preds['prot_bl']+preds['tau_bl']})
    preds.update({'prot_bl+ptau_bl': preds['prot_bl']+preds['ptau_bl']})
    preds.update({'prot_bl+amy_bl': preds['prot_bl']+preds['amy_bl']})
    preds.update({'prot_bl+av45_bl': preds['prot_bl']+preds['av45_bl']})
    preds.update({'prot_bl+cdrsb_bl': preds['prot_bl']+preds['cdrsb_bl']})
    preds.update({'prot_bl+moca_bl': preds['prot_bl']+preds['moca_bl']})
    preds.update({'prot_bl+mol_bl': preds['prot_bl']+preds['mol_bl']})
    preds.update({'all_bl': preds['amy_all_bl'] +
                 preds['tau_all_bl']+preds['deg_all_bl']+preds['cog_all_bl']})

    # ###########
    # # TABLE 1 #
    # ###########
    # # loop through each dx
    # dxs = np.unique(data.feats['DX_bl_new'])
    # for dx in dxs:
    #     bl = data.feats['DX_bl_new']==dx
    #     print('%s:%d'% (dx,np.sum(bl)))

    # # progressors
    # progs = np.unique(data.feats['change_CN_MCIdementia'])
    # for prog in progs:
    #     bl = data.feats['change_CN_MCIdementia']==prog
    #     print('%s:%d'%(prog,np.sum(bl)))

    # # average age
    # mcols = ['AGE']
    # for col in mcols:
    #     print('%s: Mean: %.02f STD: %.02f' % (col,np.mean(data.feats[col]),np.std(data.feats[col])))

    # # dx by percentage, ATN, APOE,etc
    # orcols = ['DX_bl_new','ATN','APOE_geno','change_CN_MCIdementia','PTGENDER']
    # for col in orcols:
    #     ents = np.unique(data.feats[col])
    #     for ent in ents:
    #         y = data.feats[col]
    #         print('%s: %.02f %.02f' % (ent,sum(y==ent)/len(y),sum(y==ent)))

    # #######################
    # # FIGURE BASELINE DX  #
    # #######################

    # orcols = ['DX_bl_new','ATN','ATN','ATN','T','P','A','change_CN_MCIdementia']
    # orcond = [['CN','Dementia'],['A-T-','A-T+'],['A-T-','A+T-'],['A-T-','A+T+'],['T-','T+'],['P-','P+'],['A-','A+'],['NonProg','Prog']]
    # names =  ['baseDEM','baseA-T+','baseA+T-','baseA+T+','baseT','baseP','baseA','baseProg']
    # models = ['LGRCV','LGRCV','LGRCV','LGRCV','LGRCV','LGRCV','LGRCV','LGRCV']
    # feats =  ['PROT','GENO','CLI','MOL','MOL + PROT','AMY','TAU','PTAU','RAT']
    # score =  pd.DataFrame(index=preds['prot_bl'],columns=names)
    # score_csv = pd.DataFrame(index=feats,columns=['P','P-PROT','P-PROT-MOL'])
    # score_diff = pd.DataFrame(index=preds['prot_bl'],columns=['A-T+','A+T-','A+T+'])
    # score_auc = pd.DataFrame(index=preds['prot_bl'],columns=['A-T+','A+T-','A+T+'])
    # score_p = pd.DataFrame(index=preds['prot_bl'],columns=['A-T+','A+T-','A+T+'])

    # # loop through each comparision
    # for col,cond,name,model in zip(orcols,orcond,names,models):
    #     print('CLASS: %s' % col)
    #     print('MODEL: %s' % model)

    #     # load adni data and labels
    #     data = ADNIData(merge_path = merge_path, feats_path=prot_path,trajs=cog+mri,bls=merge_bls).load(obj_path)
    #     data.feats['change_CN_MCIdementia'][data.feats['change_CN_MCIdementia']=='MCI to MCI']='NonProg'
    #     data.feats['change_CN_MCIdementia'][data.feats['change_CN_MCIdementia']=='MCI to Dementia']='Prog'
    #     data.feats['change_CN_MCIdementia'][data.feats['change_CN_MCIdementia']=='MCI to CN']='NonProg'

    #     data.or_filter([col]*2,cond)
    #     y = data.get_y(col,cond,[0,1])

    #     # run model for each protein
    #     ml = ML(x=data.feats[preds['prot_bl']],y=y).norm_nan().diff(0,1).impute().init('LGR').select(1).train_test_uni()

    #     score_diff[cond[1]] = ml.diffex['log2(1/0)']
    #     score_auc[cond[1]] = ml.score_uni['AUC']
    #     score_p[cond[1]] = ml.diffex['PFDR']

    #     # save these results to a csv file
    #     rn = 'log2(%s/%s)' % (cond[1],cond[0])
    #     new = ml.diffex.rename(columns={"log2(1/0)":rn})
    #     new[[rn,'P','PFDR','AUC']].sort_values(by='P').to_csv(join(rawfig_path,'TableS_%s.csv' % name))

    #     # get number of significant peptides and proteins
    #     pbool = (ml.diffex['PFDR']) & (ml.diffex['log2(1/0)']>0)
    #     nbool = (ml.diffex['PFDR']) & (ml.diffex['log2(1/0)']<0)
    #     pm = [x.split("|", 1)[0] for x in new.index[pbool].tolist()]
    #     nm = [x.split("|", 1)[0] for x in new.index[nbool].tolist()]

    #     print('+ Peptides:%d' % sum(pbool))
    #     print('- Peptides:%d' % sum(nbool))
    #     print('+ Prots:%d' % len(np.unique(pm)))
    #     print('- Prots:%d' % len(np.unique(nm)))
    #     print(new.sort_values(by='AUC'))

    #     ml1 = ML(x=data.feats[preds['prot_bl']],y=y).impute().norm_clip().init(model).train_test(score='AUC').select(1).train_test_uni()
    #     if name == 'baseA-T+':
    #         Plotter(xlim=[0,1],ylim=[0,1],newfig=False).roc([y],[ml1.yprob],join(rawfig_path,'atn_roc.pdf'),cols=['cornflowerblue'])
    #     elif name == 'baseA+T+':
    #         Plotter(xlim=[0,1],ylim=[0,1],newfig=False).roc([y],[ml1.yprob],join(rawfig_path,'atn_roc.pdf'),cols=['coral'])
    #     elif name == 'baseA+T-':
    #         Plotter(xlim=[0,1],ylim=[0,1],newfig=False).roc([y],[ml1.yprob],join(rawfig_path,'atn_roc.pdf'),cols=['grey'])

    #     # run model with all proteins
    #     print('PROT:')
    #     ml1 = ML(x=data.feats[preds['prot_bl']],y=y).impute().norm_clip().init(model).train_test(score='AUC').select(1).train_test_uni()

    #     score.loc[:,name]= ml1.diffex['AUC']
    #     score_csv.loc['PROT','%s AUC' % name] = '%.2f' % ml1.score
    #     print(ml1.score)

    #     # run apoe genotype
    #     print('GENO:')
    #     ml2 = ML(x=data.feats[preds['gen_bl']],y=y).impute().init('LGR').train_test(score='AUC')
    #     print(ml2.score)
    #     score_csv.loc['GENO','%s AUC' % name] = '%.2f' % ml2.score

    #     # run clinical co-variates and batch
    #     print('CLI + BATCH:')
    #     ml3 = ML(x=data.feats[preds['cli_bl+bat_bl']],y=y).impute().norm_clip().init('LGR').train_test(score='AUC')
    #     print(ml3.score)
    #     score_csv.loc['CLI','%s AUC' % name] = '%.2f' % ml3.score
    #     score_csv.loc['CLI','%s P-PROT' % name] = '%.2e' %  ml1.compare(ml3.ypred)

    #     # run ad biomarkers
    #     print('MOL:')
    #     ml4 = ML(x=data.feats[preds['mol_bl']],y=y).impute().norm_clip().init(model).train_test(score='AUC')
    #     print(ml4.score)
    #     score_csv.loc['MOL','%s AUC' % name] = '%.2f' % ml4.score
    #     score_csv.loc['MOL','%s P-PROT' % name] = '%.2e' %  ml1.compare(ml4.ypred)

    #     # run ad biomarkers and proteins
    #     print('MOL + PROT:')
    #     ml5 = ML(x=data.feats[preds['prot_bl+mol_bl']],y=y).impute().norm_clip().init(model).train_test(score='AUC')
    #     print(ml5.score)
    #     score_csv.loc['MOL + PROT','%s AUC' % name] = '%.2f' % ml5.score
    #     score_csv.loc['MOL + PROT','%s P-PROT' % name] = '%.2e' %  ml1.compare(ml5.ypred)
    #     score_csv.loc['MOL','%s P-PROT-MOL' % name] = '%.2e '% ml5.compare(ml4.ypred)

    #     # # run amyloid
    #     # print('AMY:')
    #     # mla = ML(x=data.feats[preds['amy_bl']],y=y).impute().norm_clip().init(model).train_test(score='AUC')
    #     # print(mla.score)
    #     # score_csv.loc['AMY','%s AUC' % name] = '%.2f' % mla.score

    #     # # run tau
    #     # print('TAU:')
    #     # mlt = ML(x=data.feats[preds['tau_bl']],y=y).impute().norm_clip().init(model).train_test(score='AUC')
    #     # print(mlt.score)
    #     # score_csv.loc['TAU','%s AUC' % name] = '%.2f' % mlt.score

    #     # # run ptau
    #     # print('PTAU:')
    #     # mlpt = ML(x=data.feats[preds['ptau_bl']],y=y).impute().norm_clip().init(model).train_test(score='AUC')
    #     # print(mlpt.score)
    #     # score_csv.loc['PTAU','%s AUC' % name] = '%.2f' % mlpt.score

    #     # # run ratio
    #     # print('RAT:')
    #     # mlr = ML(x=data.feats[preds['rat_bl']],y=y).impute().norm_clip().init(model).train_test(score='AUC')
    #     # print(mlr.score)
    #     # score_csv.loc['RAT','%s AUC' % name] = '%.2f' % mlr.score

    #     # # if av45
    #     # if col=='AV45':
    #     #     print('AMY:')
    #     #     mlamy = ML(x=data.feats[preds['amy_bl']],y=y).impute().norm_clip().init('LGR').train_test(score='AUC')
    #     #     print(mlamy.score)

    #     #     ml5 = ML(x=data.feats[preds['prot_bl+amy_bl']],y=y).impute().norm_clip().init(model).train_test(score='AUC')
    #     #     print(ml5.score)

    #     # # if tau
    #     # if col=='T':
    #     #     data2 = ADNIData(merge_path = merge_path, feats_path=prot_path,trajs=cog+mri,bls=merge_bls).load(obj_path).or_filter([col]*2,cond)
    #     #     y2 = data.get_y('P',['P+','P-'],[0,1])
    #     #     mlp = ML(x=data2.feats[preds['prot_bl']],y=y2).impute().norm_clip().init(model).train_test(score='AUC')
    #     #     print('PTAU:')
    #     #     print(mlp.score)

    #     print('STATS:')
    #     print('PROT to MOL %.2e' % ml1.compare(ml4.ypred))
    #     print('PROT to AGE: %.2e' % ml1.compare(ml3.ypred))
    #     print('PROT to Genotype: %.2e' % ml1.compare(ml2.ypred))
    #     print('PROT+MOL to MOL: %.2e '% ml5.compare(ml4.ypred))

    #     # plot roc curve and diffex
    #     if (name == 'baseA-T+') | (name == 'baseA+T+') | (name == 'baseA+T-'):
    #         ml6 = ML(x=data.feats[preds['prot_bl']],y=y).diff(0,1).impute().init('LGR').train_test_uni()
    #         ml6.diffex.sort_values(by='AUC',ascending=False).to_csv(join(rawfig_path,'diffex_%s.csv' % name))
    #         Plotter(xlim=[0,1],ylim=[0,1]).roc([y]*5,[ml1.yprob],join(rawfig_path,'roc_%s.pdf' % name),cols=['cornflowerblue','lightgrey','#f1b7f0','lightgreen'])
    #     elif (name == 'baseT'):
    #         Plotter(xlim=[0,1],ylim=[0,1]).roc([y,y2],[ml1.yprob,mlp.yprob],join(rawfig_path,'roc_%s.pdf' % name),cols=['cornflowerblue','#f1b7f0','lightgreen'])
    #         pdb.set_trace()
    #     elif (name =='baseAV'):
    #         Plotter(xlim=[0,1],ylim=[0,1]).roc([y]*5,[ml1.yprob,mlamy.yprob,ml5.yprob],join(rawfig_path,'roc_%s.pdf' % name),cols=['cornflowerblue','#f1b7f0','lightgreen'])
    #     else:

    #         Plotter(xlim=[0,1],ylim=[0,1]).roc([y]*5,[ml1.yprob,ml4.yprob,ml5.yprob],join(rawfig_path,'roc_%s.pdf' % name),cols=['cornflowerblue','#f1b7f0','lightgreen'])
    #         pdb.set_trace()
    #         # control vs ad
    #         ml6 = ML(x=data.feats[preds['prot_bl+mol_bl']],y=y).diff(0,1).impute().init('LGR').train_test_uni()

    #         # get unique proteins
    #         uni_prots= np.array([x.split('|')[0] for x in ml6.diffex.index])
    #         score_auc = pd.DataFrame(index=np.unique(uni_prots),columns = ml6.diffex.columns)

    #         for prot in np.unique(uni_prots):
    #             score_pep = ml6.diffex[uni_prots==prot]
    #             score_auc.loc[prot]=score_pep.iloc[np.argmax(score_pep['AUC'])]

    #         # save out table
    #         rn = 'log2(%s/%s)' % (cond[1],cond[0])
    #         new = score_auc.rename(columns={"log2(1/0)":rn})
    #         new[[rn,'P','PFDR','AUC']].sort_values(by='P').to_csv(join(rawfig_path,'TableS_%s.csv' % name))

    #         score_auc['AUC'][~score_auc['PFDR']] =np.nan
    #         pl = Plotter(xlim=[-1,1]).diffex(score_auc,join(rawfig_path,'diffex_%s.pdf' % name),x='log2(1/0)',s=100,cmap='Blues')
    #         Plotter(xlim=[0,1],ylim=[0,1]).roc([y]*5,[ml1.yprob,ml4.yprob,ml5.yprob],join(rawfig_path,'roc_%s.pdf' % name),cols=['cornflowerblue','#f1b7f0','lightgreen'])

    #     #print(ml6.diffex.sort_values(by='AUC',ascending=False))
    #     pdb.set_trace()
    # pdb.set_trace()
    # AT = ['A+T+','A-T+','A+T-']
    # uni_prots= np.array([x.split('|')[0] for x in score_p.index])
    # score_diffp = pd.DataFrame(index=np.unique(uni_prots),columns = score_diff.columns)
    # score_pp = pd.DataFrame(index=np.unique(uni_prots),columns = score_diff.columns)
    # score_aucp = pd.DataFrame(index=np.unique(uni_prots),columns = score_diff.columns)
    # for prot in np.unique(uni_prots):
    #     score_diffp.loc[prot]= np.max(score_diff[uni_prots==prot])
    #     score_pp.loc[prot]= np.sum(score_p[uni_prots==prot])!=0
    #     score_aucp.loc[prot]= np.max(score_auc[uni_prots==prot])

    # pdb.set_trace()
    # score_diffAT = score_diffp[AT][score_pp[AT]]
    # score_diffAUC = score_aucp[AT][score_pp[AT]]
    # sig_prots = np.sum(score_pp[AT],axis=1)!=0
    # sig_diff =score_diffAT[sig_prots].astype('float64').sort_values(by='A+T+',ascending=False)
    # sig_AUC =score_diffAUC[sig_prots].astype('float64').rename({'A+T+':'A+T+ vs A-T-','A+T-':'A+T- vs A-T-','A-T+':'A-T+ vs A-T-'},axis=1)
    # sig_AUC['A+T+'].sort_values(by='A+T+',ascending=False).to_csv(join(rawfig_path,'TableS_ATN.csv'))

    # Plotter(size=[8,10]).heat_sns(sig_diff,xticklabels=sig_diff.columns,path=join(rawfig_path,'heat_AT.pdf'),annot=True,cmap='RdBu_r',linewidths=0.1,linecolor='black',vmin=-1,vmax=1)

    # # ###################
    # # # FIGURE BOX ATN  #
    # # ###################
    # # prots = ['SMOC1|0','1433Z|0','KPYM|0','VGF|0']
    # # for prot in prots:sca
    # #     print(prot)

    # #     Plotter().box_sns(data.feats,'ATN',prot,join(rawfig_path,'%s_box_ATN.pdf' % prot),palette=['cornflowerblue','cornflowerblue','cornflowerblue','cornflowerblue'],order=['A-T-','A+T-','A-T+','A+T+'],alpha=0.5)
    # # pdb.set_trace()

    #######################
    # FIGURE CORRELATIONS #
    #######################

    trajs = ['AV45', 'ABETA_bl', 'PTAU_bl', 'TAU_bl', 'Hippocampus_bl',
             'FDG_bl', 'MOCA_bl', 'CDRSB_bl']  # 'MOCA','Hippocampus','CDRSB']
    # trajs = ['FDG_bl','AV45','ABETA_bl','PTAU_bl','TAU_bl','MOCA_bl','CDRSB_bl']
    #trajs = ['Hippocampus_bl']
    score_uni = pd.DataFrame(index=preds['prot_bl+mol_bl'], columns=trajs)
    score_uni2 = pd.DataFrame(index=preds['prot_bl+mol_bl'], columns=trajs)
    sig_uni = pd.DataFrame(index=preds['prot_bl+mol_bl'], columns=trajs)
    score_p = pd.DataFrame(index=preds['prot_bl+mol_bl'], columns=trajs)
    cols = ['cornflowerblue', '#91ffff', 'lightgreen', '#ffffc2', '#f1b7f0',
            '#f1b7f0', 'lightgrey', 'lightgrey', 'lightgrey', 'lightgrey', 'cornflowerblue']
    cols = ['cornflowerblue', '#f1b7f0', 'cornflowerblue', '#f1b7f0', 'cornflowerblue',
            '#f1b7f0', 'cornflowerblue', '#f1b7f0', 'lightgrey', 'lightgrey', 'cornflowerblue']

    for traj, col in zip(trajs, cols):
        # feature combinations for each trait
        if traj == 'AV45':
            keys = ['cli_bl+bat_bl', 'amy_bl', 'prot_bl', 'prot_bl+amy_bl']
        elif traj == 'ABETA_bl':
            keys = ['cli_bl+bat_bl', 'av45_bl', 'prot_bl', 'prot_bl+av45_bl']
        elif traj == 'TAU_bl':
            keys = ['cli_bl+bat_bl', 'ptau_bl', 'prot_bl', 'prot_bl+ptau_bl']
        elif traj == 'PTAU_bl':
            keys = ['cli_bl+bat_bl', 'tau_bl', 'prot_bl', 'prot_bl+tau_bl']
        else:
            keys = ['cli_bl+bat_bl', 'mol_bl', 'prot_bl', 'prot_bl+mol_bl']

        # store R and p value for each trait across different feature combos
        score = pd.DataFrame(index=trajs, columns=keys)
        ps = pd.DataFrame(index=trajs, columns=keys)

        # filter by z-score, max visit time, and get demographics if trajectory else just filter by outlier
        y = data.feats[traj].astype('float64')
        zy = y.copy()
        zy = ((zy-zy.mean())/zy.std())

        if (traj == 'Hippocampus') | (traj == 'MOCA') | (traj == 'CDRSB'):
            filt = (~np.isnan(y)) & (np.abs(zy) < 4) & (
                data.feats[traj+'_lv'] > 3)  # & (data.feats['ATN']=='A+T+')
        else:
            filt = (~np.isnan(y)) & (np.abs(zy) < 4)
        ypreds = pd.DataFrame(index=y[filt].index, columns=keys)

        # loop through each trait
        for count, key in enumerate(keys):

            # get features and labels
            print(traj + ' ' + key)

            x_filt = data.feats[preds[key]][filt]
            y_filt = y[filt]

            # run elastic net and use R for performance
            mltraj = ML(x=x_filt, y=y_filt.to_numpy()).impute(
            ).norm_clip(5).init('ENCV').train_test(score='R')
            score.loc[traj, key] = mltraj.score
            ps.loc[traj, key] = mltraj.p

            print(' R: %.02f P: %.2e \n' % (mltraj.score, mltraj.p))
            ypreds.loc[:, key] = mltraj.ypred

            # run for individual peptides
            if key == 'tau_bl':
                Plotter().scatter([mltraj.ypred], [mltraj.y], join(
                    rawfig_path, '%s_%s_reg_dx.pdf' % (traj, key)), cols=[col], alpha=1, s=100)

            if key == 'prot_bl':
                ml = ML(x=data.feats[preds['prot_bl']][filt], y=y_filt.to_numpy()).impute(
                ).norm_clip(5).init('LR').train_test_uni(score='R', alpha=0.01)
                score_uni.loc[:, traj] = data.feats[preds['prot_bl']].corrwith(
                    y)  # ml.score_uni['R']
                score_uni2.loc[:, traj] = ml.score_uni['R'].astype('float64')
                score_p.loc[:, traj] = ml.score_uni['P']
                sig_uni.loc[:, traj] = ml.score_uni['FDR']
                ypreds.loc[:, key] = mltraj.ypred

                # get top 10
                sort_prots = score_uni2.sort_values(
                    by=traj, ascending=False).index
                top10 = []
                topR = []
                for prot in sort_prots:
                    if len(top10) < 10:
                        if prot.split('|')[0] not in top10:
                            top10.append(prot.split('|')[0])
                            topR.append(score_uni2[traj][prot])

                print(top10)
                print(topR)

                if (traj == 'PTAU_bl') | (traj == 'TAU_bl'):
                    Plotter(xlim=[0, 1]).bar([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], topR, join(
                        rawfig_path, '%s_top10.pdf' % (traj)), [col], alpha=1, orient='h')
                else:
                    Plotter(xlim=[0, 0.5]).bar([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], topR, join(
                        rawfig_path, '%s_top10.pdf' % (traj)), [col], alpha=1, orient='h')

                # plot proteomimc regressino
                if key == 'prot_bl':
                    Plotter().scatter([mltraj.ypred], [mltraj.y], join(
                        rawfig_path, '%s_%s_reg_dx.pdf' % (traj, key)), cols=[col], alpha=1, s=100)

            # plot scores of all regressions
            scores = score.loc[traj, keys].to_numpy()
            if (traj == 'MOCA_bl') | (traj == 'CDRSB_bl') | (traj == 'MOCA') | (traj == 'CDRSB') | (traj == 'Hippocampus'):
                Plotter().bar([0, 1, 2, 3], np.abs(scores), join(
                    rawfig_path, '%s_reg_atn.pdf' % (traj)), [col], alpha=1, orient='h')

            else:
                Plotter().bar([0, 1, 2, 3], np.abs(scores), join(
                    rawfig_path, '%s_reg.pdf' % (traj)), [col], alpha=1)

        # compare different models
        ypred0 = ypreds[keys[0]].to_numpy()
        ypred1 = ypreds[keys[1]].to_numpy()
        ypred2 = ypreds[keys[2]].to_numpy()
        ypred3 = ypreds[keys[3]].to_numpy()

        print(traj + ': ' + keys[1] + '  ' + keys[0])
        gr = mltraj.get_perm_diff(y_filt, ypred1, ypred0, 1000, 'CORR')
        print(' O:%.05f' % gr[0])
        print(' P:%.05f\n' % gr[1])

        print(traj + ': ' + keys[2] + '  ' + keys[0])
        gr = mltraj.get_perm_diff(y_filt, ypred2, ypred0, 1000, 'CORR')
        print(' O:%.05f' % gr[0])
        print(' P:%.05f\n' % gr[1])

        print(traj + ': ' + keys[2] + '  ' + keys[1])
        gr = mltraj.get_perm_diff(y_filt, ypred2, ypred1, 1000, 'CORR')
        print(' O:%.05f' % gr[0])
        print(' P:%.05f\n' % gr[1])

        print(traj + ': ' + keys[3] + '  ' + keys[1])
        gr = mltraj.get_perm_diff(y_filt, ypred3, ypred1, 1000, 'CORR')
        print(' O:%.05f' % gr[0])
        print(' P:%.05f\n' % gr[1])
    pdb.set_trace()

    uni_prots = np.array([x.split('|')[0] for x in score_uni2.index])

    #ps = ['AV45 P', 'ABETA_bl P', 'Hippocampus_bl P', 'FDG_bl P', 'PTAU_bl P', 'TAU_bl P', 'MOCA_bl P', 'CDRSB_bl P']
    ps = [col + ' P' for col in score_uni2.columns]
    score_r = pd.DataFrame(index=np.unique(uni_prots),
                           columns=list(score_uni2.columns)+list(ps))
    score_r2 = pd.DataFrame(index=np.unique(uni_prots),
                            columns=list(score_uni2.columns))
    for prot in np.unique(uni_prots):
        for col in (score_p.columns):
            score_ppep = score_p[uni_prots == prot][col]
            score_rpep = score_uni[uni_prots == prot][col]
            score_r2pep = score_uni2[uni_prots == prot][col]
            sig_rpep = sig_uni[uni_prots == prot][col]

            score_r.loc[prot, col+' P'] = sig_rpep[np.argmin(score_ppep)]
            score_r.loc[prot, col] = score_rpep[np.argmin(score_ppep)]
            score_r2.loc[prot, col] = score_r2pep[np.argmin(score_ppep)]

    # unnormalized scatters
    pdb.set_trace()
    trajs3 = ['AV45', 'ABETA_bl', 'Hippocampus_bl',
              'FDG_bl', 'PTAU_bl', 'TAU_bl', 'MOCA_bl', 'CDRSB_bl']
    ndf = (score_r2-score_r2.min())/(score_r2.max()-score_r2.min())
    sig_uni[ndf < .03] = False

    ndf['Mean'] = ndf.mean(axis=1)
    sig_uni['Mean'] = ndf['Mean']
    score_uni2['Mean'] = ndf['Mean']
    score_uni['Mean'] = ndf['Mean']
    sort_sig = ndf.sort_values(by='Mean', ascending=False)[
        sig_uni.sort_values(by='Mean', ascending=False)[trajs]][trajs]
    raw_sig = score_uni.sort_values(by='Mean', ascending=False)[
        sig_uni.sort_values(by='Mean', ascending=False)[trajs]][trajs]

    score_r2['col'] = 1
    Plotter().scatter_sns(score_r2, 'TAU_bl', 'PTAU_bl', join(rawfig_path,
                                                              'tau_ptau_nscatter.pdf'), hue='col', palette=['cornflowerblue'], alpha=1, s=200)
    Plotter().scatter_sns(score_r2, 'ABETA_bl', 'AV45', join(rawfig_path,
                                                             'amy_av45_nscatter.pdf'), hue='col', palette=['cornflowerblue'], alpha=1, s=200)
    Plotter(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5]).scatter_sns(score_r2, 'Hippocampus_bl', 'FDG_bl',
                                                            join(rawfig_path, 'hipp_fdg_nscatter.pdf'), hue='col', palette=['cornflowerblue'], alpha=1, s=200)
    Plotter(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5]).scatter_sns(score_r2, 'MOCA_bl', 'CDRSB_bl', join(
        rawfig_path, 'moca_cdrsb_nscatter.pdf'), hue='col', palette=['cornflowerblue'], alpha=1, s=200)
    Plotter().scatter_sns(score_r2, 'AV45', 'PTAU_bl', join(rawfig_path,
                                                            'av45_ptau_nscatter.pdf'), hue='col', palette=['cornflowerblue'], alpha=1, s=200)

    # # protein similarity
    sim = ML(score_r2).cluster_corr(
        ndf.astype('float64').corr().loc[trajs3, trajs3])
    Plotter().heat_sns(sim.corr, path=join(rawfig_path,
                                           'similarity.pdf'), annot=True, cmap='Blues')

    pdb.set_trace()
    # score_r.to_csv(join(rawfig_path,'TableS_corr.csv'))

    # pdb.set_trace()

    # pdb.set_trace()
    # unnormalized scatters
    score_uni2['col'] = 1
    Plotter().scatter_sns(np.abs(score_uni2), 'ABETA_bl', 'AV45', join(rawfig_path,
                                                                       'abeta_av45_nscatter.pdf'), hue='col', palette=['white'], alpha=1, s=200)
    Plotter().scatter_sns(np.abs(score_uni2), 'TAU_bl', 'PTAU_bl', join(rawfig_path,
                                                                        'tau_ptau_nscatter.pdf'), hue='col', palette=['#f1b7f0'], alpha=1, s=200)
    Plotter().scatter_sns(np.abs(score_uni2), 'ABETA_bl', 'AV45', join(rawfig_path,
                                                                       'abeta_av45_nscatter.pdf'), hue='col', palette=['white'], alpha=1, s=200)
    Plotter().scatter_sns(np.abs(score_uni2), 'Hippocampus_bl', 'FDG_bl', join(rawfig_path,
                                                                               'hipp_fdg_nscatter.pdf'), hue='col', palette=['lightgreen'], alpha=1, s=200)
    Plotter().scatter_sns(np.abs(score_uni2), 'MOCA_bl', 'CDRSB_bl', join(rawfig_path,
                                                                          'moca_cdrsb_nscatter.pdf'), hue='col', palette=['lightgrey'], alpha=1, s=200)

    # pdb.set_trace()
    # # intersection of AV45 and AB
    # pept = (sig_uni['MOCA_bl']==1) & (sig_uni['CDRSB_bl']==1)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('FDG & Hipp: %d' % len(np.unique(nm)))

    # pept = (sig_uni['CDRSB_bl']==1)# & (sig_uni['ABETA_bl']==0)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('FDG: %d' % len(np.unique(nm)))

    # pept = (sig_uni['MOCA_bl']==1)# (sig_uni['AV45']==0) &
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('Hipp: %d' % len(np.unique(nm)))

    # # intersection of AV45 and AB
    # pept = (sig_uni['AV45']==1) & (sig_uni['ABETA_bl']==1)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('AV45 & AB: %d' % len(np.unique(nm)))

    # pept = (sig_uni['AV45']==1)# & (sig_uni['ABETA_bl']==0)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('AV45: %d' % len(np.unique(nm)))

    # pept = (sig_uni['ABETA_bl']==1)# (sig_uni['AV45']==0) &
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('AB: %d' % len(np.unique(nm)))

    # # intersection of tau and ptau
    # pept = (sig_uni['PTAU_bl']==1) & (sig_uni['TAU_bl']==1)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('TAU & PTAU: %d' % len(np.unique(nm)))

    # pept = (sig_uni['PTAU_bl']==1)# & (sig_uni['TAU_bl']==0)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('PTAU: %d' % len(np.unique(nm)))
    # pept = (sig_uni['TAU_bl']==1)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('TAU: %d' % len(np.unique(nm)))

    # # intersection of ptau and av45
    # pept = (sig_uni['PTAU_bl']==1) & (sig_uni['AV45']==1)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('AV45 & PTAU: %d' % len(np.unique(nm)))
    # pept = (sig_uni['PTAU_bl']==1) & (sig_uni['AV45']==0)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('PTAU: %d' % len(np.unique(nm)))
    # pept = (sig_uni['PTAU_bl']==0) & (sig_uni['AV45']==1)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('AV45: %d' % len(np.unique(nm)))

    # # intersection of tau and av45
    # pept = (sig_uni['PTAU_bl']==1) & (sig_uni['ABETA_bl']==1)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('AB & PTAU: %d' % len(np.unique(nm)))
    # pept = (sig_uni['PTAU_bl']==1) & (sig_uni['ABETA_bl']==0)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('PTAU: %d' % len(np.unique(nm)))
    # pept = (sig_uni['PTAU_bl']==0) & (sig_uni['ABETA_bl']==1)
    # nm = [x.split("|", 1)[0] for x in pept.index[pept].tolist()]
    # print('AB: %d' % len(np.unique(nm)))

    ###############################
    # FIGURE COGNITIVE TRAJECTORY #
    ###############################

    # figure - trajectories by cognitive dx and atn status
    dx_cols = ['grey', 'lightskyblue', 'lightcoral']
    trajs = ['Hippocampus', 'MOCA', 'CDRSB']
    lims = [[-600, 200], [-6, 4], [-1, 5]]
    dxs = {'MCI': 'cornflowerblue', 'CN': 'lightgrey', 'Dementia': '#f1b7f0'}
    dxs1 = {'A-T+': 'lightgreen', 'A+T-': 'cornflowerblue',
            'A-T-': 'grey', 'A+T+': '#f1b7f0'}
    for traj, yl in zip(trajs, lims):
        print(traj)

        # minimum of three visits and remove trajectories greater than 4 std
        y = data.feats[traj]
        zy = y.copy()
        zy = ((zy-zy.mean())/zy.std())
        filt = (~np.isnan(y)) & (np.abs(zy) < 4) & (data.feats[traj+'_lv'] > 3)

        # apply this boolean to xy coordinates
        bool2 = data.xy.RID == 0
        for rid in filt[filt].index.to_list():
            bool1 = data.xy.RID == rid
            bool2 = bool1 + bool2
        xy = data.xy[bool2]

        # table 2 demogrpahics for filtered participants
        feats_filt = data.feats[filt]

        # print('N:%d'% np.sum(filt))

        # # mean age, last visit, number of visits
        # mcols = ['AGE',traj,traj+'_lv',traj+'_nv']
        # for col in mcols:
        #     print(' %s: Mean: %.02f STD: %.02f' % (col,np.mean(feats_filt[col]),np.std(feats_filt[col])))
        # # last diagnosis
        # dxsl = np.unique(feats_filt['DX_last'])
        # print('DX_last')
        # for dx in dxsl:
        #     bl = feats_filt['DX_last']==dx
        #     print(' %s:%d'% (dx,np.sum(bl)))

        # # ['e33', 'e34', 'e23', 'e44', 'e24', 'e22']
        # # characeristics
        # orcols = ['DX_last','DX_bl_new','ATN','APOE_geno','change_CN_MCIdementia','PTGENDER']
        # for col in orcols:
        #     ents = np.unique(feats_filt[col])
        #     print(col)
        #     for ent in ents:
        #         y = feats_filt[col]
        #         print(' %s: %.01f %.01f' % (ent,sum(y==ent)/len(y)*100,sum(y==ent)))

        # raw trajectories by dx
        new_feats = feats_filt
        new_dxs = new_feats['DX_last'].to_list()
        dx_cols = [dxs[dx] for dx in new_dxs]
        Plotter().line_sns(xy[xy['traj'] == traj], 'Years_bl', 'ypred', join(
            rawfig_path, '%s_dx.pdf' % traj), hue='RID', alpha=0.3, palette=dx_cols)

        # raw trajectories by dx
        new_feats = feats_filt
        new_dxs = new_feats['ATN'].to_list()
        dx_cols = [dxs1[dx] for dx in new_dxs]
        Plotter().line_sns(xy[xy['traj'] == traj], 'Years_bl', 'ypred', join(
            rawfig_path, '%s_atn.pdf' % traj), hue='RID', alpha=0.3, palette=dx_cols)
        plt.close('all')

        # #boxplot by dx
        Plotter(ylim=yl).box_sns(new_feats, 'DX_last', traj, join(rawfig_path, '%s_box_dx.pdf' %
                                                                  traj), palette=['lightgrey', 'cornflowerblue', '#f1b7f0'], order=['CN', 'MCI', 'Dementia'])

        # boxplot by atn
        Plotter(ylim=yl).box_sns(new_feats, 'ATN', traj, join(rawfig_path, '%s_box_atn.pdf' % traj), palette=[
            'lightgrey', 'lightgreen', 'cornflowerblue', '#f1b7f0'], order=['A-T-', 'A-T+', 'A+T-', 'A+T+'])

        # stats
        c0 = new_feats[traj][new_feats['DX_last'] == 'CN']
        c1 = new_feats[traj][new_feats['DX_last'] == 'MCI']
        c2 = new_feats[traj][new_feats['DX_last'] == 'Dementia']
        _, pval = ttest_ind(c0, c1, nan_policy='omit')
        dp = (np.mean(c1) - np.mean(c0)) / \
            (np.sqrt((np.std(c0) ** 2 + np.std(c1) ** 2) / 2))
        print(' Con vs MCI: p:%2e, d:%.02f' % (pval, dp))
        _, pval = ttest_ind(c0, c2, nan_policy='omit')
        dp = (np.mean(c2) - np.mean(c0)) / \
            (np.sqrt((np.std(c0) ** 2 + np.std(c2) ** 2) / 2))
        print(' Con vs Dementia: p:%2e, d:%.02f' % (pval, dp))

        c0 = new_feats[traj][new_feats['ATN'] == 'A+T+']
        c1 = new_feats[traj][new_feats['ATN'] == 'A+T-']
        c2 = new_feats[traj][new_feats['ATN'] == 'A-T+']
        c3 = new_feats[traj][new_feats['ATN'] == 'A-T-']
        _, pval = ttest_ind(c0, c3, nan_policy='omit')
        dp = (np.mean(c0) - np.mean(c3)) / \
            (np.sqrt((np.std(c0) ** 2 + np.std(c3) ** 2) / 2))
        print(' A+T+ vs A-T-: p:%2e, d:%.02f' % (pval, dp))
        _, pval = ttest_ind(c0, c2, nan_policy='omit')
        dp = (np.mean(c1) - np.mean(c3)) / \
            (np.sqrt((np.std(c1) ** 2 + np.std(c3) ** 2) / 2))
        print(' A+T- vs A-T-: p:%2e, d:%.02f' % (pval, dp))
        _, pval = ttest_ind(c2, c3, nan_policy='omit')
        dp = (np.mean(c2) - np.mean(c3)) / \
            (np.sqrt((np.std(c2) ** 2 + np.std(c3) ** 2) / 2))
        print(' A-T+ vs A-T-: p:%2e, d:%.02f' % (pval, dp))

    # pdb.set_trace()

    # # protein similarity
    # sim = ML(score_uni2).cluster_corr(ndf.corr().loc[trajs,trajs])
    # Plotter().heat_sns(sim.corr,path=join(rawfig_path,'similarity.pdf'),annot=True,cmap='Blues')

    pdb.set_trace()
    ndf = (score_uni2-score_uni2.min())/(score_uni2.max()-score_uni2.min())
    sig_uni[ndf < .03] = False

    ndf['Mean'] = ndf.mean(axis=1)
    sig_uni['Mean'] = ndf['Mean']
    score_uni2['Mean'] = ndf['Mean']
    score_uni['Mean'] = ndf['Mean']
    sort_sig = ndf.sort_values(by='Mean', ascending=False)[
        sig_uni.sort_values(by='Mean', ascending=False)[trajs]][trajs]
    raw_sig = score_uni.sort_values(by='Mean', ascending=False)[
        sig_uni.sort_values(by='Mean', ascending=False)[trajs]][trajs]

    # # raw and normalized correlations
    # Plotter(size=[8,10]).heat_sns(sort_sig,xticklabels=sort_sig.columns,path=join(rawfig_path,'heat.pdf'),annot=True,cmap='OrRd',linewidths=0,vmin=0.5,vmax=1)
    # Plotter(size=[8,10]).heat_sns(raw_sig,xticklabels=raw_sig.columns,path=join(rawfig_path,'heat_raw.pdf'),annot=True,cmap='coolwarm',linewidths=0,vmin=-0.5,vmax=0.5)

    # # normalized scatters
    # ndf['col'] = 1
    # Plotter(xlim=[-0.2,1.2],ylim=[-0.2,1.2]).scatter_sns(ndf,'ABETA_bl','AV45',join(rawfig_path,'abeta_av45_nscatter.pdf'),hue='col',palette=['cornflowerblue'],alpha=1,s=300)
    # Plotter(xlim=[-0.2,1.2],ylim=[-0.2,1.2]).scatter_sns(ndf,'TAU_bl','PTAU_bl',join(rawfig_path,'tau_ptau_nscatter.pdf'),hue='col',palette=['#f1b7f0'],alpha=1,s=200)
    # Plotter(xlim=[-0.2,1.2],ylim=[-0.2,1.2]).scatter_sns(ndf,'Hippocampus_bl','FDG_bl',join(rawfig_path,'hipp_fdg_nscatter.pdf'),hue='col',palette=['lightgreen'],alpha=1,s=200)
    # Plotter(xlim=[-0.2,1.2],ylim=[-0.2,1.2]).scatter_sns(ndf,'MOCA_bl','CDRSB_bl',join(rawfig_path,'moca_cdrsb_nscatter.pdf'),hue='col',palette=['lightgrey'],alpha=1,s=200)

    # unnormalized scatters
    score_uni['col'] = 1
    Plotter(xlim=[-1, 1], ylim=[-1, 1]).scatter_sns(score_uni, 'TAU_bl', 'PTAU_bl', join(
        rawfig_path, 'tau_ptau_scatter.pdf'), hue='col', palette=['#f1b7f0'], alpha=1, s=200)
    Plotter(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5]).scatter_sns(score_uni, 'ABETA_bl', 'AV45', join(
        rawfig_path, 'amy_av45_scatter.pdf'), hue='col', palette=['cornflowerblue'], alpha=1, s=200)
    Plotter(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5]).scatter_sns(score_uni, 'Hippocampus_bl', 'FDG_bl',
                                                            join(rawfig_path, 'hipp_fdg_scatter.pdf'), hue='col', palette=['lightgreen'], alpha=1, s=200)
    Plotter(xlim=[-0.5, 0.5], ylim=[-0.5, 0.5]).scatter_sns(score_uni, 'MOCA_bl', 'CDRSB_bl',
                                                            join(rawfig_path, 'moca_cdrsb_scatter.pdf'), hue='col', palette=['lightgrey'], alpha=1, s=200)

    Plotter().scatter_sns(score_uni, 'AV45', 'PTAU_bl', join(rawfig_path,
                                                             'av45_ptau_scatter.pdf'), hue='col', palette=['lightgrey'], alpha=1, s=200)

    # # protein similarity
    # sim = ML(score_uni2).cluster_corr(ndf.corr().loc[trajs,trajs])
    # Plotter().heat_sns(sim.corr,path=join(rawfig_path,'similarity.pdf'),annot=True,cmap='OrRd')

    # # traj correlation
    # Plotter().heat_sns(data.feats[trajs][filt].corr().loc[trajs,trajs],path=join(rawfig_path,'traj_corr.pdf'),annot=True)

    # # figure - correlation and similarity
    # trajs = ['AV45','ABETA_bl','PTAU_bl','TAU_bl','Hippocampus_bl','FDG_bl','MOCA_bl','CDRSB_bl','MOCA','CDRSB','Hippocampus']
    # keys = ['prot_bl']
    # score_uni = pd.DataFrame(index=preds['prot_bl'],columns=trajs)
    # sig_uni = pd.DataFrame(index=preds['prot_bl'],columns=trajs)
    # threshs = [3]

    # #score3D = np.empty((len(threshs),len(trajs),len(keys)));score3D[:]=np.nan
    # cols  = ['white','lightgrey','lightgreen','cornflowerblue','#f1b7f0','lightgreen','cornflowerblue','lightgreen','cornflowerblue','lightgreen','cornflowerblue']

    # trajs = ['Hippocampus_bl','FDG_bl','AV45','ABETA_bl','PTAU_bl','TAU_bl','MOCA_bl','CDRSB_bl']

    # score_uni = pd.DataFrame(index=preds['prot_bl'],columns=trajs)
    # score_uni2 = pd.DataFrame(index=preds['prot_bl'],columns=trajs)
    # sig_uni = pd.DataFrame(index=preds['prot_bl'],columns=trajs)
    # cols  = ['lightgreen','lightgreen','cornflowerblue','cornflowerblue','#f1b7f0','#f1b7f0','lightgrey','lightgrey','lightgrey','lightgrey','lightgreen']

    # for count,thresh in enumerate(threshs):
    #     score = pd.DataFrame(index=trajs,columns=keys)
    #     for traj,col in zip(trajs,cols):
    #         if traj=='MOCA':
    #             keys = ['gen_bl','amy_all_bl','tau_all_bl','deg_all_bl','cog_all_bl','prot_bl','all_bl']
    #         if traj=='CDRSB':
    #             keys = ['gen_bl','amy_all_bl','tau_all_bl','deg_all_bl','cog_all_bl','prot_bl','all_bl']
    #         if traj=='Hippocampus':
    #             keys = ['gen_bl','amy_all_bl','tau_all_bl','deg_all_bl','cog_all_bl','prot_bl','all_bl']
    #         score = pd.DataFrame(index=trajs,columns=keys)

    #         for key in keys:
    #             print(traj + ' ' + key)
    #             # loop through features
    #             y = data.feats[traj].astype('float64')
    #             zy = y.copy(); zy = ((zy-zy.mean())/zy.std())

    #             # filter by z-score, max visit time, and get demographics
    #             if (traj == 'MOCA') | (traj == 'CDRSB') | (traj == 'Hippocampus'):
    #                 filt = (~np.isnan(y))& (np.abs(zy)<4) & (data.feats[traj+'_lv']>thresh)
    #             else:
    #                 filt = (~np.isnan(y))& (np.abs(zy)<4)
    #             x_filt = data.feats[preds[key]][filt]
    #             y_filt = y[filt]

    #             #elastic net
    #             # mltraj = ML(x=x_filt,y=y_filt.to_numpy()).impute().norm_clip(5).init('ENCV').train_test(score='R')
    #             #score.loc[traj,key]= mltraj.score
    #             #print(' R: %.02f P: %.2e' % (mltraj.score,mltraj.p))

    #             if key == 'prot_bl':
    #                 ml = ML(x=data.feats[preds['prot_bl']][filt],y=y_filt.to_numpy()).impute().norm_clip(5).init('LR').train_test_uni(score='R',alpha=0.01)
    #                 score_uni.loc[:,traj] = data.feats[preds['prot_bl']].corrwith(y)#ml.score_uni['R']

    #                 score_uni2.loc[:,traj] = ml.score_uni['R'].astype('float64')
    #                 sig_uni.loc[:,traj] = ml.score_uni['FDR']

    #             # if key == 'prot_bl':
    #             #     Plotter().scatter([mltraj.ypred],[mltraj.y],join(rawfig_path,'%s_%s_reg_dx.pdf' % (traj,key)),cols=[col],alpha=1,s=100)

    #             # plt.close('all')

    #             # # atn plot
    #             # dx_filt = data.feats['ATN'][filt].to_list(); dx_cols = [dxs1[dx] for dx in dx_filt]
    #             # Plotter(newfig=False).scatter([mltraj.ypred],[mltraj.y],join(rawfig_path,'%s_%s_reg_atn.pdf' % (traj,key)),cols=[col],alpha=0.7,s=100)

    #             # Plotter().scatter([mltraj.ypred],[mltraj.y],join(rawfig_path,'%s_%s_reg.pdf' % (traj,key)),cols=[col],alpha=0.7,s=100)

    #             plt.close('all')
    #             if traj=='MOCA':
    #                 scores = score.loc[traj,['amy_all_bl','tau_all_bl','deg_all_bl','cog_all_bl','prot_bl','all_bl']].to_numpy()
    #             if traj=='CDRSB':
    #                 scores = score.loc[traj,['amy_all_bl','tau_all_bl','deg_all_bl','cog_all_bl','prot_bl','all_bl']].to_numpy()
    #             if traj=='Hippocampus':
    #                 scores = score.loc[traj,['amy_all_bl','tau_all_bl','deg_all_bl','cog_all_bl','prot_bl','all_bl']].to_numpy()

    #         # Plotter().bar([1,2,3,4,5,6],np.abs(scores),join(rawfig_path,'%s_reg.pdf' % (traj)),['cornflowerblue','#f1b7f0','lightgreen','lightgrey','white','black'],alpha=1,orient='h')

    # pdb.set_trace()
    # ndf=(score_uni2-score_uni2.min())/(score_uni2.max()-score_uni2.min())
    # sig_uni[ndf<.03] = False

    # ndf['Mean'] = ndf.mean(axis=1)
    # sig_uni['Mean'] = ndf['Mean']
    # score_uni2['Mean'] = ndf['Mean']
    # score_uni['Mean'] = ndf['Mean']
    # sort_sig = ndf.sort_values(by='Mean',ascending=False)[sig_uni.sort_values(by='Mean',ascending=False)[trajs]][trajs]
    # raw_sig = score_uni.sort_values(by='Mean',ascending=False)[sig_uni.sort_values(by='Mean',ascending=False)[trajs]][trajs]

    # # raw and normalized correlations
    # Plotter(size=[6,10]).heat_sns(sort_sig,xticklabels=sort_sig.columns,path=join(rawfig_path,'heat_traj.pdf'),vmin=0,vmax=1,annot=True)
    # Plotter(size=[6,10]).heat_sns(raw_sig,xticklabels=raw_sig.columns,path=join(rawfig_path,'heat_raw_traj.pdf'),annot=True)

    # # normalized scatters
    # ndf['col'] = 1
    # Plotter(xlim=[-0.2,1.2],ylim=[-0.2,1.2]).scatter_sns(ndf,'ABETA_bl','AV45',join(rawfig_path,'abeta_av45_nscatter.pdf'),hue='col',palette=['cornflowerblue'],alpha=1,s=300)
    # Plotter(xlim=[-0.2,1.2],ylim=[-0.2,1.2]).scatter_sns(ndf,'TAU_bl','PTAU_bl',join(rawfig_path,'tau_ptau_nscatter.pdf'),hue='col',palette=['#f1b7f0'],alpha=1,s=200)
    # Plotter(xlim=[-0.2,1.2],ylim=[-0.2,1.2]).scatter_sns(ndf,'Hippocampus_bl','FDG_bl',join(rawfig_path,'hipp_fdg_nscatter.pdf'),hue='col',palette=['lightgreen'],alpha=1,s=200)
    # Plotter(xlim=[-0.2,1.2],ylim=[-0.2,1.2]).scatter_sns(ndf,'MOCA_bl','CDRSB_bl',join(rawfig_path,'moca_cdrsb_nscatter.pdf'),hue='col',palette=['lightgrey'],alpha=1,s=200)

    # unnormalized scatters
    score_uni2['col'] = 1
    Plotter().scatter_sns(np.abs(score_uni2), 'TAU_bl', 'PTAU_bl', join(rawfig_path,
                                                                        'tau_ptau_nscatter.pdf'), hue='col', palette=['lightgrey'], alpha=1, s=200)
    Plotter().scatter_sns(np.abs(score_uni2), 'ABETA_bl', 'AV45', join(rawfig_path,
                                                                       'abeta_av45_nscatter.pdf'), hue='col', palette=['cornflowerblue'], alpha=1, s=200)
    Plotter().scatter_sns(np.abs(score_uni2), 'Hippocampus_bl', 'FDG_bl', join(rawfig_path,
                                                                               'hipp_fdg_nscatter.pdf'), hue='col', palette=['lightgreen'], alpha=1, s=200)
    Plotter().scatter_sns(np.abs(score_uni2), 'MOCA_bl', 'CDRSB_bl', join(rawfig_path,
                                                                          'moca_cdrsb_nscatter.pdf'), hue='col', palette=['lightgrey'], alpha=1, s=200)

    # protein similarity
    Plotter().heat_sns(ndf.corr().loc[['MOCA', 'CDRSB', 'Hippocampus'], ['Hippocampus_bl', 'FDG_bl', 'AV45', 'ABETA_bl',
                                                                         'PTAU_bl', 'TAU_bl', 'MOCA_bl', 'CDRSB_bl']], path=join(rawfig_path, 'similarity_traj.pdf'), annot=True)

    # traj correlation
    Plotter().heat_sns(data.feats[trajs][filt].corr().loc[trajs, trajs], path=join(
        rawfig_path, 'traj_corr.pdf'), annot=True)

    # # pdb.set_trace()

    # # corr = np.abs(score_uni).corr()
    # # cos = corr.copy()
    # # for i1 in score_uni.columns:
    # #     for i2 in score_uni.columns:
    # #         a = np.abs(score_uni[i1])
    # #         b = np.abs(score_uni[i2])
    # #         cos.loc[i1,i2] = np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

    # # pdb.set_trace()


if __name__ == "__main__":

    main()
