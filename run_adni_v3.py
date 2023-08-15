'''
run_adni_v3.py
Rafi Haque rafihaq90@gmail.com
--------------------
Description:
--------------------
Instructions:
'''
from MLADNI import ML
from ADNIData import ADNIData
from Plotter import Plotter
from os.path import join
import matplotlib.pyplot as plt
import csv
from itertools import compress
from scipy.stats import ttest_ind
import pickle
import argparse
import os
import pdb
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)


# paths
prot_path = 'Final_peptides_after_QC.csv'
merge_path = 'Final_Clinical_Data.csv'
obj_path = 'ADNIDATA.obj'
rawfig_path = 'rawfig/'


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
        'mol_bl': ['ABETA_bl', 'TAU_bl', 'PTAU_bl']
    }
    merge_bls = preds['amy_all_bl'] + preds['tau_all_bl'] + preds['cog_all_bl'] + \
        preds['cli_bl'] + preds['deg_all_bl'] + preds['gen_bl'] + ['APOE_geno']

    # load and save adni data
    cog = ['CDRSB', 'MOCA']
    mri = ['Ventricles', 'Hippocampus', 'WholeBrain',
           'Entorhinal', 'Fusiform', 'MidTemp']

    # save data
    #data = ADNIData(merge_path = merge_path, feats_path=prot_path,trajs=cog+mri,bls=merge_bls,clis=preds['cli_bl'],tras=preds['tra_bl']).get_base().get_trajs().save(obj_path)
    # load adni data
    data = ADNIData(merge_path=merge_path, feats_path=prot_path,
                    trajs=cog+mri, bls=merge_bls).load(obj_path)

    # update dictionary with feature combinations features
    preds.update({'cli_bl+bat_bl': preds['cli_bl']+preds['bat_bl']})
    preds.update({'prot_bl': data.new_prots.tolist()})
    preds.update({'prot_bl+tau_bl': preds['prot_bl']+preds['tau_bl']})
    preds.update({'prot_bl+ptau_bl': preds['prot_bl']+preds['ptau_bl']})
    preds.update({'prot_bl+amy_bl': preds['prot_bl']+preds['amy_bl']})
    preds.update({'prot_bl+av45_bl': preds['prot_bl']+preds['av45_bl']})
    preds.update({'prot_bl+mol_bl': preds['prot_bl']+preds['mol_bl']})
    preds.update({'all_bl': preds['prot_bl']+preds['mol_bl']+preds['age_bl']})

    ##########
    # TABLES #
    ##########
    # loop through each dx
    dxs = np.unique(data.feats['DX_bl_new'])
    for dx in dxs:
        bl = data.feats['DX_bl_new'] == dx
        print('%s:%d' % (dx, np.sum(bl)))

    # average age
    mcols = ['AGE']
    for col in mcols:
        print('%s: Mean: %.02f STD: %.02f' %
              (col, np.mean(data.feats[col]), np.std(data.feats[col])))

    # table 1  dx by percentage, ATN, APOE,etc
    orcols = ['DX_bl_new', 'ATN', 'PTGENDER']

    for col in orcols:
        ents = np.unique(data.feats[col])
        for ent in ents:
            y = data.feats[col]
            print('%s: %.02f %.02f' %
                  (ent, sum(y == ent)/len(y), sum(y == ent)))

    # table 2 - breakdown by dx
    y = data.feats['DX_bl_new']
    cols = ['PTGENDER', 'APOE_geno', 'ATN', 'MOCA_bl', 'CDRSB_bl', 'AGE']
    dxs = ['CN', 'MCI', 'Dementia']

    apoe = ['33', '34', '23', '44', '24', '22']
    for col in cols:
        for dx in dxs:
            num = len(data.feats[y == dx][col])
            print('******%s %s %d' % (dx, col, num))

            if (col == 'MOCA_bl') | (col == 'CDRSB_bl') | (col == 'AGE'):
                mean = np.mean(data.feats[y == dx][col])
                std = np.std(data.feats[y == dx][col])
                print('%s:%.2f +/- %.1f' % (dx, mean, std))
            else:
                for col2 in np.unique(data.feats[col]):
                    num = np.sum(data.feats[y == dx][col] == col2)
                    print('%s %s:%d' % (col2, dx, num))

    # trajs
    trajs = ['MOCA', 'Hippocampus', 'CDRSB']
    for traj in trajs:
        y = data.feats[traj].astype('float64')
        zy = y.copy()
        zy = ((zy-zy.mean())/zy.std())
        filt = (~np.isnan(y)) & (np.abs(zy) < 4) & (data.feats[traj+'_lv'] > 3)
        m = data.feats[filt][traj]
        age = data.feats[filt]['AGE']
        nv = data.feats[filt][traj+'_nv']
        lv = data.feats[filt][traj+'_lv']
        print('%s: %.02f + %.02f' % (traj, np.mean(m), np.std(m)))
        print(' Age: %.02f + %.02f' % (np.mean(age), np.std(age)))
        print(' LV: %.02f + %.02f' % (np.mean(lv), np.std(lv)))
        print(' NV: %.02f + %.02f\n' % (np.mean(nv), np.std(nv)))

    ########################################
    # CLASSIFY DX, ATN STATUS, and DIFFEX  #
    ########################################
    orcols = ['DX_bl_new', 'AV', 'A', 'ATN']
    orcond = [['CN', 'Dementia'], ['AV-', 'AV+'],
              ['A-', 'A+'], ['A-T-', 'A+T+']]
    names = ['baseDEM', 'baseAV+', 'baseA+', 'baseA+T+']
    models = ['LGRCV', 'LGRCV', 'LGRCV']
    feats = ['PROT', 'GENO', 'CLI', 'MOL',
             'MOL + PROT', 'AMY', 'TAU', 'PTAU', 'RAT']
    score = pd.DataFrame(index=preds['prot_bl'], columns=names)
    score_csv = pd.DataFrame(index=feats, columns=[
                             'P', 'P-PROT', 'P-PROT-MOL'])

    # loop through each comparision
    for col, cond, name, model in zip(orcols, orcond, names, models):
        print('CLASS: %s' % col)
        print('MODEL: %s' % model)

        # load adni data and labels
        data = ADNIData(merge_path=merge_path, feats_path=prot_path,
                        trajs=cog+mri, bls=merge_bls).load(obj_path)
        data.or_filter([col]*2, cond)
        y = data.get_y(col, cond, [0, 1])

        # run model for each protein
        ml = ML(x=data.feats[preds['mol_bl']], y=y).diff(
            0, 1, .05).impute().init('LGR').train_test_uni()
        rn = 'log2(%s/%s)' % (cond[1], cond[0])
        new = ml.diffex.rename(columns={"log2(1/0)": rn})

        new[[rn, 'P', 'PFDR', 'AUC', 'CI']].sort_values(
            by='P').to_csv(join(rawfig_path, 'TableS%s.csv' % name))
        pdb.set_trace()
        # get number of significant peptides and proteins
        pbool = (ml.diffex['PFDR']) & (ml.diffex['log2(1/0)'] > 0)
        nbool = (ml.diffex['PFDR']) & (ml.diffex['log2(1/0)'] < 0)
        pm = [x.split("|", 1)[0] for x in new.index[pbool].tolist()]
        nm = [x.split("|", 1)[0] for x in new.index[nbool].tolist()]
        print('+ Peptides:%d' % sum(pbool))
        print('- Peptides:%d' % sum(nbool))
        print('+ Prots:%d' % len(np.unique(pm)))
        print('- Prots:%d' % len(np.unique(nm)))
        print(new.sort_values(by='AUC'))
        pdb.set_trace()

        # run model with all proteins
        print('PROT:')
        ml1 = ML(x=data.feats[preds['prot_bl']], y=y).impute().norm_clip().init(
            model).train_test(score='AUC').select(1).train_test_uni()
        score.loc[:, name] = ml1.diffex['AUC']
        score_csv.loc['PROT', '%s AUC' % name] = '%.2f' % ml1.score
        print(ml1.score)
        print(ml1.ci)

        # run apoe genotype
        print('GENO:')
        ml2 = ML(x=data.feats[preds['gen_bl']], y=y).impute().init(
            'LGR').train_test(score='AUC')
        print(ml2.score)
        print(ml2.ci)
        score_csv.loc['GENO', '%s AUC' % name] = '%.2f' % ml2.score

        # run clinical co-variates and batch
        print('CLI + BATCH:')
        ml3 = ML(x=data.feats[preds['cli_bl+bat_bl']],
                 y=y).impute().norm_clip().init('LGR').train_test(score='AUC')
        print(ml3.score)
        print(ml3.ci)
        score_csv.loc['CLI', '%s AUC' % name] = '%.2f' % ml3.score
        score_csv.loc['CLI', '%s P-PROT' %
                      name] = '%.2e' % ml1.compare(ml3.ypred)

        # run ad biomarkers
        print('MOL:')
        ml4 = ML(x=data.feats[preds['mol_bl']], y=y).impute(
        ).norm_clip().init(model).train_test(score='AUC')
        print(ml4.score)
        print(ml4.ci)
        score_csv.loc['MOL', '%s AUC' % name] = '%.2f' % ml4.score
        score_csv.loc['MOL', '%s P-PROT' %
                      name] = '%.2e' % ml1.compare(ml4.ypred)

        # run ad biomarkers and proteins
        print('MOL + PROT:')
        ml5 = ML(x=data.feats[preds['prot_bl+mol_bl']],
                 y=y).impute().norm_clip().init(model).train_test(score='AUC')
        print(ml5.score)
        print(ml5.ci)
        score_csv.loc['MOL + PROT', '%s AUC' % name] = '%.2f' % ml5.score
        score_csv.loc['MOL + PROT', '%s P-PROT' %
                      name] = '%.2e' % ml1.compare(ml5.ypred)
        score_csv.loc['MOL', '%s P-PROT-MOL' %
                      name] = '%.2e ' % ml5.compare(ml4.ypred)

        # run amyloid
        print('AMY:')
        mla = ML(x=data.feats[preds['amy_bl']], y=y).impute(
        ).norm_clip().init(model).train_test(score='AUC')
        print(mla.score)
        print(mla.ci)
        score_csv.loc['AMY', '%s AUC' % name] = '%.2f' % mla.score

        # run tau
        print('TAU:')
        mlt = ML(x=data.feats[preds['tau_bl']], y=y).impute(
        ).norm_clip().init(model).train_test(score='AUC')
        print(mlt.score)
        print(mlt.ci)
        score_csv.loc['TAU', '%s AUC' % name] = '%.2f' % mlt.score

        # run ptau
        print('PTAU:')
        mlpt = ML(x=data.feats[preds['ptau_bl']], y=y).impute(
        ).norm_clip().init(model).train_test(score='AUC')
        print(mlpt.score)
        print(mlpt.ci)
        score_csv.loc['PTAU', '%s AUC' % name] = '%.2f' % mlpt.score2

        # run ratio
        print('RAT:')
        mlr = ML(x=data.feats[preds['rat_bl']], y=y).impute(
        ).norm_clip().init(model).train_test(score='AUC')
        print(mlr.score)
        print(mlr.ci)
        score_csv.loc['RAT', '%s AUC' % name] = '%.2f' % mlr.score

        print('STATS:')
        print('PROT to MOL %.2e' % ml1.compare(ml4.ypred))
        print('PROT to AGE: %.2e' % ml1.compare(ml3.ypred))
        print('PROT to Genotype: %.2e' % ml1.compare(ml2.ypred))
        print('PROT+MOL to MOL: %.2e ' % ml5.compare(ml4.ypred))

        gr = ml1.get_perm_diff(y, ml1.yprob, ml4.yprob, 1000, 'AUC')
        print('PROT to MOL %.05f' % gr[1])

        gr = ml1.get_perm_diff(y, ml1.yprob, ml3.yprob, 1000, 'AUC')
        print('PROT to AGE: %.05f' % gr[1])

        gr = ml1.get_perm_diff(y, ml1.yprob, ml2.yprob, 1000, 'AUC')
        print('PROT to Genotype: %.05f' % gr[1])

        gr = ml1.get_perm_diff(y, ml5.yprob, ml4.yprob, 1000, 'AUC')
        print('PROT+MOL to MOL: %.05f ' % gr[1])

        if (name == 'baseA+T+'):
            ml6 = ML(x=data.feats[preds['prot_bl']], y=y).diff(
                0, 1, .05).impute().init('LGR').train_test_uni()
            ml6.diffex.sort_values(by='AUC', ascending=False).to_csv(
                join(rawfig_path, 'diffex_%s.csv' % name))
            # Plotter(xlim=[0,1],ylim=[0,1]).roc([y]*5,[ml1.yprob],join(rawfig_path,'roc_%s.pdf' % name),cols=['cornflowerblue','lightgrey','#f1b7f0','lightgreen'])
        else:
            ml6 = ML(x=data.feats[preds['prot_bl+mol_bl']], y=y).diff(0,
                                                                      1, .05).impute().init('LGR').train_test_uni()
            #Plotter(xlim=[0,1],ylim=[0,1]).roc([y]*5,[ml4.yprob,ml1.yprob,ml5.yprob],join(rawfig_path,'roc_%s.pdf' % name),cols=['grey','cornflowerblue','lightcoral'])

        # get unique proteins
        uni_prots = np.array([x.split('|')[0] for x in ml6.diffex.index])
        score_auc = pd.DataFrame(index=np.unique(
            uni_prots), columns=ml6.diffex.columns)

        for prot in np.unique(uni_prots):
            score_pep = ml6.diffex[uni_prots == prot]
            score_auc.loc[prot] = score_pep.iloc[np.argmax(score_pep['AUC'])]

        # save out table
        rn = 'log2(%s/%s)' % (cond[1], cond[0])
        m1 = cond[1]+' Mean'
        m0 = cond[0]+' Mean'
        s1 = cond[1]+' STD'
        s0 = cond[0]+' STD'

        new = score_auc.rename(
            columns={"log2(1/0)": rn, 'M1': m1, 'M0': m0, 'S1': s1, 'S0': s0})
        new[[rn, m1, s1, m0, s0, 'P', 'PFDR', 'AUC', 'CI']].sort_values(
            by='P').to_csv(join(rawfig_path, 'TableSU_%s.csv' % name))
        score_auc['AUC'][~score_auc['PFDR']] = np.nan
        pdb.set_trace()

        pl = Plotter(xlim=[-1, 1]).diffex(score_auc, join(rawfig_path,
                                                          'diffex_%s.pdf' % name), x='log2(1/0)', s=100, cmap='Blues')

        # # pdb.set_trace()

        # #print(ml6.diffex.sort_values(by='AUC',ascending=False))

    ################
    # CORRELATIONS #
    ################
    trajs = ['MOCA', 'CDRSB', 'Hippocampus', 'FDG_bl', 'Hippocampus_bl',
             'AV45', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'MOCA_bl', 'CDRSB_bl']
    #trajs = ['Hippocampus_bl']
    #trajs = ['FDG_bl','Hippocampus_bl','MOCA_bl','CDRSB_bl']
    trajs = ['AV45', 'ABETA_bl', 'TAU_bl', 'PTAU_bl']
    #trajs = ['MOCA','CDRSB','Hippocampus']
    score_uni = pd.DataFrame(index=preds['prot_bl+mol_bl'], columns=trajs)
    sig_uni = pd.DataFrame(index=preds['prot_bl+mol_bl'], columns=trajs)
    score_p = pd.DataFrame(index=preds['prot_bl+mol_bl'], columns=trajs)

    score_uni = pd.DataFrame(index=preds['prot_bl'], columns=trajs)
    sig_uni = pd.DataFrame(index=preds['prot_bl'], columns=trajs)
    score_p = pd.DataFrame(index=preds['prot_bl'], columns=trajs)

    # loop through each outcome
    for traj in trajs:
        # feature combinations for each trait
        if traj == 'AV45':
            keys = ['age_bl', 'ptau_bl', 'amy_bl', 'prot_bl', 'prot_bl+amy_bl']
        elif traj == 'ABETA_bl':
            keys = ['age_bl', 'ptau_bl', 'av45_bl',
                    'prot_bl', 'prot_bl+av45_bl']
        elif traj == 'TAU_bl':
            keys = ['age_bl', 'ptau_bl', 'amy_bl',
                    'prot_bl', 'prot_bl+ptau_bl']
        elif traj == 'PTAU_bl':
            keys = ['age_bl', 'tau_bl', 'amy_bl', 'prot_bl', 'prot_bl+tau_bl']
        else:
            keys = ['age_bl', 'gen_bl', 'mol_bl',
                    'prot_bl', 'prot_bl+mol_bl', 'all_bl']

        # store R and p value for each outcome for each feature combos
        score = pd.DataFrame(index=trajs, columns=keys)
        pr = pd.DataFrame(index=trajs, columns=keys)

        # filter by z-score, max visit time, and get demographics if trajectory else just filter by outlier
        y = data.feats[traj].astype('float64')
        zy = y.copy()
        zy = ((zy-zy.mean())/zy.std())

        print(sum(np.isnan(y)))
        if (traj == 'Hippocampus') | (traj == 'MOCA') | (traj == 'CDRSB'):
            filt = (~np.isnan(y)) & (np.abs(zy) < 4) & (
                data.feats[traj+'_lv'] > 3) & (data.feats['ATN'] == 'A+T+')
        else:
            filt = (~np.isnan(y)) & (np.abs(zy) < 4)
        print(sum(filt))

        # loop through each trait
        ypreds = pd.DataFrame(index=y[filt].index, columns=keys)
        for count, key in enumerate(keys):

            # get features and labels
            print(traj + ' ' + key)
            x_filt = data.feats[preds[key]][filt]
            y_filt = y[filt]

            # # run elastic net and use pearson correlation for performance
            # mltraj = ML(x=x_filt,y=y_filt.to_numpy()).impute().norm_clip(5).init('ENCV').train_test(score='PR')
            # #mltraj = ML(x=x_filt,y=y_filt.to_numpy()).impute().init('ENCV').train_test(score='PR')
            # score.loc[traj,key]= mltraj.score
            # pr.loc[traj,key]= mltraj.p
            # print(' R: %.02f P: %.2e \n' % (mltraj.score,mltraj.p))
            # ypreds.loc[:,key] =  mltraj.ypred

            # store correlations for each individual protein
            if (traj == 'ABETA_bl') | (traj == 'TAU_bl') | (traj == 'PTAU_bl') | (traj == 'AV45'):
                if (key == 'prot_bl'):
                    ml = ML(x=data.feats[preds['prot_bl']][filt], y=y_filt.to_numpy()).impute(
                    ).norm_clip(5).init('LR').train_test_uni(score='PR', alpha=0.01)
                    score_uni.loc[:, traj] = ml.score_uni['PR'].astype(
                        'float64')
                    score_p.loc[:, traj] = ml.score_uni['P']
                    sig_uni.loc[:, traj] = ml.score_uni['FDR']
            else:
                if (key == 'prot_bl+mol_bl'):
                    ml = ML(x=data.feats[preds['prot_bl+mol_bl']][filt], y=y_filt.to_numpy(
                    )).impute().norm_clip(5).init('LR').train_test_uni(score='PR', alpha=0.01)
                    score_uni.loc[:, traj] = ml.score_uni['PR'].astype(
                        'float64')
                    score_p.loc[:, traj] = ml.score_uni['P']
                    sig_uni.loc[:, traj] = ml.score_uni['FDR']

            # plot proteomic regression
            if (key == 'prot_bl') | (key == 'amy_bl') | (key == 'ptau_bl') | (key == 'av45_bl') | (key == 'prot_bl+amy_bl') | (key == 'prot_bl+av45_bl'):
                Plotter().scatter([mltraj.ypred], [mltraj.y], join(
                    rawfig_path, '%s_%s_reg.pdf' % (traj, key)), cols=[col], alpha=1, s=100)

            # plot scores of all regressions
            scores = score.loc[traj, keys].to_numpy()
            prs = pr.loc[traj, keys].to_numpy()
            cols2 = ['black', 'white', 'lightgrey',
                     'cornflowerblue', 'lightcoral']
            scores[prs > 0.01] = 0
            # Plotter(xlim=[0,0.80]).bar(range(0,len(scores)),scores,join(rawfig_path,'%s_reg.pdf' % (traj)),cols2,alpha=1,orient='h')

        # permutation procedure for stats
        if len(keys) == 6:
            cind1 = [2, 3, 4, 4, 4, 3, 5, 5]
            cind2 = [0, 0, 0, 3, 2, 2, 4, 3]
        else:
            cind1 = [2, 3, 4, 4, 4, 3]
            cind2 = [0, 0, 0, 3, 2, 2]

        for c1, c2 in zip(cind1, cind2):
            print(traj + ': ' + keys[c1] + '  ' + keys[c2])
            ypred1 = ypreds[keys[c1]].to_numpy()
            ypred2 = ypreds[keys[c2]].to_numpy()
            gr = mltraj.get_perm_diff(y_filt, ypred1, ypred2, 1000, 'CORR')
            print(' O:%.05f' % gr[0])
            print(' P:%.05f\n' % gr[1])

    # unique protein table

    uni_prots = np.array([x.split('|')[2].split(';')[0]
                         for x in score_uni.index])
    ps = [col + ' P' for col in score_uni.columns]
    psig = [col + ' PSIG' for col in score_uni.columns]
    score_r = pd.DataFrame(index=np.unique(uni_prots), columns=list(
        score_uni.columns)+list(ps)+list(psig))
    for prot in np.unique(uni_prots):
        for col in (score_p.columns):
            score_ppep = score_p[uni_prots == prot][col]
            score_rpep = score_uni[uni_prots == prot][col]
            sig_rpep = sig_uni[uni_prots == prot][col]
            score_r.loc[prot, col+' PSIG'] = sig_rpep[np.argmin(score_ppep)]
            score_r.loc[prot, col+' P'] = np.min(score_ppep)
            score_r.loc[prot, col] = score_rpep[np.argmin(score_ppep)]
    score_r['Mean'] = np.mean(np.abs(score_r[score_uni.columns]), axis=1)
    pdb.set_trace()

    score_r.sort_values(
        by='Mean', ascending=False).iloc[:, :-1].to_csv(join(rawfig_path, 'TableS_corr.csv'))

    # unique protein figure
    corr = score_r.sort_values(by='CDRSB_bl', ascending=False)[
        score_uni.columns].astype('float64')
    notsig = score_r.sort_values(by='CDRSB_bl', ascending=False)[
        psig].astype('float64') != 1
    corr[notsig.to_numpy()] = np.nan
    order = ['TAU_bl', 'PTAU_bl', 'AV45', 'FDG_bl', 'Hippocampus_bl',
             'Hippocampus', 'MOCA_bl', 'MOCA', 'ABETA_bl', 'CDRSB_bl', 'CDRSB']
    corr = corr[order].rename(columns={'Hippocampus_bl': 'HIPP_bl', 'Hippocampus': 'HIPP_traj',
                              'MOCA': 'MOCA_traj', 'CDRSB': 'CDRSB_traj', 'AV45': 'AV45_bl'})
    # Plotter(size=[7,11]).heat_sns(corr.astype('float64'),path=join(rawfig_path,'corr.pdf'),annot=True,cmap='seismic',vmin=-0.8,vmax=0.8)

    # print number of unique proteins for each comparision
    score_rdrop = score_r.drop(labels=['ABETA_bl', 'TAU_bl', 'PTAU_bl'])
    [print('%s: %d' % (traj, np.sum(score_rdrop[traj + ' PSIG'] == True)))
     for traj in trajs]

    nm = (score_rdrop['AV45 PSIG']) & (score_rdrop['ABETA_bl PSIG'])
    print('AV45 & ABETA: %d' % np.sum(nm))
    nm = (score_rdrop['TAU_bl PSIG']) & (score_rdrop['PTAU_bl PSIG'])
    print('TAU & PTAU_bl: %d' % np.sum(nm))
    nm = (score_rdrop['FDG_bl PSIG']) & (score_rdrop['Hippocampus_bl PSIG'])
    print('FDG & Hippocampus: %d' % np.sum(nm))

    # tau amy
    at_trajs = ['AV45', 'ABETA_bl', 'PTAU_bl', 'TAU_bl']
    gr = score_r[at_trajs]
    for traj in at_trajs:
        gr.loc[score_r[traj + ' PSIG'] == False, traj] = 0
    gr2 = gr[gr.sum(axis=1) != 0]
    order2 = ['SMOC1_HUMAN',
              '1433Z_HUMAN',
              '1433B_HUMAN',
              'OSTP_HUMAN',
              'PPIA_HUMAN',
              'PKM2_HUMAN',
              'KPYM_HUMAN',
              'MDHC_HUMAN',
              'LDHC_HUMAN',
              'GELS_HUMAN',
              'ALDOA_HUMAN',
              'GUAD_HUMAN',
              'ENOA_HUMAN',
              'CALM2_HUMAN',
              'CD44_HUMAN',
              'CH3L1_HUMAN',
              'DDAH1_HUMAN',
              'GMFB_HUMAN',
              'PARK7_HUMAN',
              'SODC_HUMAN',
              'G3P_HUMAN',
              'LDHB_HUMAN',
              'MIME_HUMAN',
              'AATC_HUMAN',
              'PEBP1_HUMAN',
              'TPIS_HUMAN',
              'PTPRZ_HUMAN',
              'APOE_HUMAN',
              'NCAM1_HUMAN',
              'DKK3_HUMAN',
              'OMGP_HUMAN',
              'NRX1B_HUMAN',
              'SCG2_HUMAN',
              'NPTX2_HUMAN',
              'VGF_HUMAN',
              'NPTXR_HUMAN'
              ]
    gr3 = gr2.sort_values(by='ABETA_bl').astype('float64').T

    pdb.set_trace()
    Plotter(size=[12, 6]).heat_sns(gr2.loc[order2].astype('float64').T, path=join(
        rawfig_path, 'amytau_heat2.pdf'), annot=False, cmap='coolwarm', vmin=-0.8, vmax=0.8)

    # cog hip
    cog_trajs = ['CDRSB', 'MOCA', 'Hippocampus']
    gr = score_r[cog_trajs]
    for traj in cog_trajs:
        gr.loc[score_r[traj + ' PSIG'] == False, traj] = 0
    gr2 = gr[gr.sum(axis=1) != 0]
    # Plotter(size=[12,6]).heat_sns(gr2.sort_values(by='MOCA').astype('float64').T,path=join(rawfig_path,'moca_heat.pdf'),annot=False,cmap='coolwarm')
    # # score_r['Mean']=np.mean(np.abs(score_r[score_uni2.columns]),axis=1)
    # # score_r.sort_values(by='Mean',ascending=False).iloc[:,:-1].to_csv(join(rawfig_path,'TableS_corr.csv'))

    ########################
    # COGNITIVE TRAJECTORY #
    ########################
    # figure - trajectories by cognitive dx and atn status
    trajs = ['MOCA', 'CDRSB']
    lims = [[-6, 4], [-1, 5]]
    dxss = [{'MCI': 'cornflowerblue', 'CN': 'lightgrey',
             'Dementia': 'lightcoral'}, {'A-T-': 'grey', 'A+T+': 'lightcoral'}]
    # dxs1 =
    atns = [False, True]
    cols = ['DX_last', 'ATN']
    for atnb, col, dxs in zip(atns, cols, dxss):
        for traj, yl in zip(trajs, lims):
            print(traj)
            # minimum of three visits and remove trajectories greater than 4 std
            y = data.feats[traj]
            zy = y.copy()
            zy = ((zy-zy.mean())/zy.std())
            if atnb:
                filt = (~np.isnan(y)) & (np.abs(zy) < 4) & (
                    data.feats[traj+'_lv'] > 3) & ((data.feats['ATN'] == 'A+T+') | (data.feats['ATN'] == 'A-T-'))
            else:
                filt = (~np.isnan(y)) & (np.abs(zy) < 4) & (
                    data.feats[traj+'_lv'] > 3)

            # apply this boolean to xy coordinates
            bool2 = data.xy.RID == 0
            for rid in filt[filt].index.to_list():
                bool1 = data.xy.RID == rid
                bool2 = bool1 + bool2
            xy = data.xy[bool2]

            # raw trajectories by dx
            new_feats = data.feats[filt]
            new_dxs = new_feats[col].to_list()
            dx_cols = [dxs[dx] for dx in new_dxs]
            Plotter().line_sns(xy[xy['traj'] == traj], 'Years_bl', 'ypred', join(
                rawfig_path, '%s_%s_dx.pdf' % (traj, col)), hue='RID', alpha=0.3, palette=dx_cols)

            # boxplots
            if atnb:
                Plotter(ylim=yl).box_sns(new_feats, 'ATN', traj, join(
                    rawfig_path, '%s_box_atn.pdf' % traj), palette=['lightgrey', 'lightcoral'], order=['A-T-', 'A+T+'])
            else:
                Plotter(ylim=yl).box_sns(new_feats, 'DX_last', traj, join(rawfig_path, '%s_box_dx.pdf' % traj), palette=[
                    'lightgrey', 'cornflowerblue', 'lightcoral'], order=['CN', 'MCI', 'Dementia'])

            # stats

            c0 = new_feats[traj][new_feats['DX_last'] == 'CN']
            c1 = new_feats[traj][new_feats['DX_last'] == 'MCI']
            c2 = new_feats[traj][new_feats['DX_last'] == 'Dementia']
            _, pval1 = ttest_ind(c0, c1, nan_policy='omit')
            _, pval2 = ttest_ind(c0, c2, nan_policy='omit')

            c0 = new_feats[traj][new_feats['ATN'] == 'A+T+']
            c3 = new_feats[traj][new_feats['ATN'] == 'A-T-']
            _, pval3 = ttest_ind(c0, c3, nan_policy='omit')

            print(' Con vs MCI: p:%.2e' % pval1)
            print(' Con vs Dementia: p:%.2e ' % pval2)
            print(' A+T+ vs A-T-: p:%.2e' % pval3)


if __name__ == "__main__":

    main()
