'''
ADNIData.py v1.0
Rafi Haque rafihaque90@gmail.com
--------------------
Description:
--------------------
Instructions:
'''

import argparse
import os
import pdb
import pickle
import time
from itertools import compress

import h2o
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML
from scipy.stats import ttest_ind


class ADNIData():
    def __init__(self, merge_path='ADNI/ADNIMERGE.csv', feats_path='ADNI/JL_ADNIMERGE_64peptides_APOEpeptides_rm4SDoutliers_LastVisit.csv', trajs=None, bls=None, clis=None, tras=None):

        # TODO
        # move get_ATN and get_ratio to main

        # load merge and peptide file
        self.merge_path = merge_path
        self.feats_path = feats_path

        print('Loading ADNI dataset...')
        if merge_path is None or not os.path.isfile(merge_path):
            raise RuntimeError(
                'There is no such file %s! Provide a valid feats path.' % merge_path)
        if feats_path is None or not os.path.isfile(feats_path):
            raise RuntimeError(
                'There is no such file %s! Provide a valid traits path.' % feats_path)

        # load feats and merge into pandas dataframe
        self.merge = pd.read_csv(
            merge_path, sep=',', engine='python', header=0, index_col=0)
        self.feats = pd.read_csv(
            feats_path, sep=',', engine='python', header=0, index_col=0)
        self.feats.rename(
            columns={'PKM2;LFEELVR': 'sp|dummy|PKM2_HUMAN;LFEELVR'}, inplace=True)

        # change values due to limit of detection
        # self.merge['ABETA'][self.merge['ABETA']=='>1700']=1700
        self.merge['ABETA'][self.merge['ABETA'] == '>1700'] = np.nan
        self.merge['ABETA'][self.merge['ABETA'] == 1700] = np.nan

        self.merge['ABETA'][self.merge['ABETA'] == '<200'] = 200
        self.merge['PTAU'][self.merge['PTAU'] == '<8'] = 8
        self.merge['TAU'][self.merge['TAU'] == '<80'] = 80
        self.merge['TAU'][self.merge['TAU'] == '>1300'] = 1300
        # self.merge['ABETA_bl'][self.merge['ABETA_bl']=='>1700']=1700

        self.merge['ABETA_bl'][self.merge['ABETA_bl'] == '>1700'] = np.nan
        # self.merge['ABETA_bl'][self.merge['ABETA_bl']==1700]=np.nan
        self.merge['PTAU_bl'][self.merge['PTAU_bl'] == '<8'] = 8
        self.merge['TAU_bl'][self.merge['TAU_bl'] == '<80'] = 80

        # ['e33', 'e34', 'e23', 'e44', 'e24', 'e22']

        self.merge['APOE_geno1'] = [
            int(st.split('e')[1][0])-3 for st in self.merge['APOE_geno'].tolist()]
        self.merge['APOE_geno2'] = [
            int(st.split('e')[1][1])-3 for st in self.merge['APOE_geno'].tolist()]

        self.merge['APOE_geno'] = self.merge['APOE_geno'].factorize()[0]
        self.merge['PTGENDER'] = self.merge['PTGENDER'].factorize()[0]

        self.RIDS = self.feats.index.tolist()

        # store trajectories and baseline values of interest
        self.bls = bls
        self.clis = clis
        self.trajs = trajs
        self.tras = tras

        # rename proteins to shorthand

        prots = [prot for prot in self.feats.columns if '|' in prot]

        shorts = np.array([feat.split('|')[2].split('_')[0]
                          for feat in prots]).astype('<U19')
        self.new_prots = np.array(prots).copy()

        for short in shorts:
            self.new_prots[shorts == short] = [
                uid + '|' + str(count) for count, uid in enumerate(shorts[shorts == short])]
        names = {prots[i]: self.new_prots[i]
                 for i in range(len(self.new_prots))}
        self.feats.rename(columns={prots[i]: self.new_prots[i]
                          for i in range(len(self.new_prots))}, inplace=True)

        # initialize

        self.feats[self.trajs] = np.NaN
        self.feats[self.bls] = np.NaN
        self.feats[[traj + '_nv' for traj in self.trajs]] = np.NaN
        self.feats[[traj + '_lv' for traj in self.trajs]] = np.NaN
        self.feats['batch'] = self.feats['batch'].astype('string').factorize()[
            0]

        self.bnames = ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8']
        self.feats[self.bnames] = pd.get_dummies(
            self.feats['batch'], columns=self.bnames)

    # load feats
    def load(self, obj_path):
        return pickle.load(open(obj_path, 'rb'))

    # save feats
    def save(self, obj_path):
        pickle.dump(self, open(obj_path, 'wb'))
        return self

    # loop through participants and get baseline values from adni merge file
    def get_base(self):
        print('BASE')
        for RID in self.RIDS:
            bRID = self.merge.index == RID
            tmp = self.merge.iloc[bRID][self.bls+['VISCODE']]
            self.feats.at[self.feats.index == RID, self.bls] = tmp[tmp['VISCODE']
                                                                   == 'bl'][self.bls].astype('float64').iloc[0, :].to_numpy()
            self.feats.at[self.feats.index == RID,
                          self.tras] = self.merge.iloc[bRID][self.tras].iloc[0, :].to_list()
            self.feats.at[self.feats.index == RID,
                          self.clis] = self.merge.iloc[bRID][self.clis].iloc[0, :].to_list()

        # add av45

        # derive ratio (not in original merge file)
        self.feats['PTAU_bl/ABETA_bl'] = self.feats['PTAU_bl'] / \
            self.feats['ABETA_bl']

        # factorize gender

        # add ATN framework
        ttau_thresh = 245
        amy = self.feats['ABETA_bl'] < 980
        ptau = self.feats['PTAU_bl'] > 21.8
        ttau = self.feats['TAU_bl'] > 245
        av45 = self.feats['AV45'] > 1.2
        self.feats['ATN'] = np.nan
        self.feats['ATN'][amy & ptau] = 'A+T+'
        self.feats['ATN'][~amy & ptau] = 'A-T+'
        self.feats['ATN'][amy & ~ptau] = 'A+T-'
        self.feats['ATN'][~amy & ~ptau] = 'A-T-'
        self.feats['A'] = np.nan
        self.feats['A'][amy] = 'A+'
        self.feats['A'][~amy] = 'A-'
        self.feats['P'] = np.nan
        self.feats['P'][ptau] = 'P+'
        self.feats['P'][~ptau] = 'P-'
        self.feats['T'] = np.nan
        self.feats['T'][ttau] = 'T+'
        self.feats['T'][~ttau] = 'T-'
        self.feats['AV'] = np.nan
        self.feats['AV'][av45] = 'AV+'
        self.feats['AV'][~av45] = 'AV-'

        # # # correct baseline for volumes
        # hipp = self.feats['Hippocampus_bl']
        # gen = self.feats['PTGENDER']==1
        # m1 = np.mean(hipp[gen])
        # s1 = np.std(hipp[gen])
        # self.feats['Hippocampus_bl'][gen] = (hipp-m1)/s1

        # gen = self.feats['PTGENDER']==0
        # m1 = np.mean(hipp[gen])
        # s1 = np.std(hipp[gen])
        # self.feats['Hippocampus_bl'][gen] = (hipp-m1)/s1

        return self

    # normalize all values in merge file to baseline
    def norm_merge(self):
        mfeats = self.feats[self.bls].mean(axis=0)
        sfeats = self.feats[self.bls].std(axis=0)
        for count, traj in enumerate(self.trajs):
            self.merge[traj] = (self.merge[traj].astype(
                'float64')-mfeats[traj+'_bl'])/sfeats[traj+'_bl']
        return self

    # get trajectories
    def get_trajs(self):

        # loop through each participant and get trajectory
        self.xy = pd.DataFrame(
            columns=['Years_bl', 'yact', 'ypred', 'traj', 'RID', 'DX_bl_new'], index=[])
        count = 0
        for rcount, RID in enumerate(self.RIDS):
            # get merge and feats entries
            bRID = self.merge.index == RID
            fRID = self.feats.index == RID

            # get years
            x = self.merge.iloc[bRID]['Years_bl'].astype('float64')

            # loop through each trajectory
            for tcount, traj in enumerate(self.trajs):

                # get trajectory
                y = self.merge.iloc[bRID][traj].astype('float64')

                # require two visits
                if (sum(~np.isnan(y)) > 2):

                    # calculate slope after removing entries without value
                    m, b = np.polyfit(x[~np.isnan(y)], y[~np.isnan(y)], 1)

                    # store variables
                    xy = self.merge.iloc[bRID][['Years_bl']].astype('float64')[
                        ~np.isnan(y)]
                    xy['yact'] = y[~np.isnan(y)]
                    xy['ypred'] = xy['Years_bl']*m+b
                    xy['traj'] = traj
                    xy['RID'] = xy.index
                    xy['DX_bl_new'] = self.feats['DX_bl_new'][fRID]

                    self.feats.at[fRID, traj +
                                  '_lv'] = xy.sort_values('Years_bl')['Years_bl'].iloc[-1]
                    self.feats.at[fRID, traj+'_nv'] = len(xy)
                    self.feats.at[fRID, traj] = m

                    # add to previous
                    self.xy = pd.concat([self.xy, xy.sort_values('Years_bl')])

        self.xy.index = range(0, len(self.xy))
        self.ds = self.feats[self.trajs].mean()/self.feats[self.trajs].std()
        return self

    # retain rows with columns 1, 2, or n with conditions
    def or_filter(self, or_columns, or_conds):
        pmask = np.zeros(self.feats.shape[0], dtype=bool)
        for name, cond in zip(or_columns, or_conds):
            pmask = pmask | (self.feats[name] == cond).to_numpy()

        self.feats = self.feats.iloc[pmask, :]
        return self

    # retain rows with columns conditions 1, 2, and n
    def and_filter(self, and_names, and_conds):
        pmask = np.ones(self.feats.shape[0], dtype=bool)
        for name, cond in zip(and_names, and_conds):
            pmask = pmask & (self.feats[name] == cond).to_numpy()
        self.feats = self.feats.iloc[pmask, :]
        return self

    # retain rows meeting p-value criteria
    def p_filter(self, name, alpha):

        # loop through features and get p-value
        self.pval = np.empty(len(self.feats.columns))
        self.pval[:] = np.NaN

        y = self.get_categorical(name)
        for count, feat in enumerate(self.feats.columns):
            _, self.pval[count] = ttest_ind(
                self.feats[feat][y == 0], self.feats[feat][y == 1], nan_policy='omit')
        self.feats = self.feats[list(
            compress(self.feats.columns.to_list(), self.pval < alpha))]
        return self

    def get_categorical(self, name):
        return self.feats[name].factorize()[0]

    def get_y(self, name, cats=None, nums=None):

        feats = self.feats.copy()
        if nums is not None:
            for cat, num in zip(cats, nums):
                feats[name][feats[name] == cat] = num
        else:
            feats[name] = self.feats[name].factorize()[0]
        self.y = feats[name].to_numpy().astype('int64')
        return feats[name].to_numpy().astype('int64')
