import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# font = {'family' : 'FreeSans',
#         'size'   : 48}
#
# plt.rc('font', **font)

sns.set_style('white')

all_ppis = {'nhe-azz': 16.659209551918693, 'nhe-nch': 4.356470093581868, 'nhe-ncj': 13.863770610897575,
            'nhe-ncl': 21.51800271412455, 'nhe-ngu': 15.599893126370015, 'nhe-nhe': 2.1191635407653204,
            'nhe-nhi': 17.974917883257277, 'nhe-nhw': 3.1191545605989663, 'nhe-nhx': 36.4545400862181,
            'nhe-nhy': 15.90981590759542, 'nhe-nsu': 16.030921312782297, 'nch-azz': 13.880806873910448,
            'nch-nch': 2.136231384833878, 'nch-ncj': 22.705414511436796, 'nch-ncl': 18.677735955237953,
            'nch-ngu': 22.608100470975522, 'nch-nhe': 5.622704707526978, 'nch-nhi': 28.325659634651068,
            'nch-nhw': 4.054825437943866, 'nch-nhx': 33.90414701667625, 'nch-nhy': 21.83723938806456,
            'nch-nsu': 27.14836325826746, 'nhi-azz': 11.825196348840834, 'nhi-nch': 13.968741359805351,
            'nhi-ncj': 7.594635865057859, 'nhi-ncl': 19.9978737480027, 'nhi-ngu': 9.54944105977556,
            'nhi-nhe': 10.307485880675372, 'nhi-nhi': 2.2813732122228636, 'nhi-nhw': 10.9860902190821,
            'nhi-nhx': 31.4331902522845, 'nhi-nhy': 7.461645780552921, 'nhi-nsu': 10.429831357220882,
            'ncj-azz': 10.515042753243645, 'ncj-nch': 11.930557647833515, 'ncj-ncj': 2.2117233336105206,
            'ncj-ncl': 16.94935711169339, 'ncj-ngu': 10.247876550673876, 'ncj-nhe': 9.005190446369706,
            'ncj-nhi': 10.10503530127, 'ncj-nhw': 9.526941075872983, 'ncj-nhx': 36.47100488855389,
            'ncj-nhy': 9.955424859052277, 'ncj-nsu': 13.297488974686775, 'azz-azz': 2.0471409673608663,
            'azz-nch': 8.98147282772508, 'azz-ncj': 11.97350202197088, 'azz-ncl': 16.318559069765204,
            'azz-ngu': 14.927407272688326, 'azz-nhe': 13.58171513332911, 'azz-nhi': 15.114011325554898,
            'azz-nhw': 12.558202215402853, 'azz-nhx': 33.49302326780349, 'azz-nhy': 13.049877585204937,
            'azz-nsu': 19.963300473403084, 'ncl-azz': 17.756053230348396, 'ncl-nch': 14.885593139608632,
            'ncl-ncj': 18.826456550366004, 'ncl-ncl': 2.1328530665342114, 'ncl-ngu': 27.329534542410386,
            'ncl-nhe': 18.206020316865857, 'ncl-nhi': 26.890626641819058, 'ncl-nhw': 17.272513086115108,
            'ncl-nhx': 40.49935117869887, 'ncl-nhy': 24.822857566590464, 'ncl-nsu': 31.440812176300017,
            'nhx-azz': 21.448196720032595, 'nhx-nch': 17.50624007094231, 'nhx-ncj': 31.81757494008169,
            'nhx-ncl': 24.231898848102606, 'nhx-ngu': 22.036510463763527, 'nhx-nhe': 24.492306160215584,
            'nhx-nhi': 31.81436860057594, 'nhx-nhw': 21.898796064178665, 'nhx-nhx': 2.3623274956027998,
            'nhx-nhy': 22.98517592117037, 'nhx-nsu': 26.922438443409963, 'nsu-azz': 14.825766181532941,
            'nsu-nch': 15.443188911380863, 'nsu-ncj': 11.102058295572522, 'nsu-ncl': 21.763731528728503,
            'nsu-ngu': 10.37001251351462, 'nsu-nhe': 10.758900531574335, 'nsu-nhi': 11.365345178622169,
            'nsu-nhw': 11.600405374164446, 'nsu-nhx': 35.53212594141149, 'nsu-nhy': 5.968845982231199,
            'nsu-nsu': 2.2300941708918747, 'nhy-azz': 10.52523912123796, 'nhy-nch': 13.648757246851156,
            'nhy-ncj': 9.58132268282178, 'nhy-ncl': 19.681878935273506, 'nhy-ngu': 8.347994773852562,
            'nhy-nhe': 12.392224447059446, 'nhy-nhi': 8.883398054486992, 'nhy-nhw': 11.217322502457844,
            'nhy-nhx': 30.584407909879356, 'nhy-nhy': 2.1221921867433235, 'nhy-nsu': 7.392768058756977,
            'ngu-azz': 13.126322849355976, 'ngu-nch': 13.03233816008215, 'ngu-ncj': 9.842551353144236,
            'ngu-ncl': 20.988377123135514, 'ngu-ngu': 2.207920118914296, 'ngu-nhe': 10.215859357625273,
            'ngu-nhi': 10.767741893509173, 'ngu-nhw': 11.020911520813993, 'ngu-nhx': 27.67772509603299,
            'ngu-nhy': 8.050716976989197, 'ngu-nsu': 11.593782681712325, 'nhw-azz': 17.957789556168088,
            'nhw-nch': 3.566514621879545, 'nhw-ncj': 16.188396674853173, 'nhw-ncl': 21.879177464259673,
            'nhw-ngu': 17.095185359505653, 'nhw-nhe': 3.071794482946706, 'nhw-nhi': 21.507067685447396,
            'nhw-nhw': 2.105767922075722, 'nhw-nhx': 39.40637788714304, 'nhw-nhy': 17.73905291424692,
            'nhw-nsu': 18.76501918936717}

#
# Below is the old data that included nhx.
#
# labels = ['azz-ncj', 'azz-nhe', 'azz-nhw', 'azz-nhx', 'azz-ncl', 'ncj-azz', 'ncj-nhe',
#           'ncj-nhw', 'ncj-ncl', 'nhe-ncj', 'nhe-nhw', 'nhi-azz', 'nhi-ncj', 'nhi-nhw', 'nhw-ncj',
#           'nhw-nhe', 'nhy-azz', 'nhy-ncj', 'nch-nhe', 'nch-nhw', 'nhi-nsu', 'nhi-ncj']
#
# scores = [[62], [50, 59], [47], [41], [53], [51, 36], [40, 38], [62, 60], [58], [56],
#           [92, 98, 98], [41], [77], [58], [49], [97, 98, 94, 99, 90, 89, 94], [48], [75],
#           [91, 90], [91], [55], [55]]

labels = ['azz-ncj', 'azz-nhe', 'azz-nhw', 'azz-ncl', 'ncj-azz', 'ncj-nhe',
          'ncj-nhw', 'ncj-ncl', 'nhe-ncj', 'nhe-nhw', 'nhi-azz', 'nhi-ncj', 'nhi-nhw', 'nhw-ncj',
          'nhw-nhe', 'nhy-azz', 'nhy-ncj', 'nch-nhe', 'nch-nhw', 'nhi-nsu', 'nhi-ncj']

scores = [[62], [50, 59], [47], [53], [51, 36], [40, 38], [62, 60], [58], [56],
          [92, 98, 98], [41], [77], [58], [49], [97, 98, 94, 99, 90, 89, 94], [48], [75],
          [91, 90], [91], [55], [55]]

ppis = [all_ppis[l] for l in labels]
print([np.mean(s) for s in scores])
print(ppis)
df = pd.DataFrame({"ppi": ppis, "mutint": [np.mean(s) for s in scores], "label": labels})

sns.set_context('paper')
plt.figure(figsize=(14, 6))
p1 = sns.scatterplot('ppi',  # Horizontal axis
                     'mutint',  # Vertical axis
                     data=df, ci=None,
                     # , # Data source
                     # size = 20,
                     # legend=False,
                     color='black', s=60, alpha=0.5)

for line in range(0, df.shape[0]):
    p1.text(df.ppi[line] + 0.3, df.mutint[line] - 0.1,
            df.label[line], horizontalalignment='left',
            size=14, color='black')

# sn.regplot(ppis, mutint_nums)
# plt.xticks(range(2, 50, 2))

plt.xlabel("Cross-variant perplexity", size=20)
plt.yticks(range(35, 100, 5), size=14)
plt.xticks(range(0, 30, 5), size=14)
plt.ylabel('Reported % mutual intelligibility', size=20)
plt.grid(True)
plt.savefig("mutint_scatter_no_nhx_14x6.pdf")