import numpy as np
import pandas as pd

from collections import Counter

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import ProtParamData
from Bio.SeqUtils.ProtParam import ProteinAnalysis


from sklearn.metrics import make_scorer, precision_score, recall_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_validate
import pickle


import utils
from utils import aminos as aminos_dict



cabecalho = ['seq', 'len', 'R', 'normR', 'K', 'normK', 'Q', 'normQ', 'A',  'normA',
             'L', 'normL', 'W', 'normW', 'mass', 'isoelectric_point', 'net_charge',
             'gravy', 'hidro/total',
             'AA', 'normAA', 'AC', 'normAC', 'AD', 'normAD', 'AE', 'normAE', 'AF', 'normAF', 'AG', 'normAG', 'AH', 'normAH', 'AI', 'normAI', 'AK', 'normAK', 'AL', 'normAL', 'AM', 'normAM', 'AN', 'normAN', 'AP', 'normAP', 'AQ', 'normAQ', 'AR', 'normAR', 'AS', 'normAS', 'AT', 'normAT', 'AV', 'normAV', 'AW', 'normAW', 'AY', 'normAY',
             'CA', 'normCA', 'CC', 'normCC', 'CD', 'normCD', 'CE', 'normCE', 'CF', 'normCF', 'CG', 'normCG', 'CH', 'normCH', 'CI', 'normCI', 'CK', 'normCK', 'CL', 'normCL', 'CM', 'normCM', 'CN', 'normCN', 'CP', 'normCP', 'CQ', 'normCQ', 'CR', 'normCR', 'CS', 'normCS', 'CT', 'normCT', 'CV', 'normCV', 'CW', 'normCW', 'CY', 'normCY',
             'DA', 'normDA', 'DC', 'normDC', 'DD', 'normDD', 'DE', 'normDE', 'DF', 'normDF', 'DG', 'normDG', 'DH', 'normDH', 'DI', 'normDI', 'DK', 'normDK', 'DL', 'normDL', 'DM', 'normDM', 'DN', 'normDN', 'DP', 'normDP', 'DQ', 'normDQ', 'DR', 'normDR', 'DS', 'normDS', 'DT', 'normDT', 'DV', 'normDV', 'DW', 'normDW', 'DY', 'normDY',
             'EA', 'normEA', 'EC', 'normEC', 'ED', 'normED', 'EE', 'normEE', 'EF', 'normEF', 'EG', 'normEG', 'EH', 'normEH', 'EI', 'normEI', 'EK', 'normEK', 'EL', 'normEL', 'EM', 'normEM', 'EN', 'normEN', 'EP', 'normEP', 'EQ', 'normEQ', 'ER', 'normER', 'ES', 'normES', 'ET', 'normET', 'EV', 'normEV', 'EW', 'normEW', 'EY', 'normEY',
             'FA', 'normFA', 'FC', 'normFC', 'FD', 'normFD', 'FE', 'normFE', 'FF', 'normFF', 'FG', 'normFG', 'FH', 'normFH', 'FI', 'normFI', 'FK', 'normFK', 'FL', 'normFL', 'FM', 'normFM', 'FN', 'normFN', 'FP', 'normFP', 'FQ', 'normFQ', 'FR', 'normFR', 'FS', 'normFS', 'FT', 'normFT', 'FV', 'normFV', 'FW', 'normFW', 'FY', 'normFY',
             'GA', 'normGA', 'GC', 'normGC', 'GD', 'normGD', 'GE', 'normGE', 'GF', 'normGF', 'GG', 'normGG', 'GH', 'normGH', 'GI', 'normGI', 'GK', 'normGK', 'GL', 'normGL', 'GM', 'normGM', 'GN', 'normGN', 'GP', 'normGP', 'GQ', 'normGQ', 'GR', 'normGR', 'GS', 'normGS', 'GT', 'normGT', 'GV', 'normGV', 'GW', 'normGW', 'GY', 'normGY',
             'HA', 'normHA', 'HC', 'normHC', 'HD', 'normHD', 'HE', 'normHE', 'HF', 'normHF', 'HG', 'normHG', 'HH', 'normHH', 'HI', 'normHI', 'HK', 'normHK', 'HL', 'normHL', 'HM', 'normHM', 'HN', 'normHN', 'HP', 'normHP', 'HQ', 'normHQ', 'HR', 'normHR', 'HS', 'normHS', 'HT', 'normHT', 'HV', 'normHV', 'HW', 'normHW', 'HY', 'normHY',
             'IA', 'normIA', 'IC', 'normIC', 'ID', 'normID', 'IE', 'normIE', 'IF', 'normIF', 'IG', 'normIG', 'IH', 'normIH', 'II', 'normII', 'IK', 'normIK', 'IL', 'normIL', 'IM', 'normIM', 'IN', 'normIN', 'IP', 'normIP', 'IQ', 'normIQ', 'IR', 'normIR', 'IS', 'normIS', 'IT', 'normIT', 'IV', 'normIV', 'IW', 'normIW', 'IY', 'normIY',
             'KA', 'normKA', 'KC', 'normKC', 'KD', 'normKD', 'KE', 'normKE', 'KF', 'normKF', 'KG', 'normKG', 'KH', 'normKH', 'KI', 'normKI', 'KK', 'normKK', 'KL', 'normKL', 'KM', 'normKM', 'KN', 'normKN', 'KP', 'normKP', 'KQ', 'normKQ', 'KR', 'normKR', 'KS', 'normKS', 'KT', 'normKT', 'KV', 'normKV', 'KW', 'normKW', 'KY', 'normKY',
             'LA', 'normLA', 'LC', 'normLC', 'LD', 'normLD', 'LE', 'normLE', 'LF', 'normLF', 'LG', 'normLG', 'LH', 'normLH', 'LI', 'normLI', 'LK', 'normLK', 'LL', 'normLL', 'LM', 'normLM', 'LN', 'normLN', 'LP', 'normLP', 'LQ', 'normLQ', 'LR', 'normLR', 'LS', 'normLS', 'LT', 'normLT', 'LV', 'normLV', 'LW', 'normLW', 'LY', 'normLY',
             'MA', 'normMA', 'MC', 'normMC', 'MD', 'normMD', 'ME', 'normME', 'MF', 'normMF', 'MG', 'normMG', 'MH', 'normMH', 'MI', 'normMI', 'MK', 'normMK', 'ML', 'normML', 'MM', 'normMM', 'MN', 'normMN', 'MP', 'normMP', 'MQ', 'normMQ', 'MR', 'normMR', 'MS', 'normMS', 'MT', 'normMT', 'MV', 'normMV', 'MW', 'normMW', 'MY', 'normMY',
             'NA', 'normNA', 'NC', 'normNC', 'ND', 'normND', 'NE', 'normNE', 'NF', 'normNF', 'NG', 'normNG', 'NH', 'normNH', 'NI', 'normNI', 'NK', 'normNK', 'NL', 'normNL', 'NM', 'normNM', 'NN', 'normNN', 'NP', 'normNP', 'NQ', 'normNQ', 'NR', 'normNR', 'NS', 'normNS', 'NT', 'normNT', 'NV', 'normNV', 'NW', 'normNW', 'NY', 'normNY',
             'PA', 'normPA', 'PC', 'normPC', 'PD', 'normPD', 'PE', 'normPE', 'PF', 'normPF', 'PG', 'normPG', 'PH', 'normPH', 'PI', 'normPI', 'PK', 'normPK', 'PL', 'normPL', 'PM', 'normPM', 'PN', 'normPN', 'PP', 'normPP', 'PQ', 'normPQ', 'PR', 'normPR', 'PS', 'normPS', 'PT', 'normPT', 'PV', 'normPV', 'PW', 'normPW', 'PY', 'normPY',
             'QA', 'normQA', 'QC', 'normQC', 'QD', 'normQD', 'QE', 'normQE', 'QF', 'normQF', 'QG', 'normQG', 'QH', 'normQH', 'QI', 'normQI', 'QK', 'normQK', 'QL', 'normQL', 'QM', 'normQM', 'QN', 'normQN', 'QP', 'normQP', 'QQ', 'normQQ', 'QR', 'normQR', 'QS', 'normQS', 'QT', 'normQT', 'QV', 'normQV', 'QW', 'normQW', 'QY', 'normQY',
             'RA', 'normRA', 'RC', 'normRC', 'RD', 'normRD', 'RE', 'normRE', 'RF', 'normRF', 'RG', 'normRG', 'RH', 'normRH', 'RI', 'normRI', 'RK', 'normRK', 'RL', 'normRL', 'RM', 'normRM', 'RN', 'normRN', 'RP', 'normRP', 'RQ', 'normRQ', 'RR', 'normRR', 'RS', 'normRS', 'RT', 'normRT', 'RV', 'normRV', 'RW', 'normRW', 'RY', 'normRY',
             'SA', 'normSA', 'SC', 'normSC', 'SD', 'normSD', 'SE', 'normSE', 'SF', 'normSF', 'SG', 'normSG', 'SH', 'normSH', 'SI', 'normSI', 'SK', 'normSK', 'SL', 'normSL', 'SM', 'normSM', 'SN', 'normSN', 'SP', 'normSP', 'SQ', 'normSQ', 'SR', 'normSR', 'SS', 'normSS', 'ST', 'normST', 'SV', 'normSV', 'SW', 'normSW', 'SY', 'normSY',
             'TA', 'normTA', 'TC', 'normTC', 'TD', 'normTD', 'TE', 'normTE', 'TF', 'normTF', 'TG', 'normTG', 'TH', 'normTH', 'TI', 'normTI', 'TK', 'normTK', 'TL', 'normTL', 'TM', 'normTM', 'TN', 'normTN', 'TP', 'normTP', 'TQ', 'normTQ', 'TR', 'normTR', 'TS', 'normTS', 'TT', 'normTT', 'TV', 'normTV', 'TW', 'normTW', 'TY', 'normTY',
             'VA', 'normVA', 'VC', 'normVC', 'VD', 'normVD', 'VE', 'normVE', 'VF', 'normVF', 'VG', 'normVG', 'VH', 'normVH', 'VI', 'normVI', 'VK', 'normVK', 'VL', 'normVL', 'VM', 'normVM', 'VN', 'normVN', 'VP', 'normVP', 'VQ', 'normVQ', 'VR', 'normVR', 'VS', 'normVS', 'VT', 'normVT', 'VV', 'normVV', 'VW', 'normVW', 'VY', 'normVY',
             'WA', 'normWA', 'WC', 'normWC', 'WD', 'normWD', 'WE', 'normWE', 'WF', 'normWF', 'WG', 'normWG', 'WH', 'normWH', 'WI', 'normWI', 'WK', 'normWK', 'WL', 'normWL', 'WM', 'normWM', 'WN', 'normWN', 'WP', 'normWP', 'WQ', 'normWQ', 'WR', 'normWR', 'WS', 'normWS', 'WT', 'normWT', 'WV', 'normWV', 'WW', 'normWW', 'WY', 'normWY',
             'YA', 'normYA', 'YC', 'normYC', 'YD', 'normYD', 'YE', 'normYE', 'YF', 'normYF', 'YG', 'normYG', 'YH', 'normYH', 'YI', 'normYI', 'YK', 'normYK', 'YL', 'normYL', 'YM', 'normYM', 'YN', 'normYN', 'YP', 'normYP', 'YQ', 'normYQ', 'YR', 'normYR', 'YS', 'normYS', 'YT', 'normYT', 'YV', 'normYV', 'YW', 'normYW', 'YY', 'normYY',
             'nC', 'normnC', 'nH', 'normnH', 'nN', 'normnN', 'nO', 'normnO', 'nS', 'normnS', 'total_elements', 'label']



##############################################################################################################################################
########################################################### FEATURE MATRIX ####################################################################
##############################################################################################################################################

def calc_properties(file_path):
    sequence_list = set()

    for record in SeqIO.parse(file_path, 'fasta'):
            sequence_list.add(record.seq)


    list_len = len(sequence_list)
    matrix = np.zeros((list_len, 830), dtype=object) 
    scale = ProtParamData.gravy_scales.get('KyteDoolitle')

    for i, seq in enumerate(sequence_list):
        sequence = ProteinAnalysis(seq)

        matrix[i][0] = str(sequence.sequence)  
        matrix[i][1] = sequence.length   

        sequence.count_amino_acids()
        for j, amino in enumerate(['R', 'K', 'Q', 'A', 'L', 'W']):
            matrix[i][(j+1)*2] = sequence.amino_acids_content[amino]                
            matrix[i][(j+1)*2 + 1] = sequence.amino_acids_content[amino] / sequence.length

        matrix[i][14] = sequence.molecular_weight()   # mass
        matrix[i][15] = sequence.isoelectric_point()  # isoelectric point
        matrix[i][16] = sequence.charge_at_pH(7.0)    # charge

        gravy = sequence.gravy() 
        matrix[i][17] = gravy # Kyte-DooLittle

        hydrophilic_residues = 0
        for a in sequence.sequence:
            if scale[a] < 0:
                hydrophilic_residues += 1

        matrix[i][18] = hydrophilic_residues / sequence.length

        # dipeptide composition
        plus_one = (1 / (sequence.length - 2 + 1))
        for j in range(sequence.length - 2 + 1):
            matrix[i][(utils.aminos[sequence.sequence[j]]['id'] * 20 + utils.aminos[sequence.sequence[j+1]]['id'] ) * 2 + 19] += 1
            matrix[i][(utils.aminos[sequence.sequence[j]]['id'] * 20 + utils.aminos[sequence.sequence[j+1]]['id'])* 2 + 19 + 1] += plus_one

        for amino, count in sequence.amino_acids_content.items():
            matrix[i][819] += aminos_dict[amino]['nC'] * count
            matrix[i][821] += aminos_dict[amino]['nH'] * count
            matrix[i][823] += aminos_dict[amino]['nN'] * count
            matrix[i][825] += aminos_dict[amino]['nO'] * count
            matrix[i][827] += aminos_dict[amino]['nS'] * count

        matrix[i][821] -= 2 * (sequence.length - 1) # H2
        matrix[i][825] -= sequence.length - 1 # 0

        total_elements = np.sum(matrix[i,819:827+1])
        matrix[i][829] = total_elements
        for j in range(5):
            matrix[i][820 + (j * 2)] = matrix[i][819 + (j * 2)] / total_elements

    return matrix

############################################################################################################################################



##############################################################################################################################################
########################################################### MATRIX GENERATION ####################################################################
##############################################################################################################################################


def independent_matrix(file_path):
    independent_matrix = calc_properties(file_path)
    df_independet = pd.DataFrame(independent_matrix, columns=cabecalho).infer_objects()

    df_independet.to_csv('df_independet.csv', index=False)
    negatives, positives = df_independet['label'].value_counts()
    print(f'The independent Matrix was generated.\nPositives Samples: {positives}, Nagative Samples: {negatives}')

############################################################################################################################################





##############################################################################################################################################
########################################################### Model Running ####################################################################
##############################################################################################################################################

def run_model(cla, eff, file_path):
    model = pickle.load(open(cla, 'rb'))
    model1 = pickle.load(open(eff, 'rb'))


    independente_dataset = calc_properties(file_path)
    y_probs = model.predict_proba(independente_dataset[:, 1:])[:, 1]
    y_pred = ['CPP' if p > 0.5 else 'non-CPP' for p in y_probs]


    y_probs1 = model1.predict_proba(independente_dataset[:, 1:])[:, 1]
    y_pred1 = ['High' if p > 0.5 else 'Low' for p in y_probs1]


    D = pd.DataFrame()
    D['seq'] = independente_dataset[:, 0]
    D['Classification'], D['percent'] = y_pred, y_probs
    D['Efficiency'], D['percent'] = y_pred1, y_probs1
    D.loc[D['Classification'] == 'non-CPP', ['Efficiency', 'percent']] = '-'


    D.to_csv('predict.csv', index=False)
##############################################################################################################################################





model_stage1_path = 'trained_model.pkl'
model_stage2_path = 'efficiency_model.pkl'
independent_dataset = 'Layer1_Independent.fasta' #path of your independent seqs FASTA file

run_model(model_stage1_path, model_stage2_path, independent_dataset)


