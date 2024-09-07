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
from sklearn.metrics import roc_auc_score, confusion_matrix
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


import utils
from utils import aminos as aminos_dict
from scipy.stats import pearsonr



scoring = {
    'ACC':  'accuracy',
    'PR':   'precision',
    '-PR':  make_scorer(precision_score, pos_label=0),
    'SE':   'recall',
    'SP':   make_scorer(recall_score, pos_label=0),
    'F1':   'f1',
    'AUC':  'roc_auc',
    'MCC': 'matthews_corrcoef'
    }



cabecalho = ['seq', 'len', 'R', 'normR', 'K', 'normK', 'A',  'normA',
             'L', 'normL', 'G', 'normG', 'C', 'normC', 'W', 'normW','P', 'normP', 'H', 'normH','mass', 'isoelectric_point', 'net_charge',
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

scoring = {
    'ACC':  'accuracy',
    'PR':   'precision',
    '-PR':  make_scorer(precision_score, pos_label=0),
    'SE':   'recall',
    'SP':   make_scorer(recall_score, pos_label=0),
    'F1':   'f1',
    'AUC':  'roc_auc',
    'MCC': 'matthews_corrcoef'
    }



##############################################################################################################################################
########################################################### FEATURE MATRIX ####################################################################
##############################################################################################################################################

def calc_properties(file_path, label: int):
    sequence_list = set()
    unwanted_chars = {'*', '-', 'X', 'O', 'U', 'Z', 'B', 'J', 'u'}
    
    if file_path.split('.')[-1] == 'fasta':
        for record in SeqIO.parse(file_path, 'fasta'):
            if any(char in record.seq for char in unwanted_chars):
                continue
            sequence_list.add(record.seq)

    else:
        df = pd.read_csv(file_path)
        for seq in df.iloc[:, 0]:  
            if any(char in seq for char in unwanted_chars):
                continue
            sequence_list.add(seq)
                
    list_len = len(sequence_list)
    matrix = np.zeros((list_len, 837), dtype=object) # zeros para somar e object para colocar a string

    scale = ProtParamData.gravy_scales.get('KyteDoolitle')

    for i, seq in enumerate(sequence_list):
        sequence = ProteinAnalysis(seq)
        matrix[i][0] = str(sequence.sequence)  # string da sequência
        matrix[i][1] = sequence.length    # tamanho da sequência
        sequence.count_amino_acids()

        for j, amino in enumerate(['R', 'K', 'A', 'L', 'G', 'C', 'W', 'P', 'H']): # indices 2 ao 19
            matrix[i][(j+1)*2] = sequence.amino_acids_content[amino]                        # quantidade do aminoacido x
            matrix[i][(j+1)*2 + 1] = sequence.amino_acids_content[amino] / sequence.length  # norm do aminoacido x

        matrix[i][20] = sequence.molecular_weight()   # massa
        matrix[i][21] = sequence.isoelectric_point()  # ponto isoelétrico
        matrix[i][22] = sequence.charge_at_pH(7.0)    # carga líquida

        gravy = sequence.gravy() # hidrofobicidade
        matrix[i][23] = gravy # escala de Kyte-DooLittle

        # nessa escala, residuos hidrofilicos recebem valor negativo
        hydrophilic_residues = 0
        for a in sequence.sequence:
            if scale[a] < 0:
                hydrophilic_residues += 1

        matrix[i][24] = hydrophilic_residues / sequence.length

        # dipeptide composition
        plus_one = (1 / (sequence.length - 2 + 1))
        for j in range(sequence.length - 2 + 1):
            matrix[i][(utils.aminos[sequence.sequence[j]]['id'] * 20 + utils.aminos[sequence.sequence[j+1]]['id'] ) * 2 + 19] += 1
            matrix[i][(utils.aminos[sequence.sequence[j]]['id'] * 20 + utils.aminos[sequence.sequence[j+1]]['id'])* 2 + 19 + 1] += plus_one

        for amino, count in sequence.amino_acids_content.items():
            matrix[i][825] += aminos_dict[amino]['nC'] * count
            matrix[i][827] += aminos_dict[amino]['nH'] * count
            matrix[i][829] += aminos_dict[amino]['nN'] * count
            matrix[i][831] += aminos_dict[amino]['nO'] * count
            matrix[i][833] += aminos_dict[amino]['nS'] * count

        # tirando água que eh perdida durante as ligações
        matrix[i][827] -= 2 * (sequence.length - 1) # H2
        matrix[i][831] -= sequence.length - 1 # 0

        total_elements = np.sum(matrix[i, [825, 827, 829, 831, 833]])
        matrix[i][835] = total_elements
        for j in range(5):
            matrix[i][826 + (j * 2)] = matrix[i][825 + (j * 2)] / total_elements

        matrix[i][-1] = label

    return matrix


############################################################################################################################################



##############################################################################################################################################
########################################################### MATRIX GENERATION ####################################################################
##############################################################################################################################################


def training_matrix(positives_path, negatives_path, save_path):
    positive_matrix = calc_properties(positives_path, 1)
    negative_matrix = calc_properties(negatives_path, 0)
    df_positives = pd.DataFrame(positive_matrix, columns=cabecalho).infer_objects()
    df_negatives = pd.DataFrame(negative_matrix, columns=cabecalho).infer_objects()
    df = pd.concat([df_positives, df_negatives], ignore_index=True).sample(frac=1, random_state=35, ignore_index=True)
    df.to_csv(f'{save_path}.csv', index=False)
    negatives, positives = df['label'].value_counts()
    print(f'Training Matrix was generated.\nPositives Samples: {positives}, Nagative Samples: {negatives}')

############################################################################################################################################


def split_train_test(data_path):
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1]  # Todas as colunas exceto a última (features)
    y = df.iloc[:, -1]   # Apenas a última coluna (label)

    # Dividindo em treino e teste com 80% para treino e 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    print("Tamanho do conjunto de treino:", X_train.shape)
    print("Tamanho do conjunto de teste:", X_test.shape)
    print("Proporção de classes no conjunto de treino:", y_train.value_counts(normalize=True))
    print("Proporção de classes no conjunto de teste:", y_test.value_counts(normalize=True))

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv("Dados CPP924 tunados/TREINO80.csv", index= False)
    test_df.to_csv('Dados CPP924 tunados/TESTE80.csv',  index= False)
    return train_df, test_df


##############################################################################################################################################
################################################################## SCORES  ####################################################################
##############################################################################################################################################

def display_cv_scores(scores):
  data = {
      'Teste':  [f"{round(scores['test_'+metric].mean(), 4)} ({round(scores['test_'+metric].std(), 4)})" for metric in scoring.keys()],
      'Treino': [f"{round(scores['train_'+metric].mean(), 4)} ({round(scores['train_'+metric].std(), 4)})" for metric in scoring.keys()]
      }

  D = pd.DataFrame.from_dict(data, orient='index', columns=scoring.keys())
  print(D)
#############################################################################################################################################





##############################################################################################################################################
########################################################### MODEL TRAINING ####################################################################
##############################################################################################################################################
def training_model(training_matrix_path,save_path):
    df = pd.read_csv(training_matrix_path)
    #X = df.iloc[:, 1:-1]
    X = df[[
    "AH", "CD", "CN", "EA", "FN", "FV", "G", "GG", "GP", "gravy", "H", "hidro/total",
    "IH", "IK", "isoelectric_point", "IV", "K", "KA", "KG", "KN", "KP", "KR", "KS",
    "KV", "LA", "len", "LH", "LP", "mass", "net_charge", "NN", "normA", "normC",
    "normFL", "normGH", "normGN", "normHY", "normID", "normIF", "normIG", "normIP",
    "normIT", "normKD", "normKH", "normLD", "normLF", "normLG", "normLL", "normLN",
    "normLS", "normnC", "normnH", "normnN", "normnO", "normNV", "normP", "normPL",
    "normPS", "normQY", "normRF", "normRH", "normRK", "normRL", "normRM", "normRS",
    "normSA", "normSE", "normSL", "normSP", "normTS", "normVN", "normW", "normWL",
    "normYN", "PG", "QR", "R", "RD", "RN", "TL", "total_elements", "TQ", "VD", "VG",
    "WI", "WN", "L"
    ]]
    #X = df.iloc[:, 1:101]
    y = df.iloc[:, -1]

    model =  ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.05, min_samples_leaf=2, min_samples_split=2, n_estimators=100)
    scores = cross_validate(estimator=model, X=X, y=y,
                        cv=RepeatedKFold(n_splits=5, n_repeats=5),
                        scoring=scoring, return_train_score=True)
    
    display_cv_scores(scores)
    model.fit(X,y)
    pickle.dump(model, open(f'{save_path}.pkl', 'wb'))
    print('Trained model was saved!')
##############################################################################################################################################




##############################################################################################################################################
########################################################### RODAR MODELO ####################################################################
##############################################################################################################################################

def performance(TP, TN, FN, FP):
    MCC = ((TN * TP) - (FN * FP)) / (np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
    ACC = (TP + TN) / (TP + TN + FP + FN)
    SP = TN / (TN + FP)
    SN = TP / (TP + FN)
    PR = TP / (TP + FP)
    F1 = 2 * (PR * SN) / (PR + SN)
    return {
        'MCC': MCC,
        'Accuracy': ACC,
        'Specificity': SP,
        'Recall': SN,
        'Precision': PR,
        'F1 Score': F1
    }


def test_model(teste_path, treino_path ,model_path,save_name):
    model = pickle.load(open(model_path, 'rb'))

    df = pd.read_csv(teste_path)

    original_df = pd.read_csv(treino_path)

    #Remoção de duplicatas (sequências de treino)
    sequences_to_remove = original_df.iloc[:, 0].tolist()
    df = df[~df.iloc[:, 0].isin(sequences_to_remove)]
    
    if df.empty:
        raise ValueError("O dataframe de teste está vazio após a remoção das linhas duplicadas.")

    #X_test = df.iloc[:, 1:-1]
    X_test = df[[
    "AH", "CD", "CN", "EA", "FN", "FV", "G", "GG", "GP", "gravy", "H", "hidro/total",
    "IH", "IK", "isoelectric_point", "IV", "K", "KA", "KG", "KN", "KP", "KR", "KS",
    "KV", "LA", "len", "LH", "LP", "mass", "net_charge", "NN", "normA", "normC",
    "normFL", "normGH", "normGN", "normHY", "normID", "normIF", "normIG", "normIP",
    "normIT", "normKD", "normKH", "normLD", "normLF", "normLG", "normLL", "normLN",
    "normLS", "normnC", "normnH", "normnN", "normnO", "normNV", "normP", "normPL",
    "normPS", "normQY", "normRF", "normRH", "normRK", "normRL", "normRM", "normRS",
    "normSA", "normSE", "normSL", "normSP", "normTS", "normVN", "normW", "normWL",
    "normYN", "PG", "QR", "R", "RD", "RN", "TL", "total_elements", "TQ", "VD", "VG",
    "WI", "WN", "L"
    ]]
    #X_test = df.iloc[:, 1:101]
    y_test = df.iloc[:, -1]

    # Avalia o modelo no conjunto de teste
    y_test_probs = model.predict_proba(X_test)[:, 1]
    y_test_pred = [1 if p > 0.5 else 0 for p in y_test_probs]

    # Calcula a AUC-ROC
    auc_roc = roc_auc_score(y_test, y_test_probs)

    # Calcula a matriz de confusão
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()

    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')  # Verificação dos valores

    # Calcula as métricas de desempenho
    metrics = performance(tp, tn, fn, fp)
    metrics['AUC-ROC'] = auc_roc


    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    D = pd.DataFrame()
    D['seq'] = df.iloc[:, 0]
    D['Classification'], D['Real_Class'], D['Prob']= y_test_pred, y_test, y_test_probs
    D.to_csv(f'FINAL-21-08/predict-{save_name}.csv', index=False)
    results_df = pd.DataFrame([metrics])
    results_df.to_csv(f'FINAL-21-08/performance-{save_name}.csv', index=False)
    print('Resultas da predição gerados com sucesso!')

##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################



#training_matrix(positives_path, negatives_path, save_path)


#training_model(training_matrix_path,save_path)


#test_model(teste_matrix_path, treino_matrix_path ,model_path, save_name)