import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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



import utils
from utils import aminos as aminos_dict
from scipy.stats import pearsonr



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
             'nC', 'normnC', 'nH', 'normnH', 'nN', 'normnN', 'nO', 'normnO', 'nS', 'normnS', 'total_elements',
             'AAA', 'AAC', 'AAD', 'AAE', 'AAF', 'AAG', 'AAH', 'AAI', 'AAK', 'AAL', 'AAM', 'AAN', 'AAP', 'AAQ', 'AAR', 'AAS', 'AAT', 'AAV', 'AAW', 'AAY',
             'ACA', 'ACC', 'ACD', 'ACE', 'ACF', 'ACG', 'ACH', 'ACI', 'ACK', 'ACL', 'ACM', 'ACN', 'ACP', 'ACQ', 'ACR', 'ACS', 'ACT', 'ACV', 'ACW', 'ACY',
             'ADA', 'ADC', 'ADD', 'ADE', 'ADF', 'ADG', 'ADH', 'ADI', 'ADK', 'ADL', 'ADM', 'ADN', 'ADP', 'ADQ', 'ADR', 'ADS', 'ADT', 'ADV', 'ADW', 'ADY',
             'AEA', 'AEC', 'AED', 'AEE', 'AEF', 'AEG', 'AEH', 'AEI', 'AEK', 'AEL', 'AEM', 'AEN', 'AEP', 'AEQ', 'AER', 'AES', 'AET', 'AEV', 'AEW', 'AEY',
             'AFA', 'AFC', 'AFD', 'AFE', 'AFF', 'AFG', 'AFH', 'AFI', 'AFK', 'AFL', 'AFM', 'AFN', 'AFP', 'AFQ', 'AFR', 'AFS', 'AFT', 'AFV', 'AFW', 'AFY',
             'AGA', 'AGC', 'AGD', 'AGE', 'AGF', 'AGG', 'AGH', 'AGI', 'AGK', 'AGL', 'AGM', 'AGN', 'AGP', 'AGQ', 'AGR', 'AGS', 'AGT', 'AGV', 'AGW', 'AGY',
             'AHA', 'AHC', 'AHD', 'AHE', 'AHF', 'AHG', 'AHH', 'AHI', 'AHK', 'AHL', 'AHM', 'AHN', 'AHP', 'AHQ', 'AHR', 'AHS', 'AHT', 'AHV', 'AHW', 'AHY',
             'AIA', 'AIC', 'AID', 'AIE', 'AIF', 'AIG', 'AIH', 'AII', 'AIK', 'AIL', 'AIM', 'AIN', 'AIP', 'AIQ', 'AIR', 'AIS', 'AIT', 'AIV', 'AIW', 'AIY',
             'AKA', 'AKC', 'AKD', 'AKE', 'AKF', 'AKG', 'AKH', 'AKI', 'AKK', 'AKL', 'AKM', 'AKN', 'AKP', 'AKQ', 'AKR', 'AKS', 'AKT', 'AKV', 'AKW', 'AKY',
             'ALA', 'ALC', 'ALD', 'ALE', 'ALF', 'ALG', 'ALH', 'ALI', 'ALK', 'ALL', 'ALM', 'ALN', 'ALP', 'ALQ', 'ALR', 'ALS', 'ALT', 'ALV', 'ALW', 'ALY',
             'AMA', 'AMC', 'AMD', 'AME', 'AMF', 'AMG', 'AMH', 'AMI', 'AMK', 'AML', 'AMM', 'AMN', 'AMP', 'AMQ', 'AMR', 'AMS', 'AMT', 'AMV', 'AMW', 'AMY',
             'ANA', 'ANC', 'AND', 'ANE', 'ANF', 'ANG', 'ANH', 'ANI', 'ANK', 'ANL', 'ANM', 'ANN', 'ANP', 'ANQ', 'ANR', 'ANS', 'ANT', 'ANV', 'ANW', 'ANY',
             'APA', 'APC', 'APD', 'APE', 'APF', 'APG', 'APH', 'API', 'APK', 'APL', 'APM', 'APN', 'APP', 'APQ', 'APR', 'APS', 'APT', 'APV', 'APW', 'APY',
             'AQA', 'AQC', 'AQD', 'AQE', 'AQF', 'AQG', 'AQH', 'AQI', 'AQK', 'AQL', 'AQM', 'AQN', 'AQP', 'AQQ', 'AQR', 'AQS', 'AQT', 'AQV', 'AQW', 'AQY',
             'ARA', 'ARC', 'ARD', 'ARE', 'ARF', 'ARG', 'ARH', 'ARI', 'ARK', 'ARL', 'ARM', 'ARN', 'ARP', 'ARQ', 'ARR', 'ARS', 'ART', 'ARV', 'ARW', 'ARY',
             'ASA', 'ASC', 'ASD', 'ASE', 'ASF', 'ASG', 'ASH', 'ASI', 'ASK', 'ASL', 'ASM', 'ASN', 'ASP', 'ASQ', 'ASR', 'ASS', 'AST', 'ASV', 'ASW', 'ASY',
             'ATA', 'ATC', 'ATD', 'ATE', 'ATF', 'ATG', 'ATH', 'ATI', 'ATK', 'ATL', 'ATM', 'ATN', 'ATP', 'ATQ', 'ATR', 'ATS', 'ATT', 'ATV', 'ATW', 'ATY',
             'AVA', 'AVC', 'AVD', 'AVE', 'AVF', 'AVG', 'AVH', 'AVI', 'AVK', 'AVL', 'AVM', 'AVN', 'AVP', 'AVQ', 'AVR', 'AVS', 'AVT', 'AVV', 'AVW', 'AVY',
             'AWA', 'AWC', 'AWD', 'AWE', 'AWF', 'AWG', 'AWH', 'AWI', 'AWK', 'AWL', 'AWM', 'AWN', 'AWP', 'AWQ', 'AWR', 'AWS', 'AWT', 'AWV', 'AWW', 'AWY',
             'AYA', 'AYC', 'AYD', 'AYE', 'AYF', 'AYG', 'AYH', 'AYI', 'AYK', 'AYL', 'AYM', 'AYN', 'AYP', 'AYQ', 'AYR', 'AYS', 'AYT', 'AYV', 'AYW', 'AYY',
             'CAA', 'CAC', 'CAD', 'CAE', 'CAF', 'CAG', 'CAH', 'CAI', 'CAK', 'CAL', 'CAM', 'CAN', 'CAP', 'CAQ', 'CAR', 'CAS', 'CAT', 'CAV', 'CAW', 'CAY',
             'CCA', 'CCC', 'CCD', 'CCE', 'CCF', 'CCG', 'CCH', 'CCI', 'CCK', 'CCL', 'CCM', 'CCN', 'CCP', 'CCQ', 'CCR', 'CCS', 'CCT', 'CCV', 'CCW', 'CCY',
             'CDA', 'CDC', 'CDD', 'CDE', 'CDF', 'CDG', 'CDH', 'CDI', 'CDK', 'CDL', 'CDM', 'CDN', 'CDP', 'CDQ', 'CDR', 'CDS', 'CDT', 'CDV', 'CDW', 'CDY',
             'CEA', 'CEC', 'CED', 'CEE', 'CEF', 'CEG', 'CEH', 'CEI', 'CEK', 'CEL', 'CEM', 'CEN', 'CEP', 'CEQ', 'CER', 'CES', 'CET', 'CEV', 'CEW', 'CEY',
             'CFA', 'CFC', 'CFD', 'CFE', 'CFF', 'CFG', 'CFH', 'CFI', 'CFK', 'CFL', 'CFM', 'CFN', 'CFP', 'CFQ', 'CFR', 'CFS', 'CFT', 'CFV', 'CFW', 'CFY',
             'CGA', 'CGC', 'CGD', 'CGE', 'CGF', 'CGG', 'CGH', 'CGI', 'CGK', 'CGL', 'CGM', 'CGN', 'CGP', 'CGQ', 'CGR', 'CGS', 'CGT', 'CGV', 'CGW', 'CGY', 
             'CHA', 'CHC', 'CHD', 'CHE', 'CHF', 'CHG', 'CHH', 'CHI', 'CHK', 'CHL', 'CHM', 'CHN', 'CHP', 'CHQ', 'CHR', 'CHS', 'CHT', 'CHV', 'CHW', 'CHY', 
             'CIA', 'CIC', 'CID', 'CIE', 'CIF', 'CIG', 'CIH', 'CII', 'CIK', 'CIL', 'CIM', 'CIN', 'CIP', 'CIQ', 'CIR', 'CIS', 'CIT', 'CIV', 'CIW', 'CIY',
             'CKA', 'CKC', 'CKD', 'CKE', 'CKF', 'CKG', 'CKH', 'CKI', 'CKK', 'CKL', 'CKM', 'CKN', 'CKP', 'CKQ', 'CKR', 'CKS', 'CKT', 'CKV', 'CKW', 'CKY',
             'CLA', 'CLC', 'CLD', 'CLE', 'CLF', 'CLG', 'CLH', 'CLI', 'CLK', 'CLL', 'CLM', 'CLN', 'CLP', 'CLQ', 'CLR', 'CLS', 'CLT', 'CLV', 'CLW', 'CLY',
             'CMA', 'CMC', 'CMD', 'CME', 'CMF', 'CMG', 'CMH', 'CMI', 'CMK', 'CML', 'CMM', 'CMN', 'CMP', 'CMQ', 'CMR', 'CMS', 'CMT', 'CMV', 'CMW', 'CMY',
             'CNA', 'CNC', 'CND', 'CNE', 'CNF', 'CNG', 'CNH', 'CNI', 'CNK', 'CNL', 'CNM', 'CNN', 'CNP', 'CNQ', 'CNR', 'CNS', 'CNT', 'CNV', 'CNW', 'CNY',
             'CPA', 'CPC', 'CPD', 'CPE', 'CPF', 'CPG', 'CPH', 'CPI', 'CPK', 'CPL', 'CPM', 'CPN', 'CPP', 'CPQ', 'CPR', 'CPS', 'CPT', 'CPV', 'CPW', 'CPY',
             'CQA', 'CQC', 'CQD', 'CQE', 'CQF', 'CQG', 'CQH', 'CQI', 'CQK', 'CQL', 'CQM', 'CQN', 'CQP', 'CQQ', 'CQR', 'CQS', 'CQT', 'CQV', 'CQW', 'CQY',
             'CRA', 'CRC', 'CRD', 'CRE', 'CRF', 'CRG', 'CRH', 'CRI', 'CRK', 'CRL', 'CRM', 'CRN', 'CRP', 'CRQ', 'CRR', 'CRS', 'CRT', 'CRV', 'CRW', 'CRY',
             'CSA', 'CSC', 'CSD', 'CSE', 'CSF', 'CSG', 'CSH', 'CSI', 'CSK', 'CSL', 'CSM', 'CSN', 'CSP', 'CSQ', 'CSR', 'CSS', 'CST', 'CSV', 'CSW', 'CSY',
             'CTA', 'CTC', 'CTD', 'CTE', 'CTF', 'CTG', 'CTH', 'CTI', 'CTK', 'CTL', 'CTM', 'CTN', 'CTP', 'CTQ', 'CTR', 'CTS', 'CTT', 'CTV', 'CTW', 'CTY',
             'CVA', 'CVC', 'CVD', 'CVE', 'CVF', 'CVG', 'CVH', 'CVI', 'CVK', 'CVL', 'CVM', 'CVN', 'CVP', 'CVQ', 'CVR', 'CVS', 'CVT', 'CVV', 'CVW', 'CVY',
             'CWA', 'CWC', 'CWD', 'CWE', 'CWF', 'CWG', 'CWH', 'CWI', 'CWK', 'CWL', 'CWM', 'CWN', 'CWP', 'CWQ', 'CWR', 'CWS', 'CWT', 'CWV', 'CWW', 'CWY',
             'CYA', 'CYC', 'CYD', 'CYE', 'CYF', 'CYG', 'CYH', 'CYI', 'CYK', 'CYL', 'CYM', 'CYN', 'CYP', 'CYQ', 'CYR', 'CYS', 'CYT', 'CYV', 'CYW', 'CYY',
             'DAA', 'DAC', 'DAD', 'DAE', 'DAF', 'DAG', 'DAH', 'DAI', 'DAK', 'DAL', 'DAM', 'DAN', 'DAP', 'DAQ', 'DAR', 'DAS', 'DAT', 'DAV', 'DAW', 'DAY',
             'DCA', 'DCC', 'DCD', 'DCE', 'DCF', 'DCG', 'DCH', 'DCI', 'DCK', 'DCL', 'DCM', 'DCN', 'DCP', 'DCQ', 'DCR', 'DCS', 'DCT', 'DCV', 'DCW', 'DCY',
             'DDA', 'DDC', 'DDD', 'DDE', 'DDF', 'DDG', 'DDH', 'DDI', 'DDK', 'DDL', 'DDM', 'DDN', 'DDP', 'DDQ', 'DDR', 'DDS', 'DDT', 'DDV', 'DDW', 'DDY',
             'DEA', 'DEC', 'DED', 'DEE', 'DEF', 'DEG', 'DEH', 'DEI', 'DEK', 'DEL', 'DEM', 'DEN', 'DEP', 'DEQ', 'DER', 'DES', 'DET', 'DEV', 'DEW', 'DEY',
             'DFA', 'DFC', 'DFD', 'DFE', 'DFF', 'DFG', 'DFH', 'DFI', 'DFK', 'DFL', 'DFM', 'DFN', 'DFP', 'DFQ', 'DFR', 'DFS', 'DFT', 'DFV', 'DFW', 'DFY',
             'DGA', 'DGC', 'DGD', 'DGE', 'DGF', 'DGG', 'DGH', 'DGI', 'DGK', 'DGL', 'DGM', 'DGN', 'DGP', 'DGQ', 'DGR', 'DGS', 'DGT', 'DGV', 'DGW', 'DGY',
             'DHA', 'DHC', 'DHD', 'DHE', 'DHF', 'DHG', 'DHH', 'DHI', 'DHK', 'DHL', 'DHM', 'DHN', 'DHP', 'DHQ', 'DHR', 'DHS', 'DHT', 'DHV', 'DHW', 'DHY',
             'DIA', 'DIC', 'DID', 'DIE', 'DIF', 'DIG', 'DIH', 'DII', 'DIK', 'DIL', 'DIM', 'DIN', 'DIP', 'DIQ', 'DIR', 'DIS', 'DIT', 'DIV', 'DIW', 'DIY',
             'DKA', 'DKC', 'DKD', 'DKE', 'DKF', 'DKG', 'DKH', 'DKI', 'DKK', 'DKL', 'DKM', 'DKN', 'DKP', 'DKQ', 'DKR', 'DKS', 'DKT', 'DKV', 'DKW', 'DKY',
             'DLA', 'DLC', 'DLD', 'DLE', 'DLF', 'DLG', 'DLH', 'DLI', 'DLK', 'DLL', 'DLM', 'DLN', 'DLP', 'DLQ', 'DLR', 'DLS', 'DLT', 'DLV', 'DLW', 'DLY',
             'DMA', 'DMC', 'DMD', 'DME', 'DMF', 'DMG', 'DMH', 'DMI', 'DMK', 'DML', 'DMM', 'DMN', 'DMP', 'DMQ', 'DMR', 'DMS', 'DMT', 'DMV', 'DMW', 'DMY',
             'DNA', 'DNC', 'DND', 'DNE', 'DNF', 'DNG', 'DNH', 'DNI', 'DNK', 'DNL', 'DNM', 'DNN', 'DNP', 'DNQ', 'DNR', 'DNS', 'DNT', 'DNV', 'DNW', 'DNY',
             'DPA', 'DPC', 'DPD', 'DPE', 'DPF', 'DPG', 'DPH', 'DPI', 'DPK', 'DPL', 'DPM', 'DPN', 'DPP', 'DPQ', 'DPR', 'DPS', 'DPT', 'DPV', 'DPW', 'DPY',
             'DQA', 'DQC', 'DQD', 'DQE', 'DQF', 'DQG', 'DQH', 'DQI', 'DQK', 'DQL', 'DQM', 'DQN', 'DQP', 'DQQ', 'DQR', 'DQS', 'DQT', 'DQV', 'DQW', 'DQY',
             'DRA', 'DRC', 'DRD', 'DRE', 'DRF', 'DRG', 'DRH', 'DRI', 'DRK', 'DRL', 'DRM', 'DRN', 'DRP', 'DRQ', 'DRR', 'DRS', 'DRT', 'DRV', 'DRW', 'DRY',
             'DSA', 'DSC', 'DSD', 'DSE', 'DSF', 'DSG', 'DSH', 'DSI', 'DSK', 'DSL', 'DSM', 'DSN', 'DSP', 'DSQ', 'DSR', 'DSS', 'DST', 'DSV', 'DSW', 'DSY',
             'DTA', 'DTC', 'DTD', 'DTE', 'DTF', 'DTG', 'DTH', 'DTI', 'DTK', 'DTL', 'DTM', 'DTN', 'DTP', 'DTQ', 'DTR', 'DTS', 'DTT', 'DTV', 'DTW', 'DTY',
             'DVA', 'DVC', 'DVD', 'DVE', 'DVF', 'DVG', 'DVH', 'DVI', 'DVK', 'DVL', 'DVM', 'DVN', 'DVP', 'DVQ', 'DVR', 'DVS', 'DVT', 'DVV', 'DVW', 'DVY',
             'DWA', 'DWC', 'DWD', 'DWE', 'DWF', 'DWG', 'DWH', 'DWI', 'DWK', 'DWL', 'DWM', 'DWN', 'DWP', 'DWQ', 'DWR', 'DWS', 'DWT', 'DWV', 'DWW', 'DWY',
             'DYA', 'DYC', 'DYD', 'DYE', 'DYF', 'DYG', 'DYH', 'DYI', 'DYK', 'DYL', 'DYM', 'DYN', 'DYP', 'DYQ', 'DYR', 'DYS', 'DYT', 'DYV', 'DYW', 'DYY',
             'EAA', 'EAC', 'EAD', 'EAE', 'EAF', 'EAG', 'EAH', 'EAI', 'EAK', 'EAL', 'EAM', 'EAN', 'EAP', 'EAQ', 'EAR', 'EAS', 'EAT', 'EAV', 'EAW', 'EAY',
             'ECA', 'ECC', 'ECD', 'ECE', 'ECF', 'ECG', 'ECH', 'ECI', 'ECK', 'ECL', 'ECM', 'ECN', 'ECP', 'ECQ', 'ECR', 'ECS', 'ECT', 'ECV', 'ECW', 'ECY',
             'EDA', 'EDC', 'EDD', 'EDE', 'EDF', 'EDG', 'EDH', 'EDI', 'EDK', 'EDL', 'EDM', 'EDN', 'EDP', 'EDQ', 'EDR', 'EDS', 'EDT', 'EDV', 'EDW', 'EDY',
             'EEA', 'EEC', 'EED', 'EEE', 'EEF', 'EEG', 'EEH', 'EEI', 'EEK', 'EEL', 'EEM', 'EEN', 'EEP', 'EEQ', 'EER', 'EES', 'EET', 'EEV', 'EEW', 'EEY',
             'EFA', 'EFC', 'EFD', 'EFE', 'EFF', 'EFG', 'EFH', 'EFI', 'EFK', 'EFL', 'EFM', 'EFN', 'EFP', 'EFQ', 'EFR', 'EFS', 'EFT', 'EFV', 'EFW', 'EFY',
             'EGA', 'EGC', 'EGD', 'EGE', 'EGF', 'EGG', 'EGH', 'EGI', 'EGK', 'EGL', 'EGM', 'EGN', 'EGP', 'EGQ', 'EGR', 'EGS', 'EGT', 'EGV', 'EGW', 'EGY',
             'EHA', 'EHC', 'EHD', 'EHE', 'EHF', 'EHG', 'EHH', 'EHI', 'EHK', 'EHL', 'EHM', 'EHN', 'EHP', 'EHQ', 'EHR', 'EHS', 'EHT', 'EHV', 'EHW', 'EHY',
             'EIA', 'EIC', 'EID', 'EIE', 'EIF', 'EIG', 'EIH', 'EII', 'EIK', 'EIL', 'EIM', 'EIN', 'EIP', 'EIQ', 'EIR', 'EIS', 'EIT', 'EIV', 'EIW', 'EIY',
             'EKA', 'EKC', 'EKD', 'EKE', 'EKF', 'EKG', 'EKH', 'EKI', 'EKK', 'EKL', 'EKM', 'EKN', 'EKP', 'EKQ', 'EKR', 'EKS', 'EKT', 'EKV', 'EKW', 'EKY',
             'ELA', 'ELC', 'ELD', 'ELE', 'ELF', 'ELG', 'ELH', 'ELI', 'ELK', 'ELL', 'ELM', 'ELN', 'ELP', 'ELQ', 'ELR', 'ELS', 'ELT', 'ELV', 'ELW', 'ELY',
             'EMA', 'EMC', 'EMD', 'EME', 'EMF', 'EMG', 'EMH', 'EMI', 'EMK', 'EML', 'EMM', 'EMN', 'EMP', 'EMQ', 'EMR', 'EMS', 'EMT', 'EMV', 'EMW', 'EMY',
             'ENA', 'ENC', 'END', 'ENE', 'ENF', 'ENG', 'ENH', 'ENI', 'ENK', 'ENL', 'ENM', 'ENN', 'ENP', 'ENQ', 'ENR', 'ENS', 'ENT', 'ENV', 'ENW', 'ENY',
             'EPA', 'EPC', 'EPD', 'EPE', 'EPF', 'EPG', 'EPH', 'EPI', 'EPK', 'EPL', 'EPM', 'EPN', 'EPP', 'EPQ', 'EPR', 'EPS', 'EPT', 'EPV', 'EPW', 'EPY',
             'EQA', 'EQC', 'EQD', 'EQE', 'EQF', 'EQG', 'EQH', 'EQI', 'EQK', 'EQL', 'EQM', 'EQN', 'EQP', 'EQQ', 'EQR', 'EQS', 'EQT', 'EQV', 'EQW', 'EQY',
             'ERA', 'ERC', 'ERD', 'ERE', 'ERF', 'ERG', 'ERH', 'ERI', 'ERK', 'ERL', 'ERM', 'ERN', 'ERP', 'ERQ', 'ERR', 'ERS', 'ERT', 'ERV', 'ERW', 'ERY',
             'ESA', 'ESC', 'ESD', 'ESE', 'ESF', 'ESG', 'ESH', 'ESI', 'ESK', 'ESL', 'ESM', 'ESN', 'ESP', 'ESQ', 'ESR', 'ESS', 'EST', 'ESV', 'ESW', 'ESY',
             'ETA', 'ETC', 'ETD', 'ETE', 'ETF', 'ETG', 'ETH', 'ETI', 'ETK', 'ETL', 'ETM', 'ETN', 'ETP', 'ETQ', 'ETR', 'ETS', 'ETT', 'ETV', 'ETW', 'ETY',
             'EVA', 'EVC', 'EVD', 'EVE', 'EVF', 'EVG', 'EVH', 'EVI', 'EVK', 'EVL', 'EVM', 'EVN', 'EVP', 'EVQ', 'EVR', 'EVS', 'EVT', 'EVV', 'EVW', 'EVY',
             'EWA', 'EWC', 'EWD', 'EWE', 'EWF', 'EWG', 'EWH', 'EWI', 'EWK', 'EWL', 'EWM', 'EWN', 'EWP', 'EWQ', 'EWR', 'EWS', 'EWT', 'EWV', 'EWW', 'EWY',
             'EYA', 'EYC', 'EYD', 'EYE', 'EYF', 'EYG', 'EYH', 'EYI', 'EYK', 'EYL', 'EYM', 'EYN', 'EYP', 'EYQ', 'EYR', 'EYS', 'EYT', 'EYV', 'EYW', 'EYY',
             'FAA', 'FAC', 'FAD', 'FAE', 'FAF', 'FAG', 'FAH', 'FAI', 'FAK', 'FAL', 'FAM', 'FAN', 'FAP', 'FAQ', 'FAR', 'FAS', 'FAT', 'FAV', 'FAW', 'FAY',
             'FCA', 'FCC', 'FCD', 'FCE', 'FCF', 'FCG', 'FCH', 'FCI', 'FCK', 'FCL', 'FCM', 'FCN', 'FCP', 'FCQ', 'FCR', 'FCS', 'FCT', 'FCV', 'FCW', 'FCY',
             'FDA', 'FDC', 'FDD', 'FDE', 'FDF', 'FDG', 'FDH', 'FDI', 'FDK', 'FDL', 'FDM', 'FDN', 'FDP', 'FDQ', 'FDR', 'FDS', 'FDT', 'FDV', 'FDW', 'FDY',
             'FEA', 'FEC', 'FED', 'FEE', 'FEF', 'FEG', 'FEH', 'FEI', 'FEK', 'FEL', 'FEM', 'FEN', 'FEP', 'FEQ', 'FER', 'FES', 'FET', 'FEV', 'FEW', 'FEY',
             'FFA', 'FFC', 'FFD', 'FFE', 'FFF', 'FFG', 'FFH', 'FFI', 'FFK', 'FFL', 'FFM', 'FFN', 'FFP', 'FFQ', 'FFR', 'FFS', 'FFT', 'FFV', 'FFW', 'FFY',
             'FGA', 'FGC', 'FGD', 'FGE', 'FGF', 'FGG', 'FGH', 'FGI', 'FGK', 'FGL', 'FGM', 'FGN', 'FGP', 'FGQ', 'FGR', 'FGS', 'FGT', 'FGV', 'FGW', 'FGY',
             'FHA', 'FHC', 'FHD', 'FHE', 'FHF', 'FHG', 'FHH', 'FHI', 'FHK', 'FHL', 'FHM', 'FHN', 'FHP', 'FHQ', 'FHR', 'FHS', 'FHT', 'FHV', 'FHW', 'FHY',
             'FIA', 'FIC', 'FID', 'FIE', 'FIF', 'FIG', 'FIH', 'FII', 'FIK', 'FIL', 'FIM', 'FIN', 'FIP', 'FIQ', 'FIR', 'FIS', 'FIT', 'FIV', 'FIW', 'FIY',
             'FKA', 'FKC', 'FKD', 'FKE', 'FKF', 'FKG', 'FKH', 'FKI', 'FKK', 'FKL', 'FKM', 'FKN', 'FKP', 'FKQ', 'FKR', 'FKS', 'FKT', 'FKV', 'FKW', 'FKY',
             'FLA', 'FLC', 'FLD', 'FLE', 'FLF', 'FLG', 'FLH', 'FLI', 'FLK', 'FLL', 'FLM', 'FLN', 'FLP', 'FLQ', 'FLR', 'FLS', 'FLT', 'FLV', 'FLW', 'FLY',
             'FMA', 'FMC', 'FMD', 'FME', 'FMF', 'FMG', 'FMH', 'FMI', 'FMK', 'FML', 'FMM', 'FMN', 'FMP', 'FMQ', 'FMR', 'FMS', 'FMT', 'FMV', 'FMW', 'FMY',
             'FNA', 'FNC', 'FND', 'FNE', 'FNF', 'FNG', 'FNH', 'FNI', 'FNK', 'FNL', 'FNM', 'FNN', 'FNP', 'FNQ', 'FNR', 'FNS', 'FNT', 'FNV', 'FNW', 'FNY',
             'FPA', 'FPC', 'FPD', 'FPE', 'FPF', 'FPG', 'FPH', 'FPI', 'FPK', 'FPL', 'FPM', 'FPN', 'FPP', 'FPQ', 'FPR', 'FPS', 'FPT', 'FPV', 'FPW', 'FPY',
             'FQA', 'FQC', 'FQD', 'FQE', 'FQF', 'FQG', 'FQH', 'FQI', 'FQK', 'FQL', 'FQM', 'FQN', 'FQP', 'FQQ', 'FQR', 'FQS', 'FQT', 'FQV', 'FQW', 'FQY',
             'FRA', 'FRC', 'FRD', 'FRE', 'FRF', 'FRG', 'FRH', 'FRI', 'FRK', 'FRL', 'FRM', 'FRN', 'FRP', 'FRQ', 'FRR', 'FRS', 'FRT', 'FRV', 'FRW', 'FRY',
             'FSA', 'FSC', 'FSD', 'FSE', 'FSF', 'FSG', 'FSH', 'FSI', 'FSK', 'FSL', 'FSM', 'FSN', 'FSP', 'FSQ', 'FSR', 'FSS', 'FST', 'FSV', 'FSW', 'FSY',
             'FTA', 'FTC', 'FTD', 'FTE', 'FTF', 'FTG', 'FTH', 'FTI', 'FTK', 'FTL', 'FTM', 'FTN', 'FTP', 'FTQ', 'FTR', 'FTS', 'FTT', 'FTV', 'FTW', 'FTY',
             'FVA', 'FVC', 'FVD', 'FVE', 'FVF', 'FVG', 'FVH', 'FVI', 'FVK', 'FVL', 'FVM', 'FVN', 'FVP', 'FVQ', 'FVR', 'FVS', 'FVT', 'FVV', 'FVW', 'FVY',
             'FWA', 'FWC', 'FWD', 'FWE', 'FWF', 'FWG', 'FWH', 'FWI', 'FWK', 'FWL', 'FWM', 'FWN', 'FWP', 'FWQ', 'FWR', 'FWS', 'FWT', 'FWV', 'FWW', 'FWY',
             'FYA', 'FYC', 'FYD', 'FYE', 'FYF', 'FYG', 'FYH', 'FYI', 'FYK', 'FYL', 'FYM', 'FYN', 'FYP', 'FYQ', 'FYR', 'FYS', 'FYT', 'FYV', 'FYW', 'FYY',
             'GAA', 'GAC', 'GAD', 'GAE', 'GAF', 'GAG', 'GAH', 'GAI', 'GAK', 'GAL', 'GAM', 'GAN', 'GAP', 'GAQ', 'GAR', 'GAS', 'GAT', 'GAV', 'GAW', 'GAY',
             'GCA', 'GCC', 'GCD', 'GCE', 'GCF', 'GCG', 'GCH', 'GCI', 'GCK', 'GCL', 'GCM', 'GCN', 'GCP', 'GCQ', 'GCR', 'GCS', 'GCT', 'GCV', 'GCW', 'GCY',
             'GDA', 'GDC', 'GDD', 'GDE', 'GDF', 'GDG', 'GDH', 'GDI', 'GDK', 'GDL', 'GDM', 'GDN', 'GDP', 'GDQ', 'GDR', 'GDS', 'GDT', 'GDV', 'GDW', 'GDY',
             'GEA', 'GEC', 'GED', 'GEE', 'GEF', 'GEG', 'GEH', 'GEI', 'GEK', 'GEL', 'GEM', 'GEN', 'GEP', 'GEQ', 'GER', 'GES', 'GET', 'GEV', 'GEW', 'GEY',
             'GFA', 'GFC', 'GFD', 'GFE', 'GFF', 'GFG', 'GFH', 'GFI', 'GFK', 'GFL', 'GFM', 'GFN', 'GFP', 'GFQ', 'GFR', 'GFS', 'GFT', 'GFV', 'GFW', 'GFY',
             'GGA', 'GGC', 'GGD', 'GGE', 'GGF', 'GGG', 'GGH', 'GGI', 'GGK', 'GGL', 'GGM', 'GGN', 'GGP', 'GGQ', 'GGR', 'GGS', 'GGT', 'GGV', 'GGW', 'GGY',
             'GHA', 'GHC', 'GHD', 'GHE', 'GHF', 'GHG', 'GHH', 'GHI', 'GHK', 'GHL', 'GHM', 'GHN', 'GHP', 'GHQ', 'GHR', 'GHS', 'GHT', 'GHV', 'GHW', 'GHY',
             'GIA', 'GIC', 'GID', 'GIE', 'GIF', 'GIG', 'GIH', 'GII', 'GIK', 'GIL', 'GIM', 'GIN', 'GIP', 'GIQ', 'GIR', 'GIS', 'GIT', 'GIV', 'GIW', 'GIY',
             'GKA', 'GKC', 'GKD', 'GKE', 'GKF', 'GKG', 'GKH', 'GKI', 'GKK', 'GKL', 'GKM', 'GKN', 'GKP', 'GKQ', 'GKR', 'GKS', 'GKT', 'GKV', 'GKW', 'GKY',
             'GLA', 'GLC', 'GLD', 'GLE', 'GLF', 'GLG', 'GLH', 'GLI', 'GLK', 'GLL', 'GLM', 'GLN', 'GLP', 'GLQ', 'GLR', 'GLS', 'GLT', 'GLV', 'GLW', 'GLY',
             'GMA', 'GMC', 'GMD', 'GME', 'GMF', 'GMG', 'GMH', 'GMI', 'GMK', 'GML', 'GMM', 'GMN', 'GMP', 'GMQ', 'GMR', 'GMS', 'GMT', 'GMV', 'GMW', 'GMY',
             'GNA', 'GNC', 'GND', 'GNE', 'GNF', 'GNG', 'GNH', 'GNI', 'GNK', 'GNL', 'GNM', 'GNN', 'GNP', 'GNQ', 'GNR', 'GNS', 'GNT', 'GNV', 'GNW', 'GNY',
             'GPA', 'GPC', 'GPD', 'GPE', 'GPF', 'GPG', 'GPH', 'GPI', 'GPK', 'GPL', 'GPM', 'GPN', 'GPP', 'GPQ', 'GPR', 'GPS', 'GPT', 'GPV', 'GPW', 'GPY',
             'GQA', 'GQC', 'GQD', 'GQE', 'GQF', 'GQG', 'GQH', 'GQI', 'GQK', 'GQL', 'GQM', 'GQN', 'GQP', 'GQQ', 'GQR', 'GQS', 'GQT', 'GQV', 'GQW', 'GQY',
             'GRA', 'GRC', 'GRD', 'GRE', 'GRF', 'GRG', 'GRH', 'GRI', 'GRK', 'GRL', 'GRM', 'GRN', 'GRP', 'GRQ', 'GRR', 'GRS', 'GRT', 'GRV', 'GRW', 'GRY',
             'GSA', 'GSC', 'GSD', 'GSE', 'GSF', 'GSG', 'GSH', 'GSI', 'GSK', 'GSL', 'GSM', 'GSN', 'GSP', 'GSQ', 'GSR', 'GSS', 'GST', 'GSV', 'GSW', 'GSY',
             'GTA', 'GTC', 'GTD', 'GTE', 'GTF', 'GTG', 'GTH', 'GTI', 'GTK', 'GTL', 'GTM', 'GTN', 'GTP', 'GTQ', 'GTR', 'GTS', 'GTT', 'GTV', 'GTW', 'GTY',
             'GVA', 'GVC', 'GVD', 'GVE', 'GVF', 'GVG', 'GVH', 'GVI', 'GVK', 'GVL', 'GVM', 'GVN', 'GVP', 'GVQ', 'GVR', 'GVS', 'GVT', 'GVV', 'GVW', 'GVY',
             'GWA', 'GWC', 'GWD', 'GWE', 'GWF', 'GWG', 'GWH', 'GWI', 'GWK', 'GWL', 'GWM', 'GWN', 'GWP', 'GWQ', 'GWR', 'GWS', 'GWT', 'GWV', 'GWW', 'GWY',
             'GYA', 'GYC', 'GYD', 'GYE', 'GYF', 'GYG', 'GYH', 'GYI', 'GYK', 'GYL', 'GYM', 'GYN', 'GYP', 'GYQ', 'GYR', 'GYS', 'GYT', 'GYV', 'GYW', 'GYY',
             'HAA', 'HAC', 'HAD', 'HAE', 'HAF', 'HAG', 'HAH', 'HAI', 'HAK', 'HAL', 'HAM', 'HAN', 'HAP', 'HAQ', 'HAR', 'HAS', 'HAT', 'HAV', 'HAW', 'HAY',
             'HCA', 'HCC', 'HCD', 'HCE', 'HCF', 'HCG', 'HCH', 'HCI', 'HCK', 'HCL', 'HCM', 'HCN', 'HCP', 'HCQ', 'HCR', 'HCS', 'HCT', 'HCV', 'HCW', 'HCY',
             'HDA', 'HDC', 'HDD', 'HDE', 'HDF', 'HDG', 'HDH', 'HDI', 'HDK', 'HDL', 'HDM', 'HDN', 'HDP', 'HDQ', 'HDR', 'HDS', 'HDT', 'HDV', 'HDW', 'HDY',
             'HEA', 'HEC', 'HED', 'HEE', 'HEF', 'HEG', 'HEH', 'HEI', 'HEK', 'HEL', 'HEM', 'HEN', 'HEP', 'HEQ', 'HER', 'HES', 'HET', 'HEV', 'HEW', 'HEY',
             'HFA', 'HFC', 'HFD', 'HFE', 'HFF', 'HFG', 'HFH', 'HFI', 'HFK', 'HFL', 'HFM', 'HFN', 'HFP', 'HFQ', 'HFR', 'HFS', 'HFT', 'HFV', 'HFW', 'HFY',
             'HGA', 'HGC', 'HGD', 'HGE', 'HGF', 'HGG', 'HGH', 'HGI', 'HGK', 'HGL', 'HGM', 'HGN', 'HGP', 'HGQ', 'HGR', 'HGS', 'HGT', 'HGV', 'HGW', 'HGY',
             'HHA', 'HHC', 'HHD', 'HHE', 'HHF', 'HHG', 'HHH', 'HHI', 'HHK', 'HHL', 'HHM', 'HHN', 'HHP', 'HHQ', 'HHR', 'HHS', 'HHT', 'HHV', 'HHW', 'HHY',
             'HIA', 'HIC', 'HID', 'HIE', 'HIF', 'HIG', 'HIH', 'HII', 'HIK', 'HIL', 'HIM', 'HIN', 'HIP', 'HIQ', 'HIR', 'HIS', 'HIT', 'HIV', 'HIW', 'HIY',
             'HKA', 'HKC', 'HKD', 'HKE', 'HKF', 'HKG', 'HKH', 'HKI', 'HKK', 'HKL', 'HKM', 'HKN', 'HKP', 'HKQ', 'HKR', 'HKS', 'HKT', 'HKV', 'HKW', 'HKY',
             'HLA', 'HLC', 'HLD', 'HLE', 'HLF', 'HLG', 'HLH', 'HLI', 'HLK', 'HLL', 'HLM', 'HLN', 'HLP', 'HLQ', 'HLR', 'HLS', 'HLT', 'HLV', 'HLW', 'HLY',
             'HMA', 'HMC', 'HMD', 'HME', 'HMF', 'HMG', 'HMH', 'HMI', 'HMK', 'HML', 'HMM', 'HMN', 'HMP', 'HMQ', 'HMR', 'HMS', 'HMT', 'HMV', 'HMW', 'HMY',
             'HNA', 'HNC', 'HND', 'HNE', 'HNF', 'HNG', 'HNH', 'HNI', 'HNK', 'HNL', 'HNM', 'HNN', 'HNP', 'HNQ', 'HNR', 'HNS', 'HNT', 'HNV', 'HNW', 'HNY',
             'HPA', 'HPC', 'HPD', 'HPE', 'HPF', 'HPG', 'HPH', 'HPI', 'HPK', 'HPL', 'HPM', 'HPN', 'HPP', 'HPQ', 'HPR', 'HPS', 'HPT', 'HPV', 'HPW', 'HPY',
             'HQA', 'HQC', 'HQD', 'HQE', 'HQF', 'HQG', 'HQH', 'HQI', 'HQK', 'HQL', 'HQM', 'HQN', 'HQP', 'HQQ', 'HQR', 'HQS', 'HQT', 'HQV', 'HQW', 'HQY',
             'HRA', 'HRC', 'HRD', 'HRE', 'HRF', 'HRG', 'HRH', 'HRI', 'HRK', 'HRL', 'HRM', 'HRN', 'HRP', 'HRQ', 'HRR', 'HRS', 'HRT', 'HRV', 'HRW', 'HRY',
             'HSA', 'HSC', 'HSD', 'HSE', 'HSF', 'HSG', 'HSH', 'HSI', 'HSK', 'HSL', 'HSM', 'HSN', 'HSP', 'HSQ', 'HSR', 'HSS', 'HST', 'HSV', 'HSW', 'HSY',
             'HTA', 'HTC', 'HTD', 'HTE', 'HTF', 'HTG', 'HTH', 'HTI', 'HTK', 'HTL', 'HTM', 'HTN', 'HTP', 'HTQ', 'HTR', 'HTS', 'HTT', 'HTV', 'HTW', 'HTY',
             'HVA', 'HVC', 'HVD', 'HVE', 'HVF', 'HVG', 'HVH', 'HVI', 'HVK', 'HVL', 'HVM', 'HVN', 'HVP', 'HVQ', 'HVR', 'HVS', 'HVT', 'HVV', 'HVW', 'HVY',
             'HWA', 'HWC', 'HWD', 'HWE', 'HWF', 'HWG', 'HWH', 'HWI', 'HWK', 'HWL', 'HWM', 'HWN', 'HWP', 'HWQ', 'HWR', 'HWS', 'HWT', 'HWV', 'HWW', 'HWY',
             'HYA', 'HYC', 'HYD', 'HYE', 'HYF', 'HYG', 'HYH', 'HYI', 'HYK', 'HYL', 'HYM', 'HYN', 'HYP', 'HYQ', 'HYR', 'HYS', 'HYT', 'HYV', 'HYW', 'HYY',
             'IAA', 'IAC', 'IAD', 'IAE', 'IAF', 'IAG', 'IAH', 'IAI', 'IAK', 'IAL', 'IAM', 'IAN', 'IAP', 'IAQ', 'IAR', 'IAS', 'IAT', 'IAV', 'IAW', 'IAY',
             'ICA', 'ICC', 'ICD', 'ICE', 'ICF', 'ICG', 'ICH', 'ICI', 'ICK', 'ICL', 'ICM', 'ICN', 'ICP', 'ICQ', 'ICR', 'ICS', 'ICT', 'ICV', 'ICW', 'ICY',
             'IDA', 'IDC', 'IDD', 'IDE', 'IDF', 'IDG', 'IDH', 'IDI', 'IDK', 'IDL', 'IDM', 'IDN', 'IDP', 'IDQ', 'IDR', 'IDS', 'IDT', 'IDV', 'IDW', 'IDY',
             'IEA', 'IEC', 'IED', 'IEE', 'IEF', 'IEG', 'IEH', 'IEI', 'IEK', 'IEL', 'IEM', 'IEN', 'IEP', 'IEQ', 'IER', 'IES', 'IET', 'IEV', 'IEW', 'IEY',
             'IFA', 'IFC', 'IFD', 'IFE', 'IFF', 'IFG', 'IFH', 'IFI', 'IFK', 'IFL', 'IFM', 'IFN', 'IFP', 'IFQ', 'IFR', 'IFS', 'IFT', 'IFV', 'IFW', 'IFY',
             'IGA', 'IGC', 'IGD', 'IGE', 'IGF', 'IGG', 'IGH', 'IGI', 'IGK', 'IGL', 'IGM', 'IGN', 'IGP', 'IGQ', 'IGR', 'IGS', 'IGT', 'IGV', 'IGW', 'IGY',
             'IHA', 'IHC', 'IHD', 'IHE', 'IHF', 'IHG', 'IHH', 'IHI', 'IHK', 'IHL', 'IHM', 'IHN', 'IHP', 'IHQ', 'IHR', 'IHS', 'IHT', 'IHV', 'IHW', 'IHY',
             'IIA', 'IIC', 'IID', 'IIE', 'IIF', 'IIG', 'IIH', 'III', 'IIK', 'IIL', 'IIM', 'IIN', 'IIP', 'IIQ', 'IIR', 'IIS', 'IIT', 'IIV', 'IIW', 'IIY',
             'IKA', 'IKC', 'IKD', 'IKE', 'IKF', 'IKG', 'IKH', 'IKI', 'IKK', 'IKL', 'IKM', 'IKN', 'IKP', 'IKQ', 'IKR', 'IKS', 'IKT', 'IKV', 'IKW', 'IKY',
             'ILA', 'ILC', 'ILD', 'ILE', 'ILF', 'ILG', 'ILH', 'ILI', 'ILK', 'ILL', 'ILM', 'ILN', 'ILP', 'ILQ', 'ILR', 'ILS', 'ILT', 'ILV', 'ILW', 'ILY',
             'IMA', 'IMC', 'IMD', 'IME', 'IMF', 'IMG', 'IMH', 'IMI', 'IMK', 'IML', 'IMM', 'IMN', 'IMP', 'IMQ', 'IMR', 'IMS', 'IMT', 'IMV', 'IMW', 'IMY',
             'INA', 'INC', 'IND', 'INE', 'INF', 'ING', 'INH', 'INI', 'INK', 'INL', 'INM', 'INN', 'INP', 'INQ', 'INR', 'INS', 'INT', 'INV', 'INW', 'INY',
             'IPA', 'IPC', 'IPD', 'IPE', 'IPF', 'IPG', 'IPH', 'IPI', 'IPK', 'IPL', 'IPM', 'IPN', 'IPP', 'IPQ', 'IPR', 'IPS', 'IPT', 'IPV', 'IPW', 'IPY',
             'IQA', 'IQC', 'IQD', 'IQE', 'IQF', 'IQG', 'IQH', 'IQI', 'IQK', 'IQL', 'IQM', 'IQN', 'IQP', 'IQQ', 'IQR', 'IQS', 'IQT', 'IQV', 'IQW', 'IQY',
             'IRA', 'IRC', 'IRD', 'IRE', 'IRF', 'IRG', 'IRH', 'IRI', 'IRK', 'IRL', 'IRM', 'IRN', 'IRP', 'IRQ', 'IRR', 'IRS', 'IRT', 'IRV', 'IRW', 'IRY',
             'ISA', 'ISC', 'ISD', 'ISE', 'ISF', 'ISG', 'ISH', 'ISI', 'ISK', 'ISL', 'ISM', 'ISN', 'ISP', 'ISQ', 'ISR', 'ISS', 'IST', 'ISV', 'ISW', 'ISY',
             'ITA', 'ITC', 'ITD', 'ITE', 'ITF', 'ITG', 'ITH', 'ITI', 'ITK', 'ITL', 'ITM', 'ITN', 'ITP', 'ITQ', 'ITR', 'ITS', 'ITT', 'ITV', 'ITW', 'ITY',
             'IVA', 'IVC', 'IVD', 'IVE', 'IVF', 'IVG', 'IVH', 'IVI', 'IVK', 'IVL', 'IVM', 'IVN', 'IVP', 'IVQ', 'IVR', 'IVS', 'IVT', 'IVV', 'IVW', 'IVY',
             'IWA', 'IWC', 'IWD', 'IWE', 'IWF', 'IWG', 'IWH', 'IWI', 'IWK', 'IWL', 'IWM', 'IWN', 'IWP', 'IWQ', 'IWR', 'IWS', 'IWT', 'IWV', 'IWW', 'IWY',
             'IYA', 'IYC', 'IYD', 'IYE', 'IYF', 'IYG', 'IYH', 'IYI', 'IYK', 'IYL', 'IYM', 'IYN', 'IYP', 'IYQ', 'IYR', 'IYS', 'IYT', 'IYV', 'IYW', 'IYY',
             'KAA', 'KAC', 'KAD', 'KAE', 'KAF', 'KAG', 'KAH', 'KAI', 'KAK', 'KAL', 'KAM', 'KAN', 'KAP', 'KAQ', 'KAR', 'KAS', 'KAT', 'KAV', 'KAW', 'KAY',
             'KCA', 'KCC', 'KCD', 'KCE', 'KCF', 'KCG', 'KCH', 'KCI', 'KCK', 'KCL', 'KCM', 'KCN', 'KCP', 'KCQ', 'KCR', 'KCS', 'KCT', 'KCV', 'KCW', 'KCY',
             'KDA', 'KDC', 'KDD', 'KDE', 'KDF', 'KDG', 'KDH', 'KDI', 'KDK', 'KDL', 'KDM', 'KDN', 'KDP', 'KDQ', 'KDR', 'KDS', 'KDT', 'KDV', 'KDW', 'KDY',
             'KEA', 'KEC', 'KED', 'KEE', 'KEF', 'KEG', 'KEH', 'KEI', 'KEK', 'KEL', 'KEM', 'KEN', 'KEP', 'KEQ', 'KER', 'KES', 'KET', 'KEV', 'KEW', 'KEY',
             'KFA', 'KFC', 'KFD', 'KFE', 'KFF', 'KFG', 'KFH', 'KFI', 'KFK', 'KFL', 'KFM', 'KFN', 'KFP', 'KFQ', 'KFR', 'KFS', 'KFT', 'KFV', 'KFW', 'KFY',
             'KGA', 'KGC', 'KGD', 'KGE', 'KGF', 'KGG', 'KGH', 'KGI', 'KGK', 'KGL', 'KGM', 'KGN', 'KGP', 'KGQ', 'KGR', 'KGS', 'KGT', 'KGV', 'KGW', 'KGY',
             'KHA', 'KHC', 'KHD', 'KHE', 'KHF', 'KHG', 'KHH', 'KHI', 'KHK', 'KHL', 'KHM', 'KHN', 'KHP', 'KHQ', 'KHR', 'KHS', 'KHT', 'KHV', 'KHW', 'KHY',
             'KIA', 'KIC', 'KID', 'KIE', 'KIF', 'KIG', 'KIH', 'KII', 'KIK', 'KIL', 'KIM', 'KIN', 'KIP', 'KIQ', 'KIR', 'KIS', 'KIT', 'KIV', 'KIW', 'KIY',
             'KKA', 'KKC', 'KKD', 'KKE', 'KKF', 'KKG', 'KKH', 'KKI', 'KKK', 'KKL', 'KKM', 'KKN', 'KKP', 'KKQ', 'KKR', 'KKS', 'KKT', 'KKV', 'KKW', 'KKY',
             'KLA', 'KLC', 'KLD', 'KLE', 'KLF', 'KLG', 'KLH', 'KLI', 'KLK', 'KLL', 'KLM', 'KLN', 'KLP', 'KLQ', 'KLR', 'KLS', 'KLT', 'KLV', 'KLW', 'KLY',
             'KMA', 'KMC', 'KMD', 'KME', 'KMF', 'KMG', 'KMH', 'KMI', 'KMK', 'KML', 'KMM', 'KMN', 'KMP', 'KMQ', 'KMR', 'KMS', 'KMT', 'KMV', 'KMW', 'KMY',
             'KNA', 'KNC', 'KND', 'KNE', 'KNF', 'KNG', 'KNH', 'KNI', 'KNK', 'KNL', 'KNM', 'KNN', 'KNP', 'KNQ', 'KNR', 'KNS', 'KNT', 'KNV', 'KNW', 'KNY',
             'KPA', 'KPC', 'KPD', 'KPE', 'KPF', 'KPG', 'KPH', 'KPI', 'KPK', 'KPL', 'KPM', 'KPN', 'KPP', 'KPQ', 'KPR', 'KPS', 'KPT', 'KPV', 'KPW', 'KPY',
             'KQA', 'KQC', 'KQD', 'KQE', 'KQF', 'KQG', 'KQH', 'KQI', 'KQK', 'KQL', 'KQM', 'KQN', 'KQP', 'KQQ', 'KQR', 'KQS', 'KQT', 'KQV', 'KQW', 'KQY',
             'KRA', 'KRC', 'KRD', 'KRE', 'KRF', 'KRG', 'KRH', 'KRI', 'KRK', 'KRL', 'KRM', 'KRN', 'KRP', 'KRQ', 'KRR', 'KRS', 'KRT', 'KRV', 'KRW', 'KRY',
             'KSA', 'KSC', 'KSD', 'KSE', 'KSF', 'KSG', 'KSH', 'KSI', 'KSK', 'KSL', 'KSM', 'KSN', 'KSP', 'KSQ', 'KSR', 'KSS', 'KST', 'KSV', 'KSW', 'KSY',
             'KTA', 'KTC', 'KTD', 'KTE', 'KTF', 'KTG', 'KTH', 'KTI', 'KTK', 'KTL', 'KTM', 'KTN', 'KTP', 'KTQ', 'KTR', 'KTS', 'KTT', 'KTV', 'KTW', 'KTY',
             'KVA', 'KVC', 'KVD', 'KVE', 'KVF', 'KVG', 'KVH', 'KVI', 'KVK', 'KVL', 'KVM', 'KVN', 'KVP', 'KVQ', 'KVR', 'KVS', 'KVT', 'KVV', 'KVW', 'KVY',
             'KWA', 'KWC', 'KWD', 'KWE', 'KWF', 'KWG', 'KWH', 'KWI', 'KWK', 'KWL', 'KWM', 'KWN', 'KWP', 'KWQ', 'KWR', 'KWS', 'KWT', 'KWV', 'KWW', 'KWY',
             'KYA', 'KYC', 'KYD', 'KYE', 'KYF', 'KYG', 'KYH', 'KYI', 'KYK', 'KYL', 'KYM', 'KYN', 'KYP', 'KYQ', 'KYR', 'KYS', 'KYT', 'KYV', 'KYW', 'KYY',
             'LAA', 'LAC', 'LAD', 'LAE', 'LAF', 'LAG', 'LAH', 'LAI', 'LAK', 'LAL', 'LAM', 'LAN', 'LAP', 'LAQ', 'LAR', 'LAS', 'LAT', 'LAV', 'LAW', 'LAY',
             'LCA', 'LCC', 'LCD', 'LCE', 'LCF', 'LCG', 'LCH', 'LCI', 'LCK', 'LCL', 'LCM', 'LCN', 'LCP', 'LCQ', 'LCR', 'LCS', 'LCT', 'LCV', 'LCW', 'LCY',
             'LDA', 'LDC', 'LDD', 'LDE', 'LDF', 'LDG', 'LDH', 'LDI', 'LDK', 'LDL', 'LDM', 'LDN', 'LDP', 'LDQ', 'LDR', 'LDS', 'LDT', 'LDV', 'LDW', 'LDY',
             'LEA', 'LEC', 'LED', 'LEE', 'LEF', 'LEG', 'LEH', 'LEI', 'LEK', 'LEL', 'LEM', 'LEN', 'LEP', 'LEQ', 'LER', 'LES', 'LET', 'LEV', 'LEW', 'LEY',
             'LFA', 'LFC', 'LFD', 'LFE', 'LFF', 'LFG', 'LFH', 'LFI', 'LFK', 'LFL', 'LFM', 'LFN', 'LFP', 'LFQ', 'LFR', 'LFS', 'LFT', 'LFV', 'LFW', 'LFY',
             'LGA', 'LGC', 'LGD', 'LGE', 'LGF', 'LGG', 'LGH', 'LGI', 'LGK', 'LGL', 'LGM', 'LGN', 'LGP', 'LGQ', 'LGR', 'LGS', 'LGT', 'LGV', 'LGW', 'LGY',
             'LHA', 'LHC', 'LHD', 'LHE', 'LHF', 'LHG', 'LHH', 'LHI', 'LHK', 'LHL', 'LHM', 'LHN', 'LHP', 'LHQ', 'LHR', 'LHS', 'LHT', 'LHV', 'LHW', 'LHY',
             'LIA', 'LIC', 'LID', 'LIE', 'LIF', 'LIG', 'LIH', 'LII', 'LIK', 'LIL', 'LIM', 'LIN', 'LIP', 'LIQ', 'LIR', 'LIS', 'LIT', 'LIV', 'LIW', 'LIY',
             'LKA', 'LKC', 'LKD', 'LKE', 'LKF', 'LKG', 'LKH', 'LKI', 'LKK', 'LKL', 'LKM', 'LKN', 'LKP', 'LKQ', 'LKR', 'LKS', 'LKT', 'LKV', 'LKW', 'LKY',
             'LLA', 'LLC', 'LLD', 'LLE', 'LLF', 'LLG', 'LLH', 'LLI', 'LLK', 'LLL', 'LLM', 'LLN', 'LLP', 'LLQ', 'LLR', 'LLS', 'LLT', 'LLV', 'LLW', 'LLY',
             'LMA', 'LMC', 'LMD', 'LME', 'LMF', 'LMG', 'LMH', 'LMI', 'LMK', 'LML', 'LMM', 'LMN', 'LMP', 'LMQ', 'LMR', 'LMS', 'LMT', 'LMV', 'LMW', 'LMY',
             'LNA', 'LNC', 'LND', 'LNE', 'LNF', 'LNG', 'LNH', 'LNI', 'LNK', 'LNL', 'LNM', 'LNN', 'LNP', 'LNQ', 'LNR', 'LNS', 'LNT', 'LNV', 'LNW', 'LNY',
             'LPA', 'LPC', 'LPD', 'LPE', 'LPF', 'LPG', 'LPH', 'LPI', 'LPK', 'LPL', 'LPM', 'LPN', 'LPP', 'LPQ', 'LPR', 'LPS', 'LPT', 'LPV', 'LPW', 'LPY',
             'LQA', 'LQC', 'LQD', 'LQE', 'LQF', 'LQG', 'LQH', 'LQI', 'LQK', 'LQL', 'LQM', 'LQN', 'LQP', 'LQQ', 'LQR', 'LQS', 'LQT', 'LQV', 'LQW', 'LQY',
             'LRA', 'LRC', 'LRD', 'LRE', 'LRF', 'LRG', 'LRH', 'LRI', 'LRK', 'LRL', 'LRM', 'LRN', 'LRP', 'LRQ', 'LRR', 'LRS', 'LRT', 'LRV', 'LRW', 'LRY',
             'LSA', 'LSC', 'LSD', 'LSE', 'LSF', 'LSG', 'LSH', 'LSI', 'LSK', 'LSL', 'LSM', 'LSN', 'LSP', 'LSQ', 'LSR', 'LSS', 'LST', 'LSV', 'LSW', 'LSY',
             'LTA', 'LTC', 'LTD', 'LTE', 'LTF', 'LTG', 'LTH', 'LTI', 'LTK', 'LTL', 'LTM', 'LTN', 'LTP', 'LTQ', 'LTR', 'LTS', 'LTT', 'LTV', 'LTW', 'LTY',
             'LVA', 'LVC', 'LVD', 'LVE', 'LVF', 'LVG', 'LVH', 'LVI', 'LVK', 'LVL', 'LVM', 'LVN', 'LVP', 'LVQ', 'LVR', 'LVS', 'LVT', 'LVV', 'LVW', 'LVY',
             'LWA', 'LWC', 'LWD', 'LWE', 'LWF', 'LWG', 'LWH', 'LWI', 'LWK', 'LWL', 'LWM', 'LWN', 'LWP', 'LWQ', 'LWR', 'LWS', 'LWT', 'LWV', 'LWW', 'LWY',
             'LYA', 'LYC', 'LYD', 'LYE', 'LYF', 'LYG', 'LYH', 'LYI', 'LYK', 'LYL', 'LYM', 'LYN', 'LYP', 'LYQ', 'LYR', 'LYS', 'LYT', 'LYV', 'LYW', 'LYY', 
             'MAA', 'MAC', 'MAD', 'MAE', 'MAF', 'MAG', 'MAH', 'MAI', 'MAK', 'MAL', 'MAM', 'MAN', 'MAP', 'MAQ', 'MAR', 'MAS', 'MAT', 'MAV', 'MAW', 'MAY',
             'MCA', 'MCC', 'MCD', 'MCE', 'MCF', 'MCG', 'MCH', 'MCI', 'MCK', 'MCL', 'MCM', 'MCN', 'MCP', 'MCQ', 'MCR', 'MCS', 'MCT', 'MCV', 'MCW', 'MCY',
             'MDA', 'MDC', 'MDD', 'MDE', 'MDF', 'MDG', 'MDH', 'MDI', 'MDK', 'MDL', 'MDM', 'MDN', 'MDP', 'MDQ', 'MDR', 'MDS', 'MDT', 'MDV', 'MDW', 'MDY',
             'MEA', 'MEC', 'MED', 'MEE', 'MEF', 'MEG', 'MEH', 'MEI', 'MEK', 'MEL', 'MEM', 'MEN', 'MEP', 'MEQ', 'MER', 'MES', 'MET', 'MEV', 'MEW', 'MEY',
             'MFA', 'MFC', 'MFD', 'MFE', 'MFF', 'MFG', 'MFH', 'MFI', 'MFK', 'MFL', 'MFM', 'MFN', 'MFP', 'MFQ', 'MFR', 'MFS', 'MFT', 'MFV', 'MFW', 'MFY',
             'MGA', 'MGC', 'MGD', 'MGE', 'MGF', 'MGG', 'MGH', 'MGI', 'MGK', 'MGL', 'MGM', 'MGN', 'MGP', 'MGQ', 'MGR', 'MGS', 'MGT', 'MGV', 'MGW', 'MGY',
             'MHA', 'MHC', 'MHD', 'MHE', 'MHF', 'MHG', 'MHH', 'MHI', 'MHK', 'MHL', 'MHM', 'MHN', 'MHP', 'MHQ', 'MHR', 'MHS', 'MHT', 'MHV', 'MHW', 'MHY',
             'MIA', 'MIC', 'MID', 'MIE', 'MIF', 'MIG', 'MIH', 'MII', 'MIK', 'MIL', 'MIM', 'MIN', 'MIP', 'MIQ', 'MIR', 'MIS', 'MIT', 'MIV', 'MIW', 'MIY',
             'MKA', 'MKC', 'MKD', 'MKE', 'MKF', 'MKG', 'MKH', 'MKI', 'MKK', 'MKL', 'MKM', 'MKN', 'MKP', 'MKQ', 'MKR', 'MKS', 'MKT', 'MKV', 'MKW', 'MKY',
             'MLA', 'MLC', 'MLD', 'MLE', 'MLF', 'MLG', 'MLH', 'MLI', 'MLK', 'MLL', 'MLM', 'MLN', 'MLP', 'MLQ', 'MLR', 'MLS', 'MLT', 'MLV', 'MLW', 'MLY',
             'MMA', 'MMC', 'MMD', 'MME', 'MMF', 'MMG', 'MMH', 'MMI', 'MMK', 'MML', 'MMM', 'MMN', 'MMP', 'MMQ', 'MMR', 'MMS', 'MMT', 'MMV', 'MMW', 'MMY',
             'MNA', 'MNC', 'MND', 'MNE', 'MNF', 'MNG', 'MNH', 'MNI', 'MNK', 'MNL', 'MNM', 'MNN', 'MNP', 'MNQ', 'MNR', 'MNS', 'MNT', 'MNV', 'MNW', 'MNY',
             'MPA', 'MPC', 'MPD', 'MPE', 'MPF', 'MPG', 'MPH', 'MPI', 'MPK', 'MPL', 'MPM', 'MPN', 'MPP', 'MPQ', 'MPR', 'MPS', 'MPT', 'MPV', 'MPW', 'MPY',
             'MQA', 'MQC', 'MQD', 'MQE', 'MQF', 'MQG', 'MQH', 'MQI', 'MQK', 'MQL', 'MQM', 'MQN', 'MQP', 'MQQ', 'MQR', 'MQS', 'MQT', 'MQV', 'MQW', 'MQY',
             'MRA', 'MRC', 'MRD', 'MRE', 'MRF', 'MRG', 'MRH', 'MRI', 'MRK', 'MRL', 'MRM', 'MRN', 'MRP', 'MRQ', 'MRR', 'MRS', 'MRT', 'MRV', 'MRW', 'MRY',
             'MSA', 'MSC', 'MSD', 'MSE', 'MSF', 'MSG', 'MSH', 'MSI', 'MSK', 'MSL', 'MSM', 'MSN', 'MSP', 'MSQ', 'MSR', 'MSS', 'MST', 'MSV', 'MSW', 'MSY',
             'MTA', 'MTC', 'MTD', 'MTE', 'MTF', 'MTG', 'MTH', 'MTI', 'MTK', 'MTL', 'MTM', 'MTN', 'MTP', 'MTQ', 'MTR', 'MTS', 'MTT', 'MTV', 'MTW', 'MTY',
             'MVA', 'MVC', 'MVD', 'MVE', 'MVF', 'MVG', 'MVH', 'MVI', 'MVK', 'MVL', 'MVM', 'MVN', 'MVP', 'MVQ', 'MVR', 'MVS', 'MVT', 'MVV', 'MVW', 'MVY',
             'MWA', 'MWC', 'MWD', 'MWE', 'MWF', 'MWG', 'MWH', 'MWI', 'MWK', 'MWL', 'MWM', 'MWN', 'MWP', 'MWQ', 'MWR', 'MWS', 'MWT', 'MWV', 'MWW', 'MWY',
             'MYA', 'MYC', 'MYD', 'MYE', 'MYF', 'MYG', 'MYH', 'MYI', 'MYK', 'MYL', 'MYM', 'MYN', 'MYP', 'MYQ', 'MYR', 'MYS', 'MYT', 'MYV', 'MYW', 'MYY',
             'NAA', 'NAC', 'NAD', 'NAE', 'NAF', 'NAG', 'NAH', 'NAI', 'NAK', 'NAL', 'NAM', 'NAN', 'NAP', 'NAQ', 'NAR', 'NAS', 'NAT', 'NAV', 'NAW', 'NAY',
             'NCA', 'NCC', 'NCD', 'NCE', 'NCF', 'NCG', 'NCH', 'NCI', 'NCK', 'NCL', 'NCM', 'NCN', 'NCP', 'NCQ', 'NCR', 'NCS', 'NCT', 'NCV', 'NCW', 'NCY',
             'NDA', 'NDC', 'NDD', 'NDE', 'NDF', 'NDG', 'NDH', 'NDI', 'NDK', 'NDL', 'NDM', 'NDN', 'NDP', 'NDQ', 'NDR', 'NDS', 'NDT', 'NDV', 'NDW', 'NDY',
             'NEA', 'NEC', 'NED', 'NEE', 'NEF', 'NEG', 'NEH', 'NEI', 'NEK', 'NEL', 'NEM', 'NEN', 'NEP', 'NEQ', 'NER', 'NES', 'NET', 'NEV', 'NEW', 'NEY',
             'NFA', 'NFC', 'NFD', 'NFE', 'NFF', 'NFG', 'NFH', 'NFI', 'NFK', 'NFL', 'NFM', 'NFN', 'NFP', 'NFQ', 'NFR', 'NFS', 'NFT', 'NFV', 'NFW', 'NFY',
             'NGA', 'NGC', 'NGD', 'NGE', 'NGF', 'NGG', 'NGH', 'NGI', 'NGK', 'NGL', 'NGM', 'NGN', 'NGP', 'NGQ', 'NGR', 'NGS', 'NGT', 'NGV', 'NGW', 'NGY',
             'NHA', 'NHC', 'NHD', 'NHE', 'NHF', 'NHG', 'NHH', 'NHI', 'NHK', 'NHL', 'NHM', 'NHN', 'NHP', 'NHQ', 'NHR', 'NHS', 'NHT', 'NHV', 'NHW', 'NHY',
             'NIA', 'NIC', 'NID', 'NIE', 'NIF', 'NIG', 'NIH', 'NII', 'NIK', 'NIL', 'NIM', 'NIN', 'NIP', 'NIQ', 'NIR', 'NIS', 'NIT', 'NIV', 'NIW', 'NIY',
             'NKA', 'NKC', 'NKD', 'NKE', 'NKF', 'NKG', 'NKH', 'NKI', 'NKK', 'NKL', 'NKM', 'NKN', 'NKP', 'NKQ', 'NKR', 'NKS', 'NKT', 'NKV', 'NKW', 'NKY',
             'NLA', 'NLC', 'NLD', 'NLE', 'NLF', 'NLG', 'NLH', 'NLI', 'NLK', 'NLL', 'NLM', 'NLN', 'NLP', 'NLQ', 'NLR', 'NLS', 'NLT', 'NLV', 'NLW', 'NLY',
             'NMA', 'NMC', 'NMD', 'NME', 'NMF', 'NMG', 'NMH', 'NMI', 'NMK', 'NML', 'NMM', 'NMN', 'NMP', 'NMQ', 'NMR', 'NMS', 'NMT', 'NMV', 'NMW', 'NMY',
             'NNA', 'NNC', 'NND', 'NNE', 'NNF', 'NNG', 'NNH', 'NNI', 'NNK', 'NNL', 'NNM', 'NNN', 'NNP', 'NNQ', 'NNR', 'NNS', 'NNT', 'NNV', 'NNW', 'NNY',
             'NPA', 'NPC', 'NPD', 'NPE', 'NPF', 'NPG', 'NPH', 'NPI', 'NPK', 'NPL', 'NPM', 'NPN', 'NPP', 'NPQ', 'NPR', 'NPS', 'NPT', 'NPV', 'NPW', 'NPY',
             'NQA', 'NQC', 'NQD', 'NQE', 'NQF', 'NQG', 'NQH', 'NQI', 'NQK', 'NQL', 'NQM', 'NQN', 'NQP', 'NQQ', 'NQR', 'NQS', 'NQT', 'NQV', 'NQW', 'NQY',
             'NRA', 'NRC', 'NRD', 'NRE', 'NRF', 'NRG', 'NRH', 'NRI', 'NRK', 'NRL', 'NRM', 'NRN', 'NRP', 'NRQ', 'NRR', 'NRS', 'NRT', 'NRV', 'NRW', 'NRY',
             'NSA', 'NSC', 'NSD', 'NSE', 'NSF', 'NSG', 'NSH', 'NSI', 'NSK', 'NSL', 'NSM', 'NSN', 'NSP', 'NSQ', 'NSR', 'NSS', 'NST', 'NSV', 'NSW', 'NSY',
             'NTA', 'NTC', 'NTD', 'NTE', 'NTF', 'NTG', 'NTH', 'NTI', 'NTK', 'NTL', 'NTM', 'NTN', 'NTP', 'NTQ', 'NTR', 'NTS', 'NTT', 'NTV', 'NTW', 'NTY',
             'NVA', 'NVC', 'NVD', 'NVE', 'NVF', 'NVG', 'NVH', 'NVI', 'NVK', 'NVL', 'NVM', 'NVN', 'NVP', 'NVQ', 'NVR', 'NVS', 'NVT', 'NVV', 'NVW', 'NVY',
             'NWA', 'NWC', 'NWD', 'NWE', 'NWF', 'NWG', 'NWH', 'NWI', 'NWK', 'NWL', 'NWM', 'NWN', 'NWP', 'NWQ', 'NWR', 'NWS', 'NWT', 'NWV', 'NWW', 'NWY',
             'NYA', 'NYC', 'NYD', 'NYE', 'NYF', 'NYG', 'NYH', 'NYI', 'NYK', 'NYL', 'NYM', 'NYN', 'NYP', 'NYQ', 'NYR', 'NYS', 'NYT', 'NYV', 'NYW', 'NYY',
             'PAA', 'PAC', 'PAD', 'PAE', 'PAF', 'PAG', 'PAH', 'PAI', 'PAK', 'PAL', 'PAM', 'PAN', 'PAP', 'PAQ', 'PAR', 'PAS', 'PAT', 'PAV', 'PAW', 'PAY',
             'PCA', 'PCC', 'PCD', 'PCE', 'PCF', 'PCG', 'PCH', 'PCI', 'PCK', 'PCL', 'PCM', 'PCN', 'PCP', 'PCQ', 'PCR', 'PCS', 'PCT', 'PCV', 'PCW', 'PCY',
             'PDA', 'PDC', 'PDD', 'PDE', 'PDF', 'PDG', 'PDH', 'PDI', 'PDK', 'PDL', 'PDM', 'PDN', 'PDP', 'PDQ', 'PDR', 'PDS', 'PDT', 'PDV', 'PDW', 'PDY',
             'PEA', 'PEC', 'PED', 'PEE', 'PEF', 'PEG', 'PEH', 'PEI', 'PEK', 'PEL', 'PEM', 'PEN', 'PEP', 'PEQ', 'PER', 'PES', 'PET', 'PEV', 'PEW', 'PEY',
             'PFA', 'PFC', 'PFD', 'PFE', 'PFF', 'PFG', 'PFH', 'PFI', 'PFK', 'PFL', 'PFM', 'PFN', 'PFP', 'PFQ', 'PFR', 'PFS', 'PFT', 'PFV', 'PFW', 'PFY',
             'PGA', 'PGC', 'PGD', 'PGE', 'PGF', 'PGG', 'PGH', 'PGI', 'PGK', 'PGL', 'PGM', 'PGN', 'PGP', 'PGQ', 'PGR', 'PGS', 'PGT', 'PGV', 'PGW', 'PGY',
             'PHA', 'PHC', 'PHD', 'PHE', 'PHF', 'PHG', 'PHH', 'PHI', 'PHK', 'PHL', 'PHM', 'PHN', 'PHP', 'PHQ', 'PHR', 'PHS', 'PHT', 'PHV', 'PHW', 'PHY',
             'PIA', 'PIC', 'PID', 'PIE', 'PIF', 'PIG', 'PIH', 'PII', 'PIK', 'PIL', 'PIM', 'PIN', 'PIP', 'PIQ', 'PIR', 'PIS', 'PIT', 'PIV', 'PIW', 'PIY',
             'PKA', 'PKC', 'PKD', 'PKE', 'PKF', 'PKG', 'PKH', 'PKI', 'PKK', 'PKL', 'PKM', 'PKN', 'PKP', 'PKQ', 'PKR', 'PKS', 'PKT', 'PKV', 'PKW', 'PKY',
             'PLA', 'PLC', 'PLD', 'PLE', 'PLF', 'PLG', 'PLH', 'PLI', 'PLK', 'PLL', 'PLM', 'PLN', 'PLP', 'PLQ', 'PLR', 'PLS', 'PLT', 'PLV', 'PLW', 'PLY',
             'PMA', 'PMC', 'PMD', 'PME', 'PMF', 'PMG', 'PMH', 'PMI', 'PMK', 'PML', 'PMM', 'PMN', 'PMP', 'PMQ', 'PMR', 'PMS', 'PMT', 'PMV', 'PMW', 'PMY',
             'PNA', 'PNC', 'PND', 'PNE', 'PNF', 'PNG', 'PNH', 'PNI', 'PNK', 'PNL', 'PNM', 'PNN', 'PNP', 'PNQ', 'PNR', 'PNS', 'PNT', 'PNV', 'PNW', 'PNY',
             'PPA', 'PPC', 'PPD', 'PPE', 'PPF', 'PPG', 'PPH', 'PPI', 'PPK', 'PPL', 'PPM', 'PPN', 'PPP', 'PPQ', 'PPR', 'PPS', 'PPT', 'PPV', 'PPW', 'PPY',
             'PQA', 'PQC', 'PQD', 'PQE', 'PQF', 'PQG', 'PQH', 'PQI', 'PQK', 'PQL', 'PQM', 'PQN', 'PQP', 'PQQ', 'PQR', 'PQS', 'PQT', 'PQV', 'PQW', 'PQY',
             'PRA', 'PRC', 'PRD', 'PRE', 'PRF', 'PRG', 'PRH', 'PRI', 'PRK', 'PRL', 'PRM', 'PRN', 'PRP', 'PRQ', 'PRR', 'PRS', 'PRT', 'PRV', 'PRW', 'PRY',
             'PSA', 'PSC', 'PSD', 'PSE', 'PSF', 'PSG', 'PSH', 'PSI', 'PSK', 'PSL', 'PSM', 'PSN', 'PSP', 'PSQ', 'PSR', 'PSS', 'PST', 'PSV', 'PSW', 'PSY',
             'PTA', 'PTC', 'PTD', 'PTE', 'PTF', 'PTG', 'PTH', 'PTI', 'PTK', 'PTL', 'PTM', 'PTN', 'PTP', 'PTQ', 'PTR', 'PTS', 'PTT', 'PTV', 'PTW', 'PTY',
             'PVA', 'PVC', 'PVD', 'PVE', 'PVF', 'PVG', 'PVH', 'PVI', 'PVK', 'PVL', 'PVM', 'PVN', 'PVP', 'PVQ', 'PVR', 'PVS', 'PVT', 'PVV', 'PVW', 'PVY',
             'PWA', 'PWC', 'PWD', 'PWE', 'PWF', 'PWG', 'PWH', 'PWI', 'PWK', 'PWL', 'PWM', 'PWN', 'PWP', 'PWQ', 'PWR', 'PWS', 'PWT', 'PWV', 'PWW', 'PWY',
             'PYA', 'PYC', 'PYD', 'PYE', 'PYF', 'PYG', 'PYH', 'PYI', 'PYK', 'PYL', 'PYM', 'PYN', 'PYP', 'PYQ', 'PYR', 'PYS', 'PYT', 'PYV', 'PYW', 'PYY',
             'QAA', 'QAC', 'QAD', 'QAE', 'QAF', 'QAG', 'QAH', 'QAI', 'QAK', 'QAL', 'QAM', 'QAN', 'QAP', 'QAQ', 'QAR', 'QAS', 'QAT', 'QAV', 'QAW', 'QAY',
             'QCA', 'QCC', 'QCD', 'QCE', 'QCF', 'QCG', 'QCH', 'QCI', 'QCK', 'QCL', 'QCM', 'QCN', 'QCP', 'QCQ', 'QCR', 'QCS', 'QCT', 'QCV', 'QCW', 'QCY',
             'QDA', 'QDC', 'QDD', 'QDE', 'QDF', 'QDG', 'QDH', 'QDI', 'QDK', 'QDL', 'QDM', 'QDN', 'QDP', 'QDQ', 'QDR', 'QDS', 'QDT', 'QDV', 'QDW', 'QDY',
             'QEA', 'QEC', 'QED', 'QEE', 'QEF', 'QEG', 'QEH', 'QEI', 'QEK', 'QEL', 'QEM', 'QEN', 'QEP', 'QEQ', 'QER', 'QES', 'QET', 'QEV', 'QEW', 'QEY',
             'QFA', 'QFC', 'QFD', 'QFE', 'QFF', 'QFG', 'QFH', 'QFI', 'QFK', 'QFL', 'QFM', 'QFN', 'QFP', 'QFQ', 'QFR', 'QFS', 'QFT', 'QFV', 'QFW', 'QFY',
             'QGA', 'QGC', 'QGD', 'QGE', 'QGF', 'QGG', 'QGH', 'QGI', 'QGK', 'QGL', 'QGM', 'QGN', 'QGP', 'QGQ', 'QGR', 'QGS', 'QGT', 'QGV', 'QGW', 'QGY',
             'QHA', 'QHC', 'QHD', 'QHE', 'QHF', 'QHG', 'QHH', 'QHI', 'QHK', 'QHL', 'QHM', 'QHN', 'QHP', 'QHQ', 'QHR', 'QHS', 'QHT', 'QHV', 'QHW', 'QHY',
             'QIA', 'QIC', 'QID', 'QIE', 'QIF', 'QIG', 'QIH', 'QII', 'QIK', 'QIL', 'QIM', 'QIN', 'QIP', 'QIQ', 'QIR', 'QIS', 'QIT', 'QIV', 'QIW', 'QIY',
             'QKA', 'QKC', 'QKD', 'QKE', 'QKF', 'QKG', 'QKH', 'QKI', 'QKK', 'QKL', 'QKM', 'QKN', 'QKP', 'QKQ', 'QKR', 'QKS', 'QKT', 'QKV', 'QKW', 'QKY',
             'QLA', 'QLC', 'QLD', 'QLE', 'QLF', 'QLG', 'QLH', 'QLI', 'QLK', 'QLL', 'QLM', 'QLN', 'QLP', 'QLQ', 'QLR', 'QLS', 'QLT', 'QLV', 'QLW', 'QLY',
             'QMA', 'QMC', 'QMD', 'QME', 'QMF', 'QMG', 'QMH', 'QMI', 'QMK', 'QML', 'QMM', 'QMN', 'QMP', 'QMQ', 'QMR', 'QMS', 'QMT', 'QMV', 'QMW', 'QMY',
             'QNA', 'QNC', 'QND', 'QNE', 'QNF', 'QNG', 'QNH', 'QNI', 'QNK', 'QNL', 'QNM', 'QNN', 'QNP', 'QNQ', 'QNR', 'QNS', 'QNT', 'QNV', 'QNW', 'QNY',
             'QPA', 'QPC', 'QPD', 'QPE', 'QPF', 'QPG', 'QPH', 'QPI', 'QPK', 'QPL', 'QPM', 'QPN', 'QPP', 'QPQ', 'QPR', 'QPS', 'QPT', 'QPV', 'QPW', 'QPY',
             'QQA', 'QQC', 'QQD', 'QQE', 'QQF', 'QQG', 'QQH', 'QQI', 'QQK', 'QQL', 'QQM', 'QQN', 'QQP', 'QQQ', 'QQR', 'QQS', 'QQT', 'QQV', 'QQW', 'QQY',
             'QRA', 'QRC', 'QRD', 'QRE', 'QRF', 'QRG', 'QRH', 'QRI', 'QRK', 'QRL', 'QRM', 'QRN', 'QRP', 'QRQ', 'QRR', 'QRS', 'QRT', 'QRV', 'QRW', 'QRY',
             'QSA', 'QSC', 'QSD', 'QSE', 'QSF', 'QSG', 'QSH', 'QSI', 'QSK', 'QSL', 'QSM', 'QSN', 'QSP', 'QSQ', 'QSR', 'QSS', 'QST', 'QSV', 'QSW', 'QSY',
             'QTA', 'QTC', 'QTD', 'QTE', 'QTF', 'QTG', 'QTH', 'QTI', 'QTK', 'QTL', 'QTM', 'QTN', 'QTP', 'QTQ', 'QTR', 'QTS', 'QTT', 'QTV', 'QTW', 'QTY',
             'QVA', 'QVC', 'QVD', 'QVE', 'QVF', 'QVG', 'QVH', 'QVI', 'QVK', 'QVL', 'QVM', 'QVN', 'QVP', 'QVQ', 'QVR', 'QVS', 'QVT', 'QVV', 'QVW', 'QVY',
             'QWA', 'QWC', 'QWD', 'QWE', 'QWF', 'QWG', 'QWH', 'QWI', 'QWK', 'QWL', 'QWM', 'QWN', 'QWP', 'QWQ', 'QWR', 'QWS', 'QWT', 'QWV', 'QWW', 'QWY',
             'QYA', 'QYC', 'QYD', 'QYE', 'QYF', 'QYG', 'QYH', 'QYI', 'QYK', 'QYL', 'QYM', 'QYN', 'QYP', 'QYQ', 'QYR', 'QYS', 'QYT', 'QYV', 'QYW', 'QYY',
             'RAA', 'RAC', 'RAD', 'RAE', 'RAF', 'RAG', 'RAH', 'RAI', 'RAK', 'RAL', 'RAM', 'RAN', 'RAP', 'RAQ', 'RAR', 'RAS', 'RAT', 'RAV', 'RAW', 'RAY',
             'RCA', 'RCC', 'RCD', 'RCE', 'RCF', 'RCG', 'RCH', 'RCI', 'RCK', 'RCL', 'RCM', 'RCN', 'RCP', 'RCQ', 'RCR', 'RCS', 'RCT', 'RCV', 'RCW', 'RCY',
             'RDA', 'RDC', 'RDD', 'RDE', 'RDF', 'RDG', 'RDH', 'RDI', 'RDK', 'RDL', 'RDM', 'RDN', 'RDP', 'RDQ', 'RDR', 'RDS', 'RDT', 'RDV', 'RDW', 'RDY',
             'REA', 'REC', 'RED', 'REE', 'REF', 'REG', 'REH', 'REI', 'REK', 'REL', 'REM', 'REN', 'REP', 'REQ', 'RER', 'RES', 'RET', 'REV', 'REW', 'REY',
             'RFA', 'RFC', 'RFD', 'RFE', 'RFF', 'RFG', 'RFH', 'RFI', 'RFK', 'RFL', 'RFM', 'RFN', 'RFP', 'RFQ', 'RFR', 'RFS', 'RFT', 'RFV', 'RFW', 'RFY',
             'RGA', 'RGC', 'RGD', 'RGE', 'RGF', 'RGG', 'RGH', 'RGI', 'RGK', 'RGL', 'RGM', 'RGN', 'RGP', 'RGQ', 'RGR', 'RGS', 'RGT', 'RGV', 'RGW', 'RGY',
             'RHA', 'RHC', 'RHD', 'RHE', 'RHF', 'RHG', 'RHH', 'RHI', 'RHK', 'RHL', 'RHM', 'RHN', 'RHP', 'RHQ', 'RHR', 'RHS', 'RHT', 'RHV', 'RHW', 'RHY',
             'RIA', 'RIC', 'RID', 'RIE', 'RIF', 'RIG', 'RIH', 'RII', 'RIK', 'RIL', 'RIM', 'RIN', 'RIP', 'RIQ', 'RIR', 'RIS', 'RIT', 'RIV', 'RIW', 'RIY',
             'RKA', 'RKC', 'RKD', 'RKE', 'RKF', 'RKG', 'RKH', 'RKI', 'RKK', 'RKL', 'RKM', 'RKN', 'RKP', 'RKQ', 'RKR', 'RKS', 'RKT', 'RKV', 'RKW', 'RKY',
             'RLA', 'RLC', 'RLD', 'RLE', 'RLF', 'RLG', 'RLH', 'RLI', 'RLK', 'RLL', 'RLM', 'RLN', 'RLP', 'RLQ', 'RLR', 'RLS', 'RLT', 'RLV', 'RLW', 'RLY',
             'RMA', 'RMC', 'RMD', 'RME', 'RMF', 'RMG', 'RMH', 'RMI', 'RMK', 'RML', 'RMM', 'RMN', 'RMP', 'RMQ', 'RMR', 'RMS', 'RMT', 'RMV', 'RMW', 'RMY',
             'RNA', 'RNC', 'RND', 'RNE', 'RNF', 'RNG', 'RNH', 'RNI', 'RNK', 'RNL', 'RNM', 'RNN', 'RNP', 'RNQ', 'RNR', 'RNS', 'RNT', 'RNV', 'RNW', 'RNY',
             'RPA', 'RPC', 'RPD', 'RPE', 'RPF', 'RPG', 'RPH', 'RPI', 'RPK', 'RPL', 'RPM', 'RPN', 'RPP', 'RPQ', 'RPR', 'RPS', 'RPT', 'RPV', 'RPW', 'RPY',
             'RQA', 'RQC', 'RQD', 'RQE', 'RQF', 'RQG', 'RQH', 'RQI', 'RQK', 'RQL', 'RQM', 'RQN', 'RQP', 'RQQ', 'RQR', 'RQS', 'RQT', 'RQV', 'RQW', 'RQY',
             'RRA', 'RRC', 'RRD', 'RRE', 'RRF', 'RRG', 'RRH', 'RRI', 'RRK', 'RRL', 'RRM', 'RRN', 'RRP', 'RRQ', 'RRR', 'RRS', 'RRT', 'RRV', 'RRW', 'RRY',
             'RSA', 'RSC', 'RSD', 'RSE', 'RSF', 'RSG', 'RSH', 'RSI', 'RSK', 'RSL', 'RSM', 'RSN', 'RSP', 'RSQ', 'RSR', 'RSS', 'RST', 'RSV', 'RSW', 'RSY',
             'RTA', 'RTC', 'RTD', 'RTE', 'RTF', 'RTG', 'RTH', 'RTI', 'RTK', 'RTL', 'RTM', 'RTN', 'RTP', 'RTQ', 'RTR', 'RTS', 'RTT', 'RTV', 'RTW', 'RTY',
             'RVA', 'RVC', 'RVD', 'RVE', 'RVF', 'RVG', 'RVH', 'RVI', 'RVK', 'RVL', 'RVM', 'RVN', 'RVP', 'RVQ', 'RVR', 'RVS', 'RVT', 'RVV', 'RVW', 'RVY',
             'RWA', 'RWC', 'RWD', 'RWE', 'RWF', 'RWG', 'RWH', 'RWI', 'RWK', 'RWL', 'RWM', 'RWN', 'RWP', 'RWQ', 'RWR', 'RWS', 'RWT', 'RWV', 'RWW', 'RWY',
             'RYA', 'RYC', 'RYD', 'RYE', 'RYF', 'RYG', 'RYH', 'RYI', 'RYK', 'RYL', 'RYM', 'RYN', 'RYP', 'RYQ', 'RYR', 'RYS', 'RYT', 'RYV', 'RYW', 'RYY',
             'SAA', 'SAC', 'SAD', 'SAE', 'SAF', 'SAG', 'SAH', 'SAI', 'SAK', 'SAL', 'SAM', 'SAN', 'SAP', 'SAQ', 'SAR', 'SAS', 'SAT', 'SAV', 'SAW', 'SAY',
             'SCA', 'SCC', 'SCD', 'SCE', 'SCF', 'SCG', 'SCH', 'SCI', 'SCK', 'SCL', 'SCM', 'SCN', 'SCP', 'SCQ', 'SCR', 'SCS', 'SCT', 'SCV', 'SCW', 'SCY',
             'SDA', 'SDC', 'SDD', 'SDE', 'SDF', 'SDG', 'SDH', 'SDI', 'SDK', 'SDL', 'SDM', 'SDN', 'SDP', 'SDQ', 'SDR', 'SDS', 'SDT', 'SDV', 'SDW', 'SDY',
             'SEA', 'SEC', 'SED', 'SEE', 'SEF', 'SEG', 'SEH', 'SEI', 'SEK', 'SEL', 'SEM', 'SEN', 'SEP', 'SEQ', 'SER', 'SES', 'SET', 'SEV', 'SEW', 'SEY',
             'SFA', 'SFC', 'SFD', 'SFE', 'SFF', 'SFG', 'SFH', 'SFI', 'SFK', 'SFL', 'SFM', 'SFN', 'SFP', 'SFQ', 'SFR', 'SFS', 'SFT', 'SFV', 'SFW', 'SFY',
             'SGA', 'SGC', 'SGD', 'SGE', 'SGF', 'SGG', 'SGH', 'SGI', 'SGK', 'SGL', 'SGM', 'SGN', 'SGP', 'SGQ', 'SGR', 'SGS', 'SGT', 'SGV', 'SGW', 'SGY',
             'SHA', 'SHC', 'SHD', 'SHE', 'SHF', 'SHG', 'SHH', 'SHI', 'SHK', 'SHL', 'SHM', 'SHN', 'SHP', 'SHQ', 'SHR', 'SHS', 'SHT', 'SHV', 'SHW', 'SHY',
             'SIA', 'SIC', 'SID', 'SIE', 'SIF', 'SIG', 'SIH', 'SII', 'SIK', 'SIL', 'SIM', 'SIN', 'SIP', 'SIQ', 'SIR', 'SIS', 'SIT', 'SIV', 'SIW', 'SIY',
             'SKA', 'SKC', 'SKD', 'SKE', 'SKF', 'SKG', 'SKH', 'SKI', 'SKK', 'SKL', 'SKM', 'SKN', 'SKP', 'SKQ', 'SKR', 'SKS', 'SKT', 'SKV', 'SKW', 'SKY',
             'SLA', 'SLC', 'SLD', 'SLE', 'SLF', 'SLG', 'SLH', 'SLI', 'SLK', 'SLL', 'SLM', 'SLN', 'SLP', 'SLQ', 'SLR', 'SLS', 'SLT', 'SLV', 'SLW', 'SLY',
             'SMA', 'SMC', 'SMD', 'SME', 'SMF', 'SMG', 'SMH', 'SMI', 'SMK', 'SML', 'SMM', 'SMN', 'SMP', 'SMQ', 'SMR', 'SMS', 'SMT', 'SMV', 'SMW', 'SMY',
             'SNA', 'SNC', 'SND', 'SNE', 'SNF', 'SNG', 'SNH', 'SNI', 'SNK', 'SNL', 'SNM', 'SNN', 'SNP', 'SNQ', 'SNR', 'SNS', 'SNT', 'SNV', 'SNW', 'SNY',
             'SPA', 'SPC', 'SPD', 'SPE', 'SPF', 'SPG', 'SPH', 'SPI', 'SPK', 'SPL', 'SPM', 'SPN', 'SPP', 'SPQ', 'SPR', 'SPS', 'SPT', 'SPV', 'SPW', 'SPY',
             'SQA', 'SQC', 'SQD', 'SQE', 'SQF', 'SQG', 'SQH', 'SQI', 'SQK', 'SQL', 'SQM', 'SQN', 'SQP', 'SQQ', 'SQR', 'SQS', 'SQT', 'SQV', 'SQW', 'SQY',
             'SRA', 'SRC', 'SRD', 'SRE', 'SRF', 'SRG', 'SRH', 'SRI', 'SRK', 'SRL', 'SRM', 'SRN', 'SRP', 'SRQ', 'SRR', 'SRS', 'SRT', 'SRV', 'SRW', 'SRY',
             'SSA', 'SSC', 'SSD', 'SSE', 'SSF', 'SSG', 'SSH', 'SSI', 'SSK', 'SSL', 'SSM', 'SSN', 'SSP', 'SSQ', 'SSR', 'SSS', 'SST', 'SSV', 'SSW', 'SSY',
             'STA', 'STC', 'STD', 'STE', 'STF', 'STG', 'STH', 'STI', 'STK', 'STL', 'STM', 'STN', 'STP', 'STQ', 'STR', 'STS', 'STT', 'STV', 'STW', 'STY',
             'SVA', 'SVC', 'SVD', 'SVE', 'SVF', 'SVG', 'SVH', 'SVI', 'SVK', 'SVL', 'SVM', 'SVN', 'SVP', 'SVQ', 'SVR', 'SVS', 'SVT', 'SVV', 'SVW', 'SVY',
             'SWA', 'SWC', 'SWD', 'SWE', 'SWF', 'SWG', 'SWH', 'SWI', 'SWK', 'SWL', 'SWM', 'SWN', 'SWP', 'SWQ', 'SWR', 'SWS', 'SWT', 'SWV', 'SWW', 'SWY',
             'SYA', 'SYC', 'SYD', 'SYE', 'SYF', 'SYG', 'SYH', 'SYI', 'SYK', 'SYL', 'SYM', 'SYN', 'SYP', 'SYQ', 'SYR', 'SYS', 'SYT', 'SYV', 'SYW', 'SYY',
             'TAA', 'TAC', 'TAD', 'TAE', 'TAF', 'TAG', 'TAH', 'TAI', 'TAK', 'TAL', 'TAM', 'TAN', 'TAP', 'TAQ', 'TAR', 'TAS', 'TAT', 'TAV', 'TAW', 'TAY',
             'TCA', 'TCC', 'TCD', 'TCE', 'TCF', 'TCG', 'TCH', 'TCI', 'TCK', 'TCL', 'TCM', 'TCN', 'TCP', 'TCQ', 'TCR', 'TCS', 'TCT', 'TCV', 'TCW', 'TCY',
             'TDA', 'TDC', 'TDD', 'TDE', 'TDF', 'TDG', 'TDH', 'TDI', 'TDK', 'TDL', 'TDM', 'TDN', 'TDP', 'TDQ', 'TDR', 'TDS', 'TDT', 'TDV', 'TDW', 'TDY',
             'TEA', 'TEC', 'TED', 'TEE', 'TEF', 'TEG', 'TEH', 'TEI', 'TEK', 'TEL', 'TEM', 'TEN', 'TEP', 'TEQ', 'TER', 'TES', 'TET', 'TEV', 'TEW', 'TEY',
             'TFA', 'TFC', 'TFD', 'TFE', 'TFF', 'TFG', 'TFH', 'TFI', 'TFK', 'TFL', 'TFM', 'TFN', 'TFP', 'TFQ', 'TFR', 'TFS', 'TFT', 'TFV', 'TFW', 'TFY',
             'TGA', 'TGC', 'TGD', 'TGE', 'TGF', 'TGG', 'TGH', 'TGI', 'TGK', 'TGL', 'TGM', 'TGN', 'TGP', 'TGQ', 'TGR', 'TGS', 'TGT', 'TGV', 'TGW', 'TGY',
             'THA', 'THC', 'THD', 'THE', 'THF', 'THG', 'THH', 'THI', 'THK', 'THL', 'THM', 'THN', 'THP', 'THQ', 'THR', 'THS', 'THT', 'THV', 'THW', 'THY',
             'TIA', 'TIC', 'TID', 'TIE', 'TIF', 'TIG', 'TIH', 'TII', 'TIK', 'TIL', 'TIM', 'TIN', 'TIP', 'TIQ', 'TIR', 'TIS', 'TIT', 'TIV', 'TIW', 'TIY',
             'TKA', 'TKC', 'TKD', 'TKE', 'TKF', 'TKG', 'TKH', 'TKI', 'TKK', 'TKL', 'TKM', 'TKN', 'TKP', 'TKQ', 'TKR', 'TKS', 'TKT', 'TKV', 'TKW', 'TKY',
             'TLA', 'TLC', 'TLD', 'TLE', 'TLF', 'TLG', 'TLH', 'TLI', 'TLK', 'TLL', 'TLM', 'TLN', 'TLP', 'TLQ', 'TLR', 'TLS', 'TLT', 'TLV', 'TLW', 'TLY',
             'TMA', 'TMC', 'TMD', 'TME', 'TMF', 'TMG', 'TMH', 'TMI', 'TMK', 'TML', 'TMM', 'TMN', 'TMP', 'TMQ', 'TMR', 'TMS', 'TMT', 'TMV', 'TMW', 'TMY',
             'TNA', 'TNC', 'TND', 'TNE', 'TNF', 'TNG', 'TNH', 'TNI', 'TNK', 'TNL', 'TNM', 'TNN', 'TNP', 'TNQ', 'TNR', 'TNS', 'TNT', 'TNV', 'TNW', 'TNY',
             'TPA', 'TPC', 'TPD', 'TPE', 'TPF', 'TPG', 'TPH', 'TPI', 'TPK', 'TPL', 'TPM', 'TPN', 'TPP', 'TPQ', 'TPR', 'TPS', 'TPT', 'TPV', 'TPW', 'TPY',
             'TQA', 'TQC', 'TQD', 'TQE', 'TQF', 'TQG', 'TQH', 'TQI', 'TQK', 'TQL', 'TQM', 'TQN', 'TQP', 'TQQ', 'TQR', 'TQS', 'TQT', 'TQV', 'TQW', 'TQY',
             'TRA', 'TRC', 'TRD', 'TRE', 'TRF', 'TRG', 'TRH', 'TRI', 'TRK', 'TRL', 'TRM', 'TRN', 'TRP', 'TRQ', 'TRR', 'TRS', 'TRT', 'TRV', 'TRW', 'TRY',
             'TSA', 'TSC', 'TSD', 'TSE', 'TSF', 'TSG', 'TSH', 'TSI', 'TSK', 'TSL', 'TSM', 'TSN', 'TSP', 'TSQ', 'TSR', 'TSS', 'TST', 'TSV', 'TSW', 'TSY',
             'TTA', 'TTC', 'TTD', 'TTE', 'TTF', 'TTG', 'TTH', 'TTI', 'TTK', 'TTL', 'TTM', 'TTN', 'TTP', 'TTQ', 'TTR', 'TTS', 'TTT', 'TTV', 'TTW', 'TTY',
             'TVA', 'TVC', 'TVD', 'TVE', 'TVF', 'TVG', 'TVH', 'TVI', 'TVK', 'TVL', 'TVM', 'TVN', 'TVP', 'TVQ', 'TVR', 'TVS', 'TVT', 'TVV', 'TVW', 'TVY',
             'TWA', 'TWC', 'TWD', 'TWE', 'TWF', 'TWG', 'TWH', 'TWI', 'TWK', 'TWL', 'TWM', 'TWN', 'TWP', 'TWQ', 'TWR', 'TWS', 'TWT', 'TWV', 'TWW', 'TWY',
             'TYA', 'TYC', 'TYD', 'TYE', 'TYF', 'TYG', 'TYH', 'TYI', 'TYK', 'TYL', 'TYM', 'TYN', 'TYP', 'TYQ', 'TYR', 'TYS', 'TYT', 'TYV', 'TYW', 'TYY',
             'VAA', 'VAC', 'VAD', 'VAE', 'VAF', 'VAG', 'VAH', 'VAI', 'VAK', 'VAL', 'VAM', 'VAN', 'VAP', 'VAQ', 'VAR', 'VAS', 'VAT', 'VAV', 'VAW', 'VAY',
             'VCA', 'VCC', 'VCD', 'VCE', 'VCF', 'VCG', 'VCH', 'VCI', 'VCK', 'VCL', 'VCM', 'VCN', 'VCP', 'VCQ', 'VCR', 'VCS', 'VCT', 'VCV', 'VCW', 'VCY',
             'VDA', 'VDC', 'VDD', 'VDE', 'VDF', 'VDG', 'VDH', 'VDI', 'VDK', 'VDL', 'VDM', 'VDN', 'VDP', 'VDQ', 'VDR', 'VDS', 'VDT', 'VDV', 'VDW', 'VDY',
             'VEA', 'VEC', 'VED', 'VEE', 'VEF', 'VEG', 'VEH', 'VEI', 'VEK', 'VEL', 'VEM', 'VEN', 'VEP', 'VEQ', 'VER', 'VES', 'VET', 'VEV', 'VEW', 'VEY',
             'VFA', 'VFC', 'VFD', 'VFE', 'VFF', 'VFG', 'VFH', 'VFI', 'VFK', 'VFL', 'VFM', 'VFN', 'VFP', 'VFQ', 'VFR', 'VFS', 'VFT', 'VFV', 'VFW', 'VFY',
             'VGA', 'VGC', 'VGD', 'VGE', 'VGF', 'VGG', 'VGH', 'VGI', 'VGK', 'VGL', 'VGM', 'VGN', 'VGP', 'VGQ', 'VGR', 'VGS', 'VGT', 'VGV', 'VGW', 'VGY',
             'VHA', 'VHC', 'VHD', 'VHE', 'VHF', 'VHG', 'VHH', 'VHI', 'VHK', 'VHL', 'VHM', 'VHN', 'VHP', 'VHQ', 'VHR', 'VHS', 'VHT', 'VHV', 'VHW', 'VHY',
             'VIA', 'VIC', 'VID', 'VIE', 'VIF', 'VIG', 'VIH', 'VII', 'VIK', 'VIL', 'VIM', 'VIN', 'VIP', 'VIQ', 'VIR', 'VIS', 'VIT', 'VIV', 'VIW', 'VIY',
             'VKA', 'VKC', 'VKD', 'VKE', 'VKF', 'VKG', 'VKH', 'VKI', 'VKK', 'VKL', 'VKM', 'VKN', 'VKP', 'VKQ', 'VKR', 'VKS', 'VKT', 'VKV', 'VKW', 'VKY',
             'VLA', 'VLC', 'VLD', 'VLE', 'VLF', 'VLG', 'VLH', 'VLI', 'VLK', 'VLL', 'VLM', 'VLN', 'VLP', 'VLQ', 'VLR', 'VLS', 'VLT', 'VLV', 'VLW', 'VLY',
             'VMA', 'VMC', 'VMD', 'VME', 'VMF', 'VMG', 'VMH', 'VMI', 'VMK', 'VML', 'VMM', 'VMN', 'VMP', 'VMQ', 'VMR', 'VMS', 'VMT', 'VMV', 'VMW', 'VMY',
             'VNA', 'VNC', 'VND', 'VNE', 'VNF', 'VNG', 'VNH', 'VNI', 'VNK', 'VNL', 'VNM', 'VNN', 'VNP', 'VNQ', 'VNR', 'VNS', 'VNT', 'VNV', 'VNW', 'VNY',
             'VPA', 'VPC', 'VPD', 'VPE', 'VPF', 'VPG', 'VPH', 'VPI', 'VPK', 'VPL', 'VPM', 'VPN', 'VPP', 'VPQ', 'VPR', 'VPS', 'VPT', 'VPV', 'VPW', 'VPY',
             'VQA', 'VQC', 'VQD', 'VQE', 'VQF', 'VQG', 'VQH', 'VQI', 'VQK', 'VQL', 'VQM', 'VQN', 'VQP', 'VQQ', 'VQR', 'VQS', 'VQT', 'VQV', 'VQW', 'VQY',
             'VRA', 'VRC', 'VRD', 'VRE', 'VRF', 'VRG', 'VRH', 'VRI', 'VRK', 'VRL', 'VRM', 'VRN', 'VRP', 'VRQ', 'VRR', 'VRS', 'VRT', 'VRV', 'VRW', 'VRY',
             'VSA', 'VSC', 'VSD', 'VSE', 'VSF', 'VSG', 'VSH', 'VSI', 'VSK', 'VSL', 'VSM', 'VSN', 'VSP', 'VSQ', 'VSR', 'VSS', 'VST', 'VSV', 'VSW', 'VSY',
             'VTA', 'VTC', 'VTD', 'VTE', 'VTF', 'VTG', 'VTH', 'VTI', 'VTK', 'VTL', 'VTM', 'VTN', 'VTP', 'VTQ', 'VTR', 'VTS', 'VTT', 'VTV', 'VTW', 'VTY',
             'VVA', 'VVC', 'VVD', 'VVE', 'VVF', 'VVG', 'VVH', 'VVI', 'VVK', 'VVL', 'VVM', 'VVN', 'VVP', 'VVQ', 'VVR', 'VVS', 'VVT', 'VVV', 'VVW', 'VVY',
             'VWA', 'VWC', 'VWD', 'VWE', 'VWF', 'VWG', 'VWH', 'VWI', 'VWK', 'VWL', 'VWM', 'VWN', 'VWP', 'VWQ', 'VWR', 'VWS', 'VWT', 'VWV', 'VWW', 'VWY',
             'VYA', 'VYC', 'VYD', 'VYE', 'VYF', 'VYG', 'VYH', 'VYI', 'VYK', 'VYL', 'VYM', 'VYN', 'VYP', 'VYQ', 'VYR', 'VYS', 'VYT', 'VYV', 'VYW', 'VYY',
             'WAA', 'WAC', 'WAD', 'WAE', 'WAF', 'WAG', 'WAH', 'WAI', 'WAK', 'WAL', 'WAM', 'WAN', 'WAP', 'WAQ', 'WAR', 'WAS', 'WAT', 'WAV', 'WAW', 'WAY',
             'WCA', 'WCC', 'WCD', 'WCE', 'WCF', 'WCG', 'WCH', 'WCI', 'WCK', 'WCL', 'WCM', 'WCN', 'WCP', 'WCQ', 'WCR', 'WCS', 'WCT', 'WCV', 'WCW', 'WCY',
             'WDA', 'WDC', 'WDD', 'WDE', 'WDF', 'WDG', 'WDH', 'WDI', 'WDK', 'WDL', 'WDM', 'WDN', 'WDP', 'WDQ', 'WDR', 'WDS', 'WDT', 'WDV', 'WDW', 'WDY',
             'WEA', 'WEC', 'WED', 'WEE', 'WEF', 'WEG', 'WEH', 'WEI', 'WEK', 'WEL', 'WEM', 'WEN', 'WEP', 'WEQ', 'WER', 'WES', 'WET', 'WEV', 'WEW', 'WEY',
             'WFA', 'WFC', 'WFD', 'WFE', 'WFF', 'WFG', 'WFH', 'WFI', 'WFK', 'WFL', 'WFM', 'WFN', 'WFP', 'WFQ', 'WFR', 'WFS', 'WFT', 'WFV', 'WFW', 'WFY',
             'WGA', 'WGC', 'WGD', 'WGE', 'WGF', 'WGG', 'WGH', 'WGI', 'WGK', 'WGL', 'WGM', 'WGN', 'WGP', 'WGQ', 'WGR', 'WGS', 'WGT', 'WGV', 'WGW', 'WGY',
             'WHA', 'WHC', 'WHD', 'WHE', 'WHF', 'WHG', 'WHH', 'WHI', 'WHK', 'WHL', 'WHM', 'WHN', 'WHP', 'WHQ', 'WHR', 'WHS', 'WHT', 'WHV', 'WHW', 'WHY',
             'WIA', 'WIC', 'WID', 'WIE', 'WIF', 'WIG', 'WIH', 'WII', 'WIK', 'WIL', 'WIM', 'WIN', 'WIP', 'WIQ', 'WIR', 'WIS', 'WIT', 'WIV', 'WIW', 'WIY',
             'WKA', 'WKC', 'WKD', 'WKE', 'WKF', 'WKG', 'WKH', 'WKI', 'WKK', 'WKL', 'WKM', 'WKN', 'WKP', 'WKQ', 'WKR', 'WKS', 'WKT', 'WKV', 'WKW', 'WKY',
             'WLA', 'WLC', 'WLD', 'WLE', 'WLF', 'WLG', 'WLH', 'WLI', 'WLK', 'WLL', 'WLM', 'WLN', 'WLP', 'WLQ', 'WLR', 'WLS', 'WLT', 'WLV', 'WLW', 'WLY',
             'WMA', 'WMC', 'WMD', 'WME', 'WMF', 'WMG', 'WMH', 'WMI', 'WMK', 'WML', 'WMM', 'WMN', 'WMP', 'WMQ', 'WMR', 'WMS', 'WMT', 'WMV', 'WMW', 'WMY',
             'WNA', 'WNC', 'WND', 'WNE', 'WNF', 'WNG', 'WNH', 'WNI', 'WNK', 'WNL', 'WNM', 'WNN', 'WNP', 'WNQ', 'WNR', 'WNS', 'WNT', 'WNV', 'WNW', 'WNY',
             'WPA', 'WPC', 'WPD', 'WPE', 'WPF', 'WPG', 'WPH', 'WPI', 'WPK', 'WPL', 'WPM', 'WPN', 'WPP', 'WPQ', 'WPR', 'WPS', 'WPT', 'WPV', 'WPW', 'WPY',
             'WQA', 'WQC', 'WQD', 'WQE', 'WQF', 'WQG', 'WQH', 'WQI', 'WQK', 'WQL', 'WQM', 'WQN', 'WQP', 'WQQ', 'WQR', 'WQS', 'WQT', 'WQV', 'WQW', 'WQY',
             'WRA', 'WRC', 'WRD', 'WRE', 'WRF', 'WRG', 'WRH', 'WRI', 'WRK', 'WRL', 'WRM', 'WRN', 'WRP', 'WRQ', 'WRR', 'WRS', 'WRT', 'WRV', 'WRW', 'WRY',
             'WSA', 'WSC', 'WSD', 'WSE', 'WSF', 'WSG', 'WSH', 'WSI', 'WSK', 'WSL', 'WSM', 'WSN', 'WSP', 'WSQ', 'WSR', 'WSS', 'WST', 'WSV', 'WSW', 'WSY',
             'WTA', 'WTC', 'WTD', 'WTE', 'WTF', 'WTG', 'WTH', 'WTI', 'WTK', 'WTL', 'WTM', 'WTN', 'WTP', 'WTQ', 'WTR', 'WTS', 'WTT', 'WTV', 'WTW', 'WTY',
             'WVA', 'WVC', 'WVD', 'WVE', 'WVF', 'WVG', 'WVH', 'WVI', 'WVK', 'WVL', 'WVM', 'WVN', 'WVP', 'WVQ', 'WVR', 'WVS', 'WVT', 'WVV', 'WVW', 'WVY',
             'WWA', 'WWC', 'WWD', 'WWE', 'WWF', 'WWG', 'WWH', 'WWI', 'WWK', 'WWL', 'WWM', 'WWN', 'WWP', 'WWQ', 'WWR', 'WWS', 'WWT', 'WWV', 'WWW', 'WWY',
             'WYA', 'WYC', 'WYD', 'WYE', 'WYF', 'WYG', 'WYH', 'WYI', 'WYK', 'WYL', 'WYM', 'WYN', 'WYP', 'WYQ', 'WYR', 'WYS', 'WYT', 'WYV', 'WYW', 'WYY', 
             'YAA', 'YAC', 'YAD', 'YAE', 'YAF', 'YAG', 'YAH', 'YAI', 'YAK', 'YAL', 'YAM', 'YAN', 'YAP', 'YAQ', 'YAR', 'YAS', 'YAT', 'YAV', 'YAW', 'YAY',
             'YCA', 'YCC', 'YCD', 'YCE', 'YCF', 'YCG', 'YCH', 'YCI', 'YCK', 'YCL', 'YCM', 'YCN', 'YCP', 'YCQ', 'YCR', 'YCS', 'YCT', 'YCV', 'YCW', 'YCY',
             'YDA', 'YDC', 'YDD', 'YDE', 'YDF', 'YDG', 'YDH', 'YDI', 'YDK', 'YDL', 'YDM', 'YDN', 'YDP', 'YDQ', 'YDR', 'YDS', 'YDT', 'YDV', 'YDW', 'YDY',
             'YEA', 'YEC', 'YED', 'YEE', 'YEF', 'YEG', 'YEH', 'YEI', 'YEK', 'YEL', 'YEM', 'YEN', 'YEP', 'YEQ', 'YER', 'YES', 'YET', 'YEV', 'YEW', 'YEY',
             'YFA', 'YFC', 'YFD', 'YFE', 'YFF', 'YFG', 'YFH', 'YFI', 'YFK', 'YFL', 'YFM', 'YFN', 'YFP', 'YFQ', 'YFR', 'YFS', 'YFT', 'YFV', 'YFW', 'YFY',
             'YGA', 'YGC', 'YGD', 'YGE', 'YGF', 'YGG', 'YGH', 'YGI', 'YGK', 'YGL', 'YGM', 'YGN', 'YGP', 'YGQ', 'YGR', 'YGS', 'YGT', 'YGV', 'YGW', 'YGY',
             'YHA', 'YHC', 'YHD', 'YHE', 'YHF', 'YHG', 'YHH', 'YHI', 'YHK', 'YHL', 'YHM', 'YHN', 'YHP', 'YHQ', 'YHR', 'YHS', 'YHT', 'YHV', 'YHW', 'YHY',
             'YIA', 'YIC', 'YID', 'YIE', 'YIF', 'YIG', 'YIH', 'YII', 'YIK', 'YIL', 'YIM', 'YIN', 'YIP', 'YIQ', 'YIR', 'YIS', 'YIT', 'YIV', 'YIW', 'YIY',
             'YKA', 'YKC', 'YKD', 'YKE', 'YKF', 'YKG', 'YKH', 'YKI', 'YKK', 'YKL', 'YKM', 'YKN', 'YKP', 'YKQ', 'YKR', 'YKS', 'YKT', 'YKV', 'YKW', 'YKY',
             'YLA', 'YLC', 'YLD', 'YLE', 'YLF', 'YLG', 'YLH', 'YLI', 'YLK', 'YLL', 'YLM', 'YLN', 'YLP', 'YLQ', 'YLR', 'YLS', 'YLT', 'YLV', 'YLW', 'YLY',
             'YMA', 'YMC', 'YMD', 'YME', 'YMF', 'YMG', 'YMH', 'YMI', 'YMK', 'YML', 'YMM', 'YMN', 'YMP', 'YMQ', 'YMR', 'YMS', 'YMT', 'YMV', 'YMW', 'YMY',
             'YNA', 'YNC', 'YND', 'YNE', 'YNF', 'YNG', 'YNH', 'YNI', 'YNK', 'YNL', 'YNM', 'YNN', 'YNP', 'YNQ', 'YNR', 'YNS', 'YNT', 'YNV', 'YNW', 'YNY',
             'YPA', 'YPC', 'YPD', 'YPE', 'YPF', 'YPG', 'YPH', 'YPI', 'YPK', 'YPL', 'YPM', 'YPN', 'YPP', 'YPQ', 'YPR', 'YPS', 'YPT', 'YPV', 'YPW', 'YPY',
             'YQA', 'YQC', 'YQD', 'YQE', 'YQF', 'YQG', 'YQH', 'YQI', 'YQK', 'YQL', 'YQM', 'YQN', 'YQP', 'YQQ', 'YQR', 'YQS', 'YQT', 'YQV', 'YQW', 'YQY',
             'YRA', 'YRC', 'YRD', 'YRE', 'YRF', 'YRG', 'YRH', 'YRI', 'YRK', 'YRL', 'YRM', 'YRN', 'YRP', 'YRQ', 'YRR', 'YRS', 'YRT', 'YRV', 'YRW', 'YRY',
             'YSA', 'YSC', 'YSD', 'YSE', 'YSF', 'YSG', 'YSH', 'YSI', 'YSK', 'YSL', 'YSM', 'YSN', 'YSP', 'YSQ', 'YSR', 'YSS', 'YST', 'YSV', 'YSW', 'YSY',
             'YTA', 'YTC', 'YTD', 'YTE', 'YTF', 'YTG', 'YTH', 'YTI', 'YTK', 'YTL', 'YTM', 'YTN', 'YTP', 'YTQ', 'YTR', 'YTS', 'YTT', 'YTV', 'YTW', 'YTY',
             'YVA', 'YVC', 'YVD', 'YVE', 'YVF', 'YVG', 'YVH', 'YVI', 'YVK', 'YVL', 'YVM', 'YVN', 'YVP', 'YVQ', 'YVR', 'YVS', 'YVT', 'YVV', 'YVW', 'YVY',
             'YWA', 'YWC', 'YWD', 'YWE', 'YWF', 'YWG', 'YWH', 'YWI', 'YWK', 'YWL', 'YWM', 'YWN', 'YWP', 'YWQ', 'YWR', 'YWS', 'YWT', 'YWV', 'YWW', 'YWY',
             'YYA', 'YYC', 'YYD', 'YYE', 'YYF', 'YYG', 'YYH', 'YYI', 'YYK', 'YYL', 'YYM', 'YYN', 'YYP', 'YYQ', 'YYR', 'YYS', 'YYT', 'YYV', 'YYW', 'YYY',
             'Norm_mass', 'Norm_isoelectric_point', 'Norm_net_charge', 'Norm_gravy', 'Norm_hidro/total', 'norm_hydrophobic_hydrophobic_CKSAAGP',
             'norm_hydrophobic_polar_non_charged_CKSAAGP', 'norm_hydrophobic_positively_charged_CKSAAGP', 'norm_hydrophobic_negatively_charged_CKSAAGP',
             'norm_polar_non_charged_hydrophobic_CKSAAGP', 'norm_polar_non_charged_polar_non_charged_CKSAAGP', 'norm_polar_non_charged_positively_charged_CKSAAGP',
             'norm_polar_non_charged_negatively_charged_CKSAAGP', 'norm_positively_charged_hydrophobic_CKSAAGP', 'norm_positively_charged_polar_non_charged_CKSAAGP',
             'norm_positively_charged_positively_charged_CKSAAGP', 'norm_positively_charged_negatively_charged_CKSAAGP', 'norm_negatively_charged_hydrophobic_CKSAAGP',
             'norm_negatively_charged_polar_non_charged_CKSAAGP', 'norm_negatively_charged_positively_charged_CKSAAGP', 'norm_negatively_charged_negatively_charged_CKSAAGP']

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


amino_acid_groups = {
    'hydrophobic': ['A', 'V', 'L', 'I', 'M', 'F', 'W', 'C'],
    'polar_non_charged': ['S', 'T', 'N', 'Q'],
    'positively_charged': ['K', 'R', 'H'],
    'negatively_charged': ['D', 'E']
}
def group_amino_acid(amino):
    for group, amino_list in amino_acid_groups.items():
        if amino in amino_list:
            return group
    return None

def calculate_ksaagp(sequence, k):
    length = len(sequence)
    k_spaced_pairs = {f"{g1}-{g2}": 0 for g1 in amino_acid_groups for g2 in amino_acid_groups}

    for i in range(length - k - 1):
        first_amino = sequence[i]
        second_amino = sequence[i + k + 1]

        group1 = group_amino_acid(first_amino)
        group2 = group_amino_acid(second_amino)

        if group1 and group2:
            pair = f"{group1}-{group2}"
            k_spaced_pairs[pair] += 1

    # Normalização pelo comprimento da cadeia
    for pair in k_spaced_pairs:
        k_spaced_pairs[pair] /= length

    return k_spaced_pairs



##############################################################################################################################################
########################################################### FEATURE MATRIX ####################################################################
##############################################################################################################################################

def calc_properties(file_path):
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
    matrix = np.zeros((list_len, len(cabecalho)), dtype=object) 

    scale = ProtParamData.gravy_scales.get('KyteDoolitle')

    for i, seq in enumerate(sequence_list):
        sequence = ProteinAnalysis(seq)
        seq_str = str(sequence.sequence)
        matrix[i][0] = str(sequence.sequence)  
        matrix[i][1] = sequence.length   
        sequence.count_amino_acids()

        for j, amino in enumerate(['R', 'K', 'A', 'L', 'G', 'C', 'W', 'P', 'H']): 
            matrix[i][(j+1)*2] = sequence.amino_acids_content[amino]                        
            matrix[i][(j+1)*2 + 1] = sequence.amino_acids_content[amino] / sequence.length  

        matrix[i][20] = sequence.molecular_weight()   
        matrix[i][21] = sequence.isoelectric_point()  
        matrix[i][22] = sequence.charge_at_pH(7.0)    

        gravy = sequence.gravy()
        matrix[i][23] = gravy

       
        hydrophilic_residues = 0
        for a in sequence.sequence:
            if scale[a] < 0:
                hydrophilic_residues += 1

        matrix[i][24] = hydrophilic_residues / sequence.length

        
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

        
        matrix[i][827] -= 2 * (sequence.length - 1) # H2
        matrix[i][831] -= sequence.length - 1 # 0

        total_elements = np.sum(matrix[i, [825, 827, 829, 831, 833]])
        matrix[i][835] = total_elements
        for j in range(5):
            matrix[i][826 + (j * 2)] = matrix[i][825 + (j * 2)] / total_elements

        for j in range(sequence.length - 3 + 1):
            tri_id = 836 + (utils.aminos_id[sequence.sequence[j]] * 400 + 
              utils.aminos_id[sequence.sequence[j + 1]] * 20 + 
              utils.aminos_id[sequence.sequence[j + 2]])
            matrix[i][tri_id] += (1 / (sequence.length - 3 + 1))
        
        matrix[i][8836] = (sequence.molecular_weight() - 516.6314)/(5121.012000000002 - 516.6314)   
        matrix[i][8837] = (sequence.isoelectric_point() - 4.0500284194946286)/ (12.999967765808105 - 4.0500284194946286)
        matrix[i][8838] = (sequence.charge_at_pH(7.0) + 8.173321821201688)/(23.75392760999528 + 8.173321821201688)  
        matrix[i][8839] = (gravy + 4.5)/(2.7111111111111112 + 4.5) 
        matrix[i][8840] = hydrophilic_residues / sequence.length

        k_spaced_pairs = calculate_ksaagp(seq, k=1)
        start_index = 8841  

       
        for j, pair in enumerate(k_spaced_pairs):
            matrix[i][start_index + j] = k_spaced_pairs.get(pair, 0)

    return matrix


############################################################################################################################################



##############################################################################################################################################
########################################################### MATRIX GENERATION ####################################################################
##############################################################################################################################################


def matrix_generation(data):
    matrix = calc_properties(data)
    df = pd.DataFrame(matrix, columns=cabecalho).infer_objects()
    df.to_csv(f'cpps-matrix.csv', index=False)
    print(f'Training Matrix was generated and save as CSV.')
    return df

############################################################################################################################################


# Main Function
def process_cpps(data):

    #Load model
    model_cpp = pickle.load(open('PERSEU_MODEL.pkl', 'rb'))
    model_eff = pickle.load(open('PERSEU-Efficiency.pkl', 'rb'))

    # Feature Generation for classication
    df_class = matrix_generation(data)

    # Selecting just the top features were the model was trained
    X_cpps = df_class[model_cpp.feature_names_in_]

    # Classification Part
    y_cpps_probs = model_cpp.predict_proba(X_cpps)[:, 1]
    y_cpps_pred = ['CPP' if p >= 0.5 else 'NON-CPP' for p in y_cpps_probs]
    df_cpp = pd.DataFrame({
            'seq': df_class.iloc[:, 0],
            'Classification': y_cpps_pred,
        })
    

    eff_df = df_class.merge(df_cpp, on='seq')
    eff_df= eff_df[eff_df['Classification'] == 'CPP']
    eff_df = eff_df.drop(columns=['Classification'])

    # Selecting just the top features were the model was trained
    X_cpps_eff = eff_df[model_eff.feature_names_in_]
    
    # Efficiency Classification Part
    y_cpps_probs_eff = model_eff.predict_proba(X_cpps_eff)[:, 1]
    y_cpps_pred_eff = ['High' if p >= 0.5 else 'Low' for p in y_cpps_probs_eff]
        
    df_cpp_eff = pd.DataFrame({
            'seq': eff_df.iloc[:, 0],
            'Classification': y_cpps_pred_eff,
        })
    

    df_result = df_cpp.merge(df_cpp_eff, on='seq', how='left', suffixes=('_cpp', '_eff'))

    df_result['Classification_eff'] = df_result['Classification_eff'].fillna('-')
    
    df_result.to_csv(f'Results.csv', index=False)
    print('Prediction results successfully generated!')


##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################





print("""
      To run PERSEUcpp, you only need to enter the path of the desired file.
      For example, if you have the file cpps-test.fasta, simply enter its full name and submit it.
      The CPPs and their respective efficiencies will be predicted.


The model accepts both FASTA and CSV files.

FASTA Format:
- You must follow the standard FASTA file format.

CSV Format:
- A single-column file containing only the sequences.
""")

data = input("\nData path:")

process_cpps(data)

