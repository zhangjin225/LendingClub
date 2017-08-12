#Edited  by Jin Zhang 07-21-2017
#To use this script, please download LoanStats3c.csv and LoanStats3d.csv for 2014 and 2015 respectively, then put
#LoanStats3c.csv and LoanStats3d.csv with this script
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import csv
from collections import Counter
from datetime import datetime
import statistics
from scipy.stats.stats import pearsonr
from sklearn.linear_model import LinearRegression
######what is the median loan amount?###13750.0 (2014-2015)############################################################
######What fraction of all loans are for the most commom purpose?#####0.5984644995462325 (2014-2015)###############################
######What is the ratio of min average rate to the max arverage rate? #####0.6399797767058072 (2014-2015)##############################
######What is the difference in the fraction of the loans with a 36-month term between 2014 and 2015?######0.017472334236841358##########
######What is the standard deviation of this ratio for all the loans in default?#####0.19781888999345731 (2014-2015)######################
######What is the Pearson correlation coefficient?#######-0.0689489497041#########################################################
######What is the highest ratio of B/A?################43.1856381929############################################################
######What is the deviation of the actual default rate from the predicted value?#######0.06079198423682923############################
data_loan1=[]
data_pupo1=[]
typ_pupo1=[]
data_int_rate1=[]
typ_int_rate1=[]
int_rate1=[]
term1=[]
loan_status1=[]
issue_d1=[]
last_pymnt_d1=[]
total_pymnt1=[]
loan_amnt_term=[]
total_pymnt_term=[]
int_rate1=[]
int_rate_term=[]
addr_state1=[]
sub_grade1=[]
issue_d1_orig=[]
loan_status1_orig=[]
sub_grade1_orig=[]
with open ("LoanStats3c.csv", "r") as loanfour:
    next(loanfour, None)
    loanfour = csv.reader(loanfour)
    col_count = len(next(loanfour)) # read first line and count columns
with open ("LoanStats3c.csv", "r") as loanfour:
    next(loanfour, None)   #skip the first line
    loanfour = csv.reader(loanfour)   
    loanfour=list(loanfour)
    #print(data)
    row_count=len (loanfour)
    for i in range(col_count):
        if "loan_amnt" in loanfour[0][i]:
            a=i
        if "purpose" in loanfour[0][i]:
            b=i
        if "int_rate" in loanfour[0][i]:
            c=i
        if "term" in loanfour[0][i]:
            d=i
        if "loan_status" in loanfour[0][i]:
            e=i
        if "issue_d" in loanfour[0][i]:
            f=i
        if "last_pymnt_d" in loanfour[0][i]:
            g=i
        if "total_pymnt" in loanfour[0][i]:
            h=i
        if "int_rate" in loanfour[0][i]:
            k=i
        if "addr_state" in loanfour[0][i]:
            p=i
        if "sub_grade" in loanfour[0][i]:
            q=i
    for i in range(1, row_count-4):
        n=loanfour[i][a]
        data_loan1.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfour[i][b]
        data_pupo1.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfour[i][c]
        n=n.strip('%')     #delete '%' from numbers
        data_int_rate1.extend(n.split())
    for i  in range(1, row_count-4):
        n=loanfour[i][d]
        n=n.strip('months')
        term1.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfour[i][e]
        loan_status1.extend(n.split(','))
    for i in range(1, row_count-4):
        n=loanfour[i][f]
        n=n.replace("-", ",")
        issue_d1.extend(n.split())
    for i in range(1, row_count-4):
        if loanfour[i][g]=="":
            loanfour[i][g]=loanfour[i][f]
        n=loanfour[i][g]
        n=n.replace("-", ",")
        last_pymnt_d1.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfour[i][h]
        total_pymnt1.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfour[i][k]
        n=n.strip('%') 
        int_rate1.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfour[i][p]
        addr_state1.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfour[i][q]
        sub_grade1.extend(n.split())  
loan_amnt_1=data_loan1
total_rows_four=len(loan_amnt_1)
purpose_1=data_pupo1
purpose_1_counter=Counter(purpose_1)
n_typ_pupo1 = len(purpose_1_counter)
pupo1=purpose_1_counter.most_common()
most_com=purpose_1_counter.most_common(1)
most_com_ele=most_com[0][0]
most_com_fre=most_com[0][1]
for i in range(len(issue_d1)):
    issue_d1_orig.append(issue_d1[i])
for i in range(len(loan_status1)):
    loan_status1_orig.append(loan_status1[i])
for i in range(len(sub_grade1)):
    sub_grade1_orig.append(sub_grade1[i])
#######################begin: Pearson correlation coefficient##########

for i in range(len(issue_d1)):
    issue_d=datetime.strptime(issue_d1[i], "%b,%Y")
    if issue_d.month < 7:
        loan_amnt_term.append(loan_amnt_1[i])
        total_pymnt_term.append(total_pymnt1[i])
        int_rate_term.append(int_rate1[i])
loan_amnt_term=[float(i) for i in loan_amnt_term]
total_pymnt_term=[float(i) for i in total_pymnt_term]
int_rate_term=[float(i) for i in int_rate_term]
int_rate_term=[i/100 for i in int_rate_term]
loan_amnt_term=np.asarray(loan_amnt_term)
total_pymnt_term=np.asarray(total_pymnt_term)
int_rate_term=np.asarray(int_rate_term)
total_rate_re=(total_pymnt_term-loan_amnt_term)/loan_amnt_term
pearson_cor=pearsonr(total_rate_re, int_rate_term)[0]    ###answer6
####print(pearson_cor)
#######################begin: Pearson correlation coefficient##########

#####################begin: faction of the loans within a 36-month term for 2014########
n_term1=len(term1)
term_counter=Counter(term1)
n_threyears=term_counter["36"]
data_term1=term_counter.most_common()
fac_threyears1=n_threyears/n_term1   ###
#####################end: faction of the loans within a 36-month term for 2014########
data_loan2=[]
data_pupo2=[]
typ_pupo2=[]
data_int_rate2=[]
typ_int_rate2=[]
int_rate2=[]
term2=[]
term2=[]
loan_status2=[]
issue_d2=[]
last_pymnt_d2=[]
issue_d_con_total=[]
last_pymnt_d_con_total=[]
diff_mon=[]
term_con_total=[]
addr_state2=[]
sub_grade2=[]
with open ("LoanStats3d.csv", "r") as loanfive:
    next(loanfive, None)
    loanfive = csv.reader(loanfive)
    col_count = len(next(loanfive)) # read first line and count columns
with open ("LoanStats3d.csv", "r") as loanfive:
    next(loanfive, None)   #skip the first line
    loanfive = csv.reader(loanfive)   
    loanfive=list(loanfive)
    row_count=len (loanfive)
    for i in range(col_count):
        if "loan_amnt" in loanfive[0][i]:
            a=i
        if "purpose" in loanfive[0][i]:
            b=i
        if "int_rate" in loanfive[0][i]:
            c=i
        if "term" in loanfive[0][i]:
            d=i
        if "loan_status" in loanfive[0][i]:
            e=i
        if "issue_d" in loanfive[0][i]:
            f=i
        if "last_pymnt_d" in loanfive[0][i]:
            g=i
        if "addr_state" in loanfive[0][i]:
            p=i
        if "sub_grade" in loanfive[0][i]:
            q=i
    for i in range(1, row_count-4):
        n=loanfive[i][a]
        data_loan2.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfive[i][b]
        data_pupo2.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfive[i][c]
        n=n.strip('%')     #delete '%' from numbers
        data_int_rate2.extend(n.split())
    for i  in range(1, row_count-4):
        n=loanfive[i][d]
        n=n.strip('months')
        term2.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfive[i][e]
        loan_status2.extend(n.split(','))
    for i in range(1, row_count-4):
        n=loanfive[i][f]
        n=n.replace("-", ",")
        issue_d2.extend(n.split())
    for i in range(1, row_count-4):
        if loanfive[i][g]=="":
            loanfive[i][g]=loanfive[i][f]
        n=loanfive[i][g]
        n=n.replace("-", ",")
        last_pymnt_d2.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfive[i][p]
        addr_state2.extend(n.split())
    for i in range(1, row_count-4):
        n=loanfive[i][q]
        sub_grade2.extend(n.split())
        
loan_amnt_2=data_loan2
total_rows_five=len(loan_amnt_2)
purpose_2=data_pupo2
purpose_2_counter=Counter(purpose_2)
n_typ_pupo2 = len(purpose_2_counter)
pupo2=purpose_2_counter.most_common()
most_com=purpose_2_counter.most_common(1)
most_com_ele=most_com[0][0]
most_com_fre=most_com[0][1]
######################begin: the fraction of all loans are for the most commom purpose##############
total_rows=total_rows_five+total_rows_four
fra_most_com = most_com_fre/total_rows  ###answer2
#print(fra_most_com)
######################begin: the fraction of all loans are for the most commom purpose##############

##################begin: the median loan amount##########################################
loan_amnt_1.extend(loan_amnt_2)
loan_amnt = np.asarray(loan_amnt_1, dtype=np.int)
med=np.median(loan_amnt)  ### answer1
#print(med)
##################begin: the median loan amount##########################################
    
#####################begin: the difference in faction of the loans within a 36-month term for 2014########
n_term2=len(term2)
term_counter=Counter(term2)
n_threyears=term_counter["36"]
data_term2=term_counter.most_common()
fac_threyears2=n_threyears/n_term2
diff=abs(fac_threyears1-fac_threyears2) #### answer4
print(diff)
#####################end: the difference in faction of the loans within a 36-month term for 2014########

#####################begin: standard deviation of the ratio for all the loans in default for 2014-2015########
term1.extend(term2)
loan_status1.extend(loan_status2)
issue_d1.extend(issue_d2)
last_pymnt_d1.extend(last_pymnt_d2)
len_loan_status1=len(loan_status1)
term1 = [int(i) for i in term1]
for i in range(len_loan_status1):
    if loan_status1[i] != 'Fully Paid' and  loan_status1[i] != 'Current' and loan_status1[i] != 'In Grace Period':
        issue_d_con_total.append(issue_d1[i])
        last_pymnt_d_con_total.append(last_pymnt_d1[i])
        term_con_total.append(term1[i])
len_issue_d_con_total=len(issue_d_con_total)
for i in range(len_issue_d_con_total):
    d1=datetime.strptime(last_pymnt_d_con_total[i], "%b,%Y")
    d2=datetime.strptime(issue_d_con_total[i], "%b,%Y")
    diff_mon.append((d1.year - d2.year) * 12 + d1.month - d2.month)
diff_mon=np.asarray(diff_mon, dtype = np.float64)
term_con_total=np.asarray(term_con_total, dtype = np.float64)
ratio_loans=diff_mon/term_con_total
ratio_loans= np.array(ratio_loans).tolist()
std_dev_ratio=statistics.stdev(ratio_loans)  ###answer5
print(std_dev_ratio)
#####################end: standard deviation of the ratio for all the loans in default for 2014-2015#######

#####################begin: ratio of min average rate to the max average rate for 2014-2015##########################
typ_pupo_total=[]
typ_int_rate_total=[]
int_rate_total=[]
purpose_1.extend(purpose_2)
data_int_rate1.extend(data_int_rate2)
total_rows=len(purpose_1)
g=Counter(purpose_1)
n_typ_pupo_total=len(g)
pupo_total=g.most_common()

for i in range(n_typ_pupo_total):
    a=pupo_total[i][0]
    typ_pupo_total.append(a)
    

for i in range(n_typ_pupo_total):
    b=typ_pupo_total[i]
    #typ_int_rate1.append(b)
    for i in range(total_rows):
        if b == purpose_1[i]:
            #print(data_int_rate1[i])
            typ_int_rate_total.append(data_int_rate1[i])
    typ_int_rate_total=[float(i) for i in typ_int_rate_total]
    #print(typ_int_rate_total)
    int_rate_total.append(sum(typ_int_rate_total)/len(typ_int_rate_total))
    typ_int_rate_total= []
ratio_min_max=min(int_rate_total)/max(int_rate_total)  ### the answer3

#######print(ratio_min_max)
#####################end: ratio of min average rate to the max average rate for 2014-2015##########################

#########begin: the highest ratio of B/A for 2014-2015#########################################################
purpose_a_ratio=[]
state_purpose=[]
state_name=[]
state_num=[]
addr_state_b=[]
addr_stat_a_num=[]
purpose_1 = purpose_1
addr_state1.extend(addr_state2)
purpose_1_len=len(purpose_1)
addr_state1_len=len(addr_state1)
purpose_total=Counter(purpose_1)
purpose_total_a=purpose_total.most_common()
purpose_total_a_len=len(purpose_total_a)
addr_state_total=Counter(addr_state1)
addr_state_total_a=addr_state_total.most_common()

for i in range(purpose_total_a_len):
    state_purpose.append(purpose_total_a[i][0]) ### order by this purpose
    purpose_a_ratio.append(purpose_total_a[i][1]/purpose_1_len)  # A posibility


for i in range(purpose_total_a_len):
    state_name.append(addr_state_total_a[i][0])  ##state name
    state_num.append(addr_state_total_a[i][1])  # B posibility----fen mu


for i in range(purpose_total_a_len): 
    addr_state_b=[]
    for j in range(purpose_1_len):
        if purpose_1[j] == state_purpose[i]:
            addr_state_b.append(addr_state1[j])
    addr_state_b_cou=Counter(addr_state_b)
    addr_state_b_cou_most=addr_state_b_cou.most_common(1)

    addr_stat_a_num.append(addr_state_b_cou_most[0][1])
addr_stat_a_num=np.asarray(addr_stat_a_num)
state_num=np.asarray(state_num)
purpose_a_ratio=np.asarray(purpose_a_ratio)
purpose_b_ratio=addr_stat_a_num/state_num ### B posibility
#print(purpose_b_ratio/purpose_a_ratio)

hig_ratio=max(purpose_b_ratio/purpose_a_ratio)   ### the answer7

#########end: the highest ratio of B/A for 2014-2015#####################################################

############begin: the deviation of the actual default rate from the predicted value########################

sub_grade=[]
data_int_rate_total=[]
avg_int_rate_total=[]
sub_grade_total_spe=[]
sub_grade_a=[]
deft_rate=[]
deft_rate_no=[]
deft_rate_no_ag=[]
final=[]
sub_grade1.extend(sub_grade2)
sub_grade1_len=len(sub_grade1)
data_int_rate1_len=len(data_int_rate1)
sub_grade_total=Counter(sub_grade1)
sub_grade_total=sub_grade_total.most_common()
sub_grade_total_len=len(sub_grade_total)

for i in range(sub_grade_total_len):
    sub_grade_total_spe.append(sub_grade_total[i][1])
    sub_grade=[]
    data_int_rate_total=[]
    for j in range(sub_grade1_len):
        if  sub_grade_total[i][0]==sub_grade1[j]:
            sub_grade.append(sub_grade1[j])
            data_int_rate_total.append(data_int_rate1[j])
    data_int_rate_total=[float(i) for i in data_int_rate_total]
    data_int_rate_total=np.asarray(data_int_rate_total)
    avg_int_rate = np.mean(data_int_rate_total) /100 #### average interest rate
    avg_int_rate_total.append(sub_grade_total[i][0])
    avg_int_rate_total.append(avg_int_rate )
avg_int_rate_total=avg_int_rate_total      #### average interest rate order by sub_group

for i in range(len(issue_d1_orig)):
    issue_d=datetime.strptime(issue_d1_orig[i], "%b,%Y")
    if issue_d.month < 7:
        if  loan_status1[i]!="Fully Paid":
            sub_grade_a.append(sub_grade1_orig[i])
            
sub_grade_a=Counter(sub_grade_a)
sub_grade_a=sub_grade_a.most_common()
sub_grade_a_len=len(sub_grade_a)
for i in range(sub_grade_a_len):
    for j in range(sub_grade_total_len):
        if sub_grade_a[i][0]==sub_grade_total[j][0]:
            deft_rate.append(sub_grade_a[i][0])
            deft_rate.append(sub_grade_a[i][1]/sub_grade_total[j][1])
        if sub_grade_a[i][0]!=sub_grade_total[j][0] :
            deft_rate_no.append(sub_grade_total[j][0])


for i in range(sub_grade_a_len):
        deft_rate_no.remove(sub_grade_a[i][0])
#print(deft_rate_no)

deft_rate_no=Counter(deft_rate_no)
deft_rate_no=deft_rate_no.most_common()
for i in range(len(deft_rate_no)):
    deft_rate_no_ag.append(deft_rate_no[i][0])
    deft_rate_no_ag.append(0)
deft_rate_no_ag=deft_rate_no_ag


deft_rate.extend(deft_rate_no_ag)
deft_rate=np.asarray(deft_rate )  ####default rage
shap_def=deft_rate.shape
row=int(deft_rate.shape[0]/2)
deft_rate=deft_rate.reshape(row, 2)

avg_int_rate_total=np.asarray(avg_int_rate_total) 
shap_avg_int_rate_total=avg_int_rate_total.shape
row2=int(avg_int_rate_total.shape[0]/2)
avg_int_rate_total=avg_int_rate_total.reshape(row2, 2)
for i in range(len(deft_rate)):
    for j in range(len(avg_int_rate_total)):
        if deft_rate[i][0]==avg_int_rate_total[j][0]:
            final.append(deft_rate[i][0])
            final.append(avg_int_rate_total[j][1])
            final.append(deft_rate[i][1])
            
final=final
final=np.asarray(final) 
final_shap=final.shape
row3=int(final.shape[0]/3)
final=final.reshape(row3, 3)

average_int_rate=[]
for i in range(len(final)):
    average_int_rate.append([float(final[i, 1])])
#print(average_int_rate)

average_fault_rate=[]
for i in range(len(final)):
    average_fault_rate.append([float(final[i, 2])])

X=average_int_rate
y=average_fault_rate

model=LinearRegression()
model.fit(X, y)
y2=model.predict(X)

y2_new=[]
for i in range(len(y2)):
    y2_new.append([float(y2[i, 0])])
y2_new=np.asarray(y2_new)
y=np.asarray(y)
dev=max(abs(y2-y))
dev=np.array(dev).tolist()   #### the answer
print(dev)
