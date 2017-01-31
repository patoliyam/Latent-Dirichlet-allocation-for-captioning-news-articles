#!/usr/bin/python -tt

'''
This prgram finds precision, recall , accuracy , jacard coef ,Rand Index, purity for the LDA results 
'''
import os
import sys
import pickle
from itertools import combinations
from datetime import datetime


def evaluation_matrix(fname):
	f=open(fname,'r')
	line_list=f.readlines()
	doc2event={}
	doc_list =[]
	event_list=[]
	model_op = {}
	for i,line in enumerate(line_list):
		cur_line_list = line.strip().split(' : ')
		doc_id = cur_line_list[0]
		topic = cur_line_list[1]
		doc2event[doc_id] = topic
		doc_list.append(doc_id)
		#event_list.append(topic)
	doc_list.sort()
	tp=tn=fp=fn =0

	doc_pair_list = list(combinations(doc_list,2))
	doc_pair_list.sort()
	print 'len of list ', len(doc_pair_list)
	cur_label = ''
	for doc1,doc2 in doc_pair_list:
		ev1 = '_'.join( doc1.strip().split('_')[:2])
		ev2 = '_'.join( doc2.strip().split('_')[:2])
		#print ev1, ev2
		if ev1 not in event_list :
		    event_list.append(ev1)
		if ev2 not in event_list :
		    event_list.append(ev2)
		if doc2event[doc1] == doc2event[doc2] :
		    if ev1 == ev2 :
		        tp +=1
		        cur_label = 'tp'
		    else:
		        fp +=1 
		        cur_label = 'fp'
		else:
		    if ev1 == ev2 :
		        fn +=1
		        cur_label = 'fn'
		    else:
		        tn +=1
		        cur_label = 'tn'	        
		model_op[(doc1,doc2)] = { 'op' : doc2event[doc1] + ' : ' + doc2event[doc2]  , 'label' : cur_label }
		
	precision = ( tp * 1.0 ) / (tp + fp)

	recall =  ( tp * 1.0 ) / (tp + fn)

	rand_index  =  ( 1.0 * tp + tn) / (tp + tn + fp + fn ) 

	jc = ( 1.0 * tp ) / ( tp + fp + fn)

	print ' tp =%d , fp=%d, tn =%d, fn =%d , precision =%f , recall =%f, rand_index = %f, jc=%f' %(tp,fp,tn,fn,precision,recall,rand_index,jc)

	print 'Length of Doc : %d , len of TOpic : %d ' %(len(doc_list), len(event_list))


	d ={}
	d['tp'] = tp
	d['tn'] =tn
	d['fn'] = fn
	d['fp'] = fp

	d['precision'] =precision
	d['recall'] = recall
	d['jc'] = jc
	d['rand_index'] = rand_index
	'''fname = sys.argv[1] + '_result.pkl'
	f=open(fname,'w')
	pickle.dump(d,f)
	f.lcose()


	fname = sys.argv[1] + '_result.json'
	f=open(fname,'w')
	json.dump(d,f,indent=4)
	f.lcose()
	'''
	return d, model_op

def main():
	
	fname_lda = sys.argv[1]
	fname_tot_lda = sys.argv[2]
	lda_ev , model_op_lda = evaluation_matrix(fname_lda)
	tot_ev , model_op_tot_lda = evaluation_matrix(fname_tot_lda)
 	
 	suffix = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
	fname = 'tot_lda_ev_%s.csv' %(suffix)
	f=open(fname,'w')
	st = ' , LDA, TOT_LDA\n\n'
	f.write(st)
	
	ev_list =[ 'tp', 'tn' , 'fp', 'fn', 'precision' , 'recall', 'rand_index', 'jc']
  
  	for i,item in enumerate(ev_list,1):
  		if i <5:
  			st = st = '%s, %d, %d \n' %(item, lda_ev[item], tot_ev[item])
  		else:
  			st = '%s, %0.6f, %0.6f \n' %(item, lda_ev[item], tot_ev[item])
  		f.write(st)
  	f.close()
  	
  	# Find tp of Model 1 moves to tp/tn/fn/fp of Model 2 ; Model 1 : LDA , Model 2 : TOT LDA
  	
  	
  	key_list_1 =   model_op_lda.keys()
  	key_list_2 = model_op_tot_lda.keys()
  	key_list_1.sort()
  	key_list_2.sort()
  	print ' keys ::  %s'  %(key_list_1[:5])
  	print ' keys :: %s '  %(key_list_2[:5]) 
  	
  	tp_count = fp_count = tn_count = fn_count = 0
  	total_count = 0
  	for i,doc_pair in enumerate(model_op_lda):
  		#print ' Doc pair :: %d , %s'  %(i, doc_pair)
  		if model_op_lda[doc_pair]['label'] == 'tp':
  			total_count +=1
			if model_op_tot_lda[doc_pair]['label'] == 'tp':
				tp_count += 1 
		   	elif model_op_tot_lda[doc_pair]['label'] == 'tn':  			
				tn_count += 1
				print model_op_lda[doc_pair]['op'] , model_op_tot_lda[doc_pair]['op'] ,doc_pair
			elif model_op_tot_lda[doc_pair]['label'] == 'fp':
				fp_count += 1
			elif model_op_tot_lda[doc_pair]['label'] == 'fn':  		
  				fn_count += 1 
  	# Find tn of Model 1 moves to tp/tn/fn/fp of Model 2
  	
  	print 'Moved from Tp :::: Tp = %d, Tn = %d , fp =%d , fn =%d , total_Tp_count_LDA =%d ' %(tp_count, tn_count, fp_count, fn_count, total_count) 
  	
  	'''
  	tp_count = fp_count = tn_count = fn_count = 0
  	total_count = 0  	
  	for i,doc_pair in enumerate(model_op_lda):
  		#print ' Doc pair :: %d , %s'  %(i, doc_pair)
  		if model_op_lda[doc_pair]['label'] == 'tn':
  			total_count +=1
			if model_op_tot_lda[doc_pair]['label'] == 'tp':
				tp_count += 1 
		   	elif model_op_tot_lda[doc_pair]['label'] == 'tn':  			
				tn_count += 1
			elif model_op_tot_lda[doc_pair]['label'] == 'fp':
				fp_count += 1
			elif model_op_tot_lda[doc_pair]['label'] == 'fn':  		
  				fn_count += 1 
  	# Find tn of Model 1 moves to tp/tn/fn/fp of Model 2
  	
  	print ' Moved from TN :::: Tp = %d, Tn = %d , fp =%d , fn =%d , total_TN_count_LDA =%d ' %(tp_count, tn_count, fp_count, fn_count, total_count) 


	tp_count = fp_count = tn_count = fn_count = 0
  	total_count = 0  	
  	for i,doc_pair in enumerate(model_op_lda):
  		#print ' Doc pair :: %d , %s'  %(i, doc_pair)
  		if model_op_lda[doc_pair]['label'] == 'fp':
  			total_count +=1
			if model_op_tot_lda[doc_pair]['label'] == 'tp':
				tp_count += 1 
		   	elif model_op_tot_lda[doc_pair]['label'] == 'tn':  			
				tn_count += 1
			elif model_op_tot_lda[doc_pair]['label'] == 'fp':
				fp_count += 1
			elif model_op_tot_lda[doc_pair]['label'] == 'fn':  		
  				fn_count += 1 
  	# Find tn of Model 1 moves to tp/tn/fn/fp of Model 2
  	
  	print ' Moved from TN :::: Tp = %d, Tn = %d , fp =%d , fn =%d , total_FP_count_LDA =%d ' %(tp_count, tn_count, fp_count, fn_count, total_count) 



	tp_count = fp_count = tn_count = fn_count = 0
  	total_count = 0  	
  	for i,doc_pair in enumerate(model_op_lda):
  		#print ' Doc pair :: %d , %s'  %(i, doc_pair)
  		if model_op_lda[doc_pair]['label'] == 'fn':
  			total_count +=1
			if model_op_tot_lda[doc_pair]['label'] == 'tp':
				tp_count += 1 
		   	elif model_op_tot_lda[doc_pair]['label'] == 'tn':  			
				tn_count += 1
			elif model_op_tot_lda[doc_pair]['label'] == 'fp':
				fp_count += 1
			elif model_op_tot_lda[doc_pair]['label'] == 'fn':  		
  				fn_count += 1 
  	# Find tn of Model 1 moves to tp/tn/fn/fp of Model 2
  	
  	print ' Moved from TN :::: Tp = %d, Tn = %d , fp =%d , fn =%d , total_FN_count_LDA =%d ' %(tp_count, tn_count, fp_count, fn_count, total_count) 

	'''

if __name__ == '__main__':
	main()      
    
