edges_file := ../data/graph_data/processed_freebase_1.graph
seeds_file := ../data/graph_data/processed_seed1_2.txt
eval_file := ../data/graph_data/processed_eval1_2.txt
gold_file := ../data/graph_data/processed_gold1_2.txt


run:
	../../build/run_mad_sketch $(edges_file) $(seeds_file) $(eval_file)	
	
eval:
	python mrr.py --n_labels 23 --output_file result.txt --gold_file $(gold_file)

