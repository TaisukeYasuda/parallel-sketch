edges_file := ../data/graph_data/processed_freebase_2.graph
seeds_file := ../data/graph_data/processed_seed2_10.txt
eval_file := ../data/graph_data/processed_eval2_10.txt
gold_file := ../data/graph_data/processed_gold2_10.txt


run:
	../../build/run_mad_sketch $(edges_file) $(seeds_file) $(eval_file)	
	
eval:
	python mrr.py --n_labels 23 --output_file result.txt --gold_file $(gold_file)

