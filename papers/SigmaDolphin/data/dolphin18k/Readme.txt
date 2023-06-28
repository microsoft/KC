************************************
Dolphin18K Documentation
************************************
1. Original data doownload from Yahoo! Answers
run OriginalDataExtractor.py to get original version of Dolphin18K
-i input_url_file
-o output_original_file
-t thread: set the number of threads (default 10)

Example: Get dev original dataset: python OriginalDataExtractor.py -i dev_urls.json -o dev_original.json -t 10

2. Get the cleaned version
run CleanVersionExtractor.py to get the cleaned version by our human annotation
-i input_original_file
-d input_diff_file
-o output_cleaned_file

Example: Get dev cleaned dataset: python CleanVersionExtractor.py -i dev_original.json -d dev_diff.pkl -o dev_cleaned.json

3. Get different subsets
run SubsetExtractor.py to get different settings (e.g. T1, T2, T6, auto, linear)
arguments:
-i input_file
-s subset_id_file
-o output_subset_file

Example: Get dev manual subset:python SubsetExtractor.py -i dev_cleaned.json -s dev_ids\\dev_manual.txt -o dev_ids\\dev_manual.json
