import urllib
import difflib
import pickle
import json
import threading
import sys
import getopt

reload(sys)
sys.setdefaultencoding('utf8')

def generate_clean_question_file(diff_file, raw_file, out_clean_question_file):
    pkl_diff = open(diff_file, 'rb')
    records = pickle.load(pkl_diff)
    word_prob_groups = []
    fp = open(raw_file)
    word_probs_datas = json.load(fp, encoding='utf8')
    for word_prob_data in word_probs_datas:
        qid = word_prob_data["id"]
        #if qid.find("20130905131554AArnMlQ") != -1:
            #print "debug"
        original_text = word_prob_data["original_text"]
        text = original_text
        diff_oprs = records[qid]  
        for tag, i1, i2, j1, j2, ins_text in reversed(diff_oprs):
            if tag == 'delete':
                text = text[0:i1] + text[i2:]
            elif tag == 'insert':
                text = text[0:i1] + ins_text + text[i2:]
            elif tag == 'replace':
                text = text[0:i1] + ins_text + text[i2:]
        word_prob_data["text"] = text
        word_prob_groups.append(word_prob_data)
    fp.close()
    fp = open(out_clean_question_file, 'w')
    json.dump(word_prob_groups, fp, indent = 2, ensure_ascii=False)
    fp.close()

if __name__ == "__main__":
    original_file, diff_file, out_file = "", "", ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:d:o:')
    except getopt.GetoptError:
        print 'invalid input format'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            original_file = arg
        elif opt == "-d":
            diff_file = arg
        elif opt == "-o":
            out_file = arg
    generate_clean_question_file(diff_file, original_file, out_file)