import json
import sys
import getopt

reload(sys)
sys.setdefaultencoding('utf8')

def generate_subset(json_file, id_file, out_file):
    ids = []
    fp = open(id_file)
    while True:
        line = fp.readline().replace("\n", "")
        if line == "":
            break; 
        ids.append(line)
    fp.close()
    fp = open(json_file)
    word_probs_datas = json.load(fp)
    sub_word_prob = []
    for word_prob_data in word_probs_datas:
        qid = word_prob_data["id"]
        if qid in ids:
            sub_word_prob.append(word_prob_data)
    fp.close()
    fp = open(out_file, 'w')
    json.dump(sub_word_prob, fp, indent = 2, ensure_ascii=False)
    fp.close()
    
if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:s:o:')
    except getopt.GetoptError:
        print 'invalid input format'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            json_file = arg
        elif opt == "-s":
            id_file = arg
        elif opt == "-o":
            out_file = arg
    generate_subset(json_file, id_file, out_file)