import urllib
import difflib
import pickle
import json
import threading
import sys
import getopt

reload(sys)
sys.setdefaultencoding('utf8')

def getHtml(url):
    page = urllib.urlopen(url)
    html = page.read()
    return html

def extract_raw_problom(str_html):
    meta_problem_title = "<meta name=\"title\" content=\""
    meta_problem_text = "<meta name=\"description\" content=\""
    problem_text_end = "\">"
    str_content = ""
    i_problem_title_start = str_html.find(meta_problem_title)
    if i_problem_title_start != -1:
        i_problem_title_end = str_html.find(problem_text_end, i_problem_title_start)
        str_problem_title = str_html[i_problem_title_start+len(meta_problem_title):i_problem_title_end]
        if str_problem_title != "":
            str_content += str_problem_title + "\n"
    i_problem_text_start = str_html.find(meta_problem_text)
    if i_problem_text_start != -1:
        i_problem_text_end = str_html.find(problem_text_end, i_problem_text_start)
        str_problem_text = str_html[i_problem_text_start+len(meta_problem_text):i_problem_text_end]
        if str_problem_text != "":
            str_content += str_problem_text
    str_content = str_content.replace("&amp;#39;", "\'").replace("&amp;#92;", "\\").replace("&amp;lt;", "<").replace("&amp;gt;", ">").replace("&amp;quot;", "\"")
    str_content = str_content.replace("&amp;", "&")
    return str_content

def is_page_not_found(str_content):   
    if str_content.find("This question does not exist or is under review.") != -1:
        return True
    return False

'''step 1: TODO multi-thread'''
def generate_raw_question_file(input_file, out_raw_question_file):
    word_prob_groups = []
    fp = open(input_file, 'r')
    word_probs_datas = json.load(fp)
    index = 0
    for word_prob_data in word_probs_datas:
        url = word_prob_data["id"].replace("yahoo.answers.", "")
        print url
        url = "https://answers.yahoo.com/question/index?qid=" + url
        html = getHtml(url)
        #fpage = open("dev_pages\\" + str(index) + ".txt")
        #html = fpage.read()
        str_content = extract_raw_problom(html)
        word_prob_data["original_text"] = str_content
        word_prob_groups.append(word_prob_data)
        index += 1
    fp.close()
    fp = open(out_raw_question_file, 'w')
    json.dump(word_prob_groups, fp, indent = 2, ensure_ascii=False)
    fp.close()
    
def assign_original_text(qids, qid2index, word_probs_datas):
    for qid in qids:
        print qid
        url = qid.replace("yahoo.answers.", "")
        url = "https://answers.yahoo.com/question/index?qid=" + url
        html = getHtml(url)
        #fpage = open("dev_pages\\" + str(qid2index[qid]) + ".txt")
        #html = fpage.read()
        str_content = extract_raw_problom(html)
        word_probs_datas[qid2index[qid]]["original_text"] = str_content
    
def generate_raw_question_file_multi_thread(input_file, out_raw_question_file, ithread = 10):
    qid2index = {}
    fp = open(input_file, 'r')
    word_probs_datas = json.load(fp)
    index = 0
    split_qids = []
    for i in range(0, ithread):
        split_qids.append([])
    for word_prob_data in word_probs_datas:
        qid = word_prob_data["id"]
        qid2index[qid] = index       
        split_qids[index % ithread].append(qid)
        index += 1
    fp.close()
    threads = []
    for i in range(0, ithread):
        #tid = threading.Thread(target=assign_original_text, args=(split_qids[i], qid2index, word_probs_datas, ) ) 
        #tid.start()
        try:
            tid = threading.Thread(target=assign_original_text, args=(split_qids[i], qid2index, word_probs_datas, ) ) 
            tid.start()
            threads.append(tid)
        except:
            print "Error: unable to start thread"
    while True:
        is_alive = False
        for tid in threads:
            if tid.isAlive():
                is_alive = True
                break
        if is_alive == False:
            break
    fp = open(out_raw_question_file, 'w')
    json.dump(word_probs_datas, fp, indent = 2, ensure_ascii=False)
    fp.close()
    
'''temp, to be deleted'''
def generate_diff_file(json_file, out_diff_file):
    id2text = {}
    fp = open("dev_dataset_full.json")
    word_probs_datas = json.load(fp, encoding='utf8')
    for word_prob_data in word_probs_datas:
        qid = word_prob_data["id"]
        text = word_prob_data["text"]
        id2text[qid] = text
    fp.close()
    fp = open(json_file)
    word_probs_datas = json.load(fp, encoding='utf8')
    records = {}
    for word_prob_data in word_probs_datas:
        qid = word_prob_data["id"]
        #text = word_prob_data["text"]
        text = id2text[qid]
        original_text = word_prob_data["original_text"]
        matcher = difflib.SequenceMatcher(None, original_text, text)
        new_record = []
        for rec in matcher.get_opcodes():
            if rec[0] == 'equal':
                continue
            if rec[0] == 'insert' or rec[0] == 'replace':
                rec = rec + (text[rec[3]:rec[4]],)
            else:
                rec = rec + ("", )
            new_record.append(rec)
        records[qid] = new_record
    fp.close()
    output = open(out_diff_file, 'wb')
    pickle.dump(records, output)
    output.close()

def validate(gold_file, merge_file):          
    fp = open(gold_file)
    word_probs_datas = json.load(fp)
    id2text, id2ori = {}, {}
    for word_prob_data in word_probs_datas:
        qid = word_prob_data["id"]
        text = word_prob_data["text"]
        if word_prob_data.has_key("original_text"):
            original_text = word_prob_data["original_text"]
            id2ori[qid] = original_text
        id2text[qid] = text
    fp.close()   
    fp1 = open(merge_file)
    word_probs_data_merge = json.load(fp1)
    for word_prob_data in word_probs_data_merge:
        qid = word_prob_data["id"]
        text = word_prob_data["text"]
        if word_prob_data.has_key("original_text"):
            original_text = word_prob_data["original_text"]
            if id2ori.has_key(qid) and original_text != id2ori[qid]:
                print qid
        if text != id2text[qid]:
            print qid
    fp1.close()
         
if __name__ == "__main__":
    '''
    fp = open("eval_dataset_full.json")
    word_probs_datas = json.load(fp)
    for word_prob_data in word_probs_datas:
        if word_prob_data.has_key("type"):
            del word_prob_data["type"]        
    fp = open("eval_urls1.json", 'w')
    json.dump(word_probs_datas, fp, indent = 2, ensure_ascii=False)
    fp.close()
    '''
    
    url_file, out_file = "", ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:t:o:')
    except getopt.GetoptError:
        print 'invalid input format'
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-i":
            url_file = arg
        elif opt == "-t":
            i_thread = int(arg)
        elif opt == "-o":
            out_file = arg
    generate_raw_question_file_multi_thread(url_file, out_file, i_thread)
    #config = "dev"
    #generate_diff_file(config + "_original.json", config + "_diff.pkl")
