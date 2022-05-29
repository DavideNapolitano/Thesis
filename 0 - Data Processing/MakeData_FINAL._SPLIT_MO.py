import string
from nltk.util import ngrams
from tqdm import tqdm
import csv
import re
import string
import operator
from datetime import datetime
import glob
import unicodedata
import re
import contractions
from contractions import contractions_dict
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import pandas as pd
from tqdm import tqdm
from yellowbrick.text import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import spacy
import matplotlib.pyplot as plt

nltk.download('wordnet')
nltk.download('stopwords') #GUARDARE PIU AVANZATE -> NN
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')

#%% GLOBAL VARIABLES
name="HDFS"
index=28
# name="Spark"
# index=2817
Origin=True

#%% FUNCTIONS

def ImportFiles():
    print("\tIMPORT FILES")
    filepath=f"C:/Users/david/Desktop/Tesi/Dataset/{name}/{name}/*.log"
    paths=glob.glob(filepath)
    # print(len(paths))
    if len(paths)==0:
        filepath = f"C:/Users/david/Desktop/Tesi/Dataset/{name}/{name}/*/*.log"
        paths = glob.glob(filepath)
        # print(len(paths))
    # for i in paths:
    #     print(i)
    return paths

def ReadFile(paths,index):
    print("\tREAD FILE")
    path0=paths[index] # PER ORA CONSIDERO SOLO UN FILE DI LOG
    print(f"\t{path0}")
    file=open(path0,'r')
    lines=file.read().splitlines()
    file.close()
    i=0
    outread=[]
    for ln,line in enumerate(lines):
        if(len(line)>0):
            if(line.startswith((' ',"\t"))) or (line.startswith(('STARTUP_MSG',"/*","**","SHUTDOWN_MSG"))):
                None
            else:
                outread.append(line)
                i += 1
    return outread
#%%
def MakeDataframe(outread):
    print("\tMAKE PANDAS DATAFRAME")
    records=[]
    for i,line in enumerate(outread):
        tmp=[]
        counter=0
        substring=""
        firstchar=True
        errorline=False
        for char in line:
            if firstchar:
                firstchar=False
                if char.isalpha(): #FIRST CHAR AT EACH LINE IS NUMBER OF THE DATE
                    print(f"FIRST CHAR ERROR LINE: {i} - {char}")
                    errorline=True
                    break
            if char == " " and counter<4:
                tmp.append(substring)
                substring=""
                counter+=1
            else:
                substring+=char
        if counter>=4:
            tmp.append(substring)

        if errorline:
            splits = line.split(":")
            print(len(splits))
            orig = splits[0]
            if len(splits) > 1:
                ms = ":".join(splits[1:])
            else:
                ms = orig
                orig = ""
            tmp = [" ", " ", " ", orig, ms]
            print(f"ERROR LINE: {i} - {tmp}\n")

        if len(tmp)!=5:
            print("ERRORE")
            print(line)
            print(f"{tmp}\n")

        records.append(tmp)
    df=pd.DataFrame(records,columns=["DATE","TIMESTAMP","TYPE","ORIGIN","MESSAGE"])
    if Origin:
        print("\t\tWITH ORIGIN")
        df2 = df.loc[:, ["ORIGIN","MESSAGE"]]
        df2["STRING"] = df2["ORIGIN"] + " " + df2["MESSAGE"]  # PROVARE SOLO MESSAGE -> NO DIFFERENCES(TO_SAVE)
        df2 = df2.loc[:, ["STRING"]]
    else:
        print("\t\tWITHOUT ORIGIN")
        df2 = df.loc[:, ["MESSAGE"]]
    print(df.head())
    print(df2.head())
    origins = df["ORIGIN"].tolist()
    messages=df["MESSAGE"].tolist()
    # input_preproc=[]
    # for o,m in zip(origins, messages):
    #     row=o+" "+m
    #     input_preproc.append(row)
    input_preproc = df2.values.tolist()
    return input_preproc, origins, messages
#%%
def RemoveAccent(input_preproc,all=False):
    def remove_accented_chars(text):
        text = (unicodedata
                .normalize('NFKD', text)
                .encode('ascii', 'ignore')
                .decode('utf-8', 'ignore'))
        return text

    print("\tREMOVE POSSIBLE ACCENTED CHARACTERS")
    no_accent=[]
    for i,el in tqdm(enumerate(input_preproc)):
        if all:
            input=el[0]
        else:
            input=el
        #print(input)
        tmp = remove_accented_chars(input)
        #print(tmp)
        # if el!=tmp:
        #     print("MODIFIED")
        #     print(f"{i} - {el[0]}")
        #     print(f"{i} - {tmp}\n")
        #break
        no_accent.append(tmp) #LOWERCASE, CONSIDERAZIONE CAMEL CAPITALIZER

    return no_accent
#%%
def RemoveSpecial(ccs, source=True):
    
    set_upp=set()
    def remove_special_characters(text,i):  # TRA 2 /
        brackets=False
        first_len=len(text)
        text = re.sub("[\(\[].*?[\)\]]", " ", text)  # PARANTESI TONDE E QUADRE
        second_len=len(text)
        if first_len!=second_len:
            brackets=True

        build=[]
        possible_path=[]
        SOURCE=True
        for el in text.split(" "): #RIMOZIONE PATH
            if not SOURCE: #PRIMO ELEMENTO SOURCE
                if "/" in el:
                    tmp_split=el.split("/")
                    str1 = "".join([c for c in tmp_split[0] if c not in string.punctuation])
                    str2 = "".join([c for c in tmp_split[1] if c not in string.punctuation])
                    if (len(tmp_split)==2 and (len(str1)==1 or len(str2)==1)):
                        if len(str1)==1:
                            build.append(tmp_split[1])
                        else:
                            build.append(tmp_split[0])
                    else:
                        possible_path.append(el)
                else:
                    build.append(el)
            else:
                build.append(el)
            SOURCE=False
        text=" ".join(build)

        text = re.sub(' +', ' ', text)
        # print(text)
        build = []
        word_with_digit=[]
        if source:
            SOURCE=True
        else:
            SOURCE=False
        for el in text.split(" "):
            if not SOURCE:
                array = re.findall(r'[0-9]+', el)
                if len(array) == 0:
                    build.append(el)
                else:
                    word_with_digit.append(el)
            else:
                build.append(el)
            SOURCE = False
        text = " ".join(build)
        # print(text, word_with_digit)

        text = re.sub(r'[@|;|$]+', ' ', text) #SPLIT BY TOKENS
        r1 = re.compile(r"(([\w]+(:|=))+(-|)([\w]+))")#([0-9]+|[A-Za-z-]+[0-9]+))")  # npid=261268536, #pid=null

        build = []
        possible_word_eq_dp_num=[]
        for el in text.split(" "):
            if r1.match(el):
                possible_word_eq_dp_num.append(el)
            else:
                build.append(el)
        text=" ".join(build)
        # print(text,possible_word_eq_dp_num)

        r4 = re.compile(r"'[.\w]+'")
        output = []
        word_with_apex=[]
        for word in text.split():
            if r4.match(word):
                if len(word)<20:
                    word_with_apex.append(word)
                    output.append(word.replace("'", ""))
            else:
                output.append(word)
        text=" ".join(output)
        text = re.sub('[^\'a-zA-Z\s]', ' ', text)  # TENGO SOLO CARATTERI

        # print(text,word_with_apex)
        text = re.sub(r'[\r|\n|\r\n_|\r\n/]+', ' ', text) #TOLGO POTENZIALI SIMBOLI RIMASTI
        text = text.replace("[", "")
        text = text.replace("]", "")
        text = re.sub(' +', ' ', text.strip()) #TOLGO SPAZI AGGIUNTIVI
        
        res=[]
        split_upper=[]
        for el in text.split(" "):
            res.append(el)
            prevUpp = 0
            uppstring = ""
            for char in el:
                if char.isupper():
                    prevUpp += 1
                    uppstring += char
                else:
                    if prevUpp >= 3:
                        uppstring = uppstring[0:len(uppstring) - 1]
                        tmp = el
                        lowstirng = tmp.replace(uppstring, "")
                        res.remove(el)
                        res.append(uppstring)
                        res.append(lowstirng)
                        split_upper.append(uppstring)
                        split_upper.append(lowstirng)
                        set_upp.add(el)
                    prevUpp=0
                    uppstring=""
        out= " ".join(res)
        return out, possible_path, possible_word_eq_dp_num,word_with_digit, brackets,word_with_apex, split_upper

    print("\tREMOVE POSSIBLE SPECIAL CHARACTERS")
    no_special=[]
    for i,el in tqdm(enumerate(ccs)):
        tmp, path, wen, wwd, br, wwa, su = remove_special_characters(el,i)
        # if len(su)>0 and len(path)>0  and len(wwd)>0:
        #     print(f"{i} - {el}")
        #     print(f"{i} - {tmp}")
        #     print(f"{i} - PATHS: {path}")
        #     print(f"{i} - W=:N: {wen}")
        #     print(f"{i} - WWA: {wwa}")
        #     print(f"{i} - SU: {su}")
        #     print(f"{i} - WWD: {wwd}\n")

        no_special.append(tmp)
    return no_special

#%%
def CamelCase(no_accent):
    tot_words = set()
    def camel_case_split(str):
        # print(str)
        words = [[str[0]]]
        for c in str[1:]:
            if words[-1][-1].islower() and c.isupper():
                words.append(list(c))
            else:
                words[-1].append(c)
        inner_tmp = [''.join(word) for word in words]
        for ww in inner_tmp:
            tot_words.add(ww)
        out_tmp = ' '.join(inner_tmp)
        return out_tmp

    print("\tCAMEL CASE SPLIT")
    ccs=[]
    for i,el in tqdm(enumerate(no_accent)):
        # print(el)
        if len(el)>0:
            tmp = camel_case_split(el)
            tmp = tmp.lower()
        else:
            tmp=""
        # if i < 69:
        #     print(f"{i} - {el}")
        #     print(f"{i} - {tmp}")
        #     if el!=tmp:
        #         print("MODIFIED!\n")
        #     else:
        #         print("\n")
        ccs.append(tmp)
    return ccs

#%%
mergedwords=set() 
AllWord = set()
def CorrectCamel(normalized,all=False):
    if all:
        for el in (normalized):  #ALL DATA
            for wrd in el.split(" "):
                AllWord.add(wrd)
        print(len(AllWord))

    def create_merged_words(str):
        List_w = str.split(" ")
        List_w = [x for x in List_w if len(x) > 0]
        j = 0
        for i in range(len(List_w)):
            if j >= len(List_w):
                break
            if (j + 1) < len(List_w):
                tmp_w = List_w[j] + List_w[j + 1]
                if tmp_w in AllWord:
                    if "block" not in tmp_w:
                        mergedwords.add(tmp_w)
                    j += 2
                else:
                    j+=1
            else:
                j+=1

    def wrong_camel_case_split(str):
        List_w = [x for x in str.split(" ") if len(x) > 0]
        row_set = set(List_w)
        row_list = list(row_set)
        output = []
        j = 0
        flag = 0
        for i in range(len(List_w)):
            if j >= len(List_w):
                break
            if (j + 1) < len(List_w):
                tmp_w = List_w[j] + List_w[j + 1]
                if tmp_w in row_list or tmp_w in mergedwords:
                    flag = 1
                    output.append(tmp_w)
                    j += 2
                else:
                    output.append(List_w[j])
                    j += 1
            else:
                output.append(List_w[j])
                j += 1
        out = " ".join(output)
        return out, flag

    print("\tCONCATENATION WRONG CAMEL CASE")
    w_css=[]
    if all:
        for i, el in tqdm(enumerate(normalized)):
            create_merged_words(el)
        print(f"MERGED WORDS: {mergedwords}")
    for i,el in tqdm(enumerate(normalized)):
        tmp,flag = wrong_camel_case_split(el)
        # if i <69:
        #     print(f"{i} - {el}")
        #     print(f"{i} - {tmp}")
        #     if flag:
        #         print("MODIFIED!\n")
        #     else:
        #         print("\n")
        w_css.append(tmp)
    return w_css

def ExpandContraction(w_css):
    def expand_contractions(text):
        expanded_words = []
        for word in text.split():
            fix_w = contractions.fix(word)
            expanded_words.append(fix_w)
        out_tmp = ' '.join(expanded_words)
        return out_tmp

    print("\tEXPAND POSSIBLE CONTRACTIONS")
    expand=[]
    for i,el in tqdm(enumerate(w_css)):
        tmp = expand_contractions(el)

        if el!=tmp:
            print("MODIFIED!\n")
            print(f"{i} - {el}")
            print(f"{i} - {tmp}\n")
        expand.append(tmp)
    return expand

def Tokenization(expand):
    print("\tTOKENIZATION")
    tokenize=[]
    for i,el in tqdm(enumerate(expand)):
        tmp = word_tokenize(el)
        tokenize.append(tmp)
    return tokenize

def RemoveStopwords(tokenize):
    print("\tREMOVE POSSIBLE STOPWORDS")
    stop = stopwords.words('english')
    stop.append(name.lower()) #DATASET NAME
    stop.remove("not")
    stop.remove("no")
    stop.remove("nor")
    stop.remove("down")
    no_stop=[]
    for i,el in tqdm(enumerate(tokenize)):
        tmp=[]
        for x in el:
            if x not in stop:
                if x=="no" or x=="nor":
                    tmp.append("not")
                else:
                    tmp.append(x)

        # if i < 69:
        #     print(f"{i} - {el}")
        #     print(f"{i} - {tmp}")
        #     if len(el)!=len(tmp):
        #         print(f"MODIFIED! REMOVED: {[y for y in el if y not in tmp]}\n")
        #     else:
        #         print("\n")

        no_stop.append(tmp)
    return no_stop

def PosTagChunk(no_stop):
    print("\tPOS TAGGING & CHUNKING")
    tagg=[]
    for i,el in tqdm(enumerate(no_stop)):
        tmp = pos_tag(el)
        # if i < 100:
        #     print(f"{i} - {el}")
        #     print(f"{i} - {tmp}\n")
        tagg.append(tmp)
    return tagg

#%%
def StemmLemm(tagg,word_tags): 
    print("\tSTEMMING & LEMMING")
    admitted_tag=["JJ","JJR","JJS","NN","NNS","NNP","NNPS","VB","VBD","VBG","VBN","VBP","VBZ"]
    stemm=[]
    wl = WordNetLemmatizer()
    diz_word={}
    wrong_word_set=set()
    word_ing=set()
    wrong_word_idx=[]
    to_keep_wrong_tag=["not","filter","namenode","second","timeout","util","down","server","driver"]
    def get_wordnet_pos(tokens,idx=-1):
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN,
                    "V": wordnet.VERB}#, "R": wordnet.ADV} #EXTEND???
        tmp=[]
        for token in tokens:
            # print(token)

            word=token[0]
            tag=token[1]
            # if word=="remoting":
            #     print(word,tag)
            if word.endswith("ing") and tag[0]!="V":
                word_ing.add((word,tag))
                tag = "VB"
            if tag[0]!="V" and word.endswith("ed"):
                wrong_word_set.add(word)
                if idx<0:
                    wrong_word_idx.append(idx)
                tag="VB"
            if tag in admitted_tag or word in to_keep_wrong_tag:
                #diz_word[word]=tag
                tmp.append((word, tag_dict.get(tag[0], wordnet.NOUN)))
        # print(tmp)
        return tmp

    to_keep = ["max", "new", "net", "msg", "bad", "os", "ipc", "not", "eof", "dfs", 'io', "up", "map", "use", "bp",
               "nodename", "nn", "no", "src", "dst", "end", "rdd", "server"]
    avoid_lemm_error = ["dest", "os", "not", 'io', "server"]
    my_lemma = {"rescanning": "rescan", "rescanned": "rescan", "dst": "dest","remoting":"remote","deprecation":"deprecate"}
    word_3 = set()
    if len(word_tags)==0:
        print("FILL WORD TAGS")
        # java, io, eof, reset, socket, file, org, datanode, user, ipc, impl, storage, bpid, block, bytes, nn, src, dest, write, id, finalize, interval, total, fsdataset, replicas
        # TODO USER INPUT
        for i, el in (enumerate(tagg)):
            for w,t in el:
                if w not in diz_word.keys():
                    diz_word[w]=set()
                    t_wn=get_wordnet_pos([(w,t)])
                    if len(t_wn)>0:
                        diz_word[w].add(t_wn[0][1])
                    if len(diz_word[w])==0:
                        print(f"{w} NOT added since {t} NOT admitted")
                        var=""
                        while var not in ["y","n"]:
                            var = input(f"Do you want to change the TAG for {t} (prev {t}): ")
                            print("You entered: " + var)

                            if var in ["y","n"]:
                                if var=="y":
                                    var2=''
                                    while var2 not in ['a','b','n']:
                                        var2 = input(f"Choose tag between 'a','b' and 'n':")
                                        if var2 in ['a','b','n']:
                                            t_wn = get_wordnet_pos([(w, var2)])
                                            if len(t_wn) > 0:
                                                diz_word[w].add(t_wn[0][1])
                                else:
                                    print(f"Element Discarded")
                            else:
                                print(f'Input not acceptable, choose between: {["y","n"]}')
                        print("\n")
                else:
                    t_wn = get_wordnet_pos([(w, t)])
                    if len(t_wn) > 0:
                        diz_word[w].add(t_wn[0][1])
        word_tags= {}
        tot_err=0
        for k,v in diz_word.items():
            v=list(v)
            if len(v)>1:
                tot_err+=1
        print("\n")
        print(f"There are {tot_err} words with more than one tag")
        print("\n")
        tag_wn={"a":"ADJECTIVE","n":"NOUN","v":"VERB","d":"DISCARDA TAG AGGREGATION ANALYSIS FOR THAT WORD"}
        to_pop=[]
        for k,v in diz_word.items():
            if len(v)==0:
                to_pop.append(k)
            else:
                v=list(v)
                if len(v)>1:
                    kb=""
                    for char in k:
                        kb=kb+"\u0332"+char+"\u0332"
                    print(f"Conflict for: {kb}, TAGS found: {v}")
                    set_row=set()
                    for orig in no_special:
                        ispresent=False
                        for wrd in orig.split(" "):
                            if wrd.lower()==k:
                                ispresent=True
                        if ispresent:
                            tmp = []
                            for wrd in orig.split(" "):
                                if wrd.lower()==k:
                                    tmpwrd=""
                                    for char in wrd:
                                        tmpwrd = tmpwrd + "\u0332" + char + "\u0332"
                                    tmp.append(tmpwrd)
                                else:
                                    tmp.append(wrd)
                            toadd=" ".join(tmp)
                            set_row.add(toadd)
                    var=None
                    print(f"Here the first 5 row to decide the TAG")
                    it_tmp=0
                    for el in set_row:
                        if it_tmp<5:
                            print(f"{it_tmp} - {el}")
                        else:
                            break
                        it_tmp+=1
                    # v.append("d")
                    while var not in v:
                        guideline={}
                        for el in v:
                            if el=="a":
                                guideline[el]="ADJECTIVE"
                            if el=="n":
                                guideline[el]="NOUN"
                            if el=="v":
                                guideline[el]="VERB"

                        guideline["d"]="DISCARDA TAG AGGREGATION ANALYSIS FOR THAT WORD"
                        var = input(f"Please choose the desidered TAG between {v} {guideline}: ")
                        print("You entered: " + var)
                        if var not in v:
                            if var in tag_wn.keys():
                                var2 = input(f"You have inserted another acceptable TAG: {var}. Do you confirm it? [y/n]")
                                if var2=="y":
                                    word_tags[k] = var
                                    print(f"Modification Accepted!\n")
                                    break
                                if var2=="n":
                                    print(f"Modification Discarded!")
                            else:
                                print(f"Input not acceptable, choose between: {v}")
                        else:
                            word_tags[k]=var
                            print(f"Input Valid!\n")
                else:
                    word_tags[k] = v
        for el in to_pop:
            diz_word.pop(el,None)
    print("\n")

    print("Start Lemmization")
    print("\n")
    for i,el in tqdm(enumerate(tagg)):
        tmp = []
        res=get_wordnet_pos(el,i)

        for w in res:
            if len(w[0])<4 and w[0] not in to_keep:
                word_3.add(w[0])

        for w1,w2 in zip(el,res):
            if w2[0] in avoid_lemm_error:
                tmp.append(w2[0])
            else:
                if(w2[1] == "v" and len(w2[0]) > 2) or (w2[0] in to_keep) or (len(w2[0]) > 3):
                    if w2[0] in list(my_lemma.keys()):
                        tmp.append(my_lemma[w2[0]])
                    else:
                        if word_tags[w2[0]]!="d":
                            tmp.append(wl.lemmatize(w2[0], word_tags[w2[0]][0]))
                        else:
                            tmp.append(wl.lemmatize(w2[0], w2[1]))
        print(f"{i} - {el}")
        print(f"{i} - {tmp}")
        if el!=tmp:
            print("MODIFIED\n")
        else:
            print("\n")
        stemm.append(tmp)
    print(word_3)
    print(wrong_word_set)
    print(wrong_word_idx)
    print(word_ing)
    return stemm, diz_word,word_tags


#%%
def Regex(shorty):
    print("\tREGEX")
    r1r = re.compile(r"([bcdfghjklmnpqrstvwxz]*[aeiou]{3}[bcdfghjklmnpqrstvwxz]*)+")
    r2r = re.compile(r"([aeiou]*[bcdfghjklmnpqrstvwxz]{4}[aeiou]*)+")
    strange_w = set()
    r4 = set()
    r5 = set()
    to_keep=["queue", "http"]
    def cleaning(text):
        output = []
        for wrd in text:
            if r1r.match(wrd) and len(wrd) < 6 and wrd not in to_keep:
                strange_w.add(wrd)
                r5.add(wrd)
            else:
                if r2r.match(wrd) and len(wrd) < 6 and wrd not in to_keep:
                    # print(wrd)
                    strange_w.add(wrd)
                    r4.add(wrd)
                else:
                    output.append(wrd)
        return output

    final = []
    for i, el in tqdm(enumerate(shorty)):  # ngrs
        tmp = cleaning(el)
        if el != tmp:
            print(f"{i} - {el}")
            print(f"{i} - {tmp}")
            diff=[w for w in el if w not in tmp]
            print(f"{i} - {diff}")
            print("MODIFIED!\n")

        final.append(tmp)
    print(f"\tStrange Words: {strange_w}")
    print(f"{len(r4),len(r5)}")
    return final

#%%
def RemoveFinalS(input,input_tot):
    print("REMOVE FINAL S")
    TotWord = []
    for el in tqdm(enumerate(input_tot)):  # ALL DATA
        for wrd in el[1]:
            TotWord.append(wrd)

    tot_word = list(set(TotWord))
    tot_word_ext=tot_word
    print(len(tot_word_ext), len(TotWord))

    tot_w=tot_word_ext
    new_row_set=[]
    new_final = []
    count_modiication=0
    for row in input:
        new_row = []
        for word in row:
            word_trunc = ""
            last_char = ""
            for i, char in enumerate(word):
                if i < len(word) - 1:
                    word_trunc += char
                else:
                    last_char = char

            if word_trunc in tot_w and last_char == "s":
                new_row.append(word_trunc)
            else:
                new_row.append(word)

        if row != new_row:
            count_modiication+=1
        if row != new_row and new_row not in new_row_set:
            print("MODIFIED!")
            print(row)
            print(new_row)
            new_row_set.append(new_row)
        new_final.append(new_row)
    print(len(input), (len(new_final)), count_modiication)
    return new_final
#%%
def NGRAMS(final,no_mono,mono_only=False):
    print("\tMAKE NGRAMS")
    if mono_only:
        res=[]
        print(no_mono)
        for row in final:
            tmp=[]
            for el in row:
                if el not in no_mono:
                    print(el)
                    tmp.append(el)
            res.append(tmp)
        return res
    print("ALL")
    ngrs = []
    total_bigrams=set()  #NON NECESSARIO FARLO A PRIORI, OGNI RIGA SI BASA PER LE CORREZIONE SUI BIGRAMMI DELLE RIGHE PRECEDENTI
    total_trigrams=set()
    for i, el in enumerate(final):
        print(el)
        tmp1_t = list(map(" ".join, list(ngrams(el, 1))))
        tmp1=[el for el in tmp1_t if el not in no_mono]
        tmp2 = list(map(" ".join, list(ngrams(el, 2))))
        tmp3 = list(map(" ".join, list(ngrams(el, 3))))
        print(tmp2)
        print(tmp3)
        tmp2_3=[]
        for idx,ngrams_list in enumerate([tmp2,tmp3]): #PRIMA BI E POI TRIGRAMMI
            inner_tmp=[]
            for concat_words in ngrams_list:
                inner_words=concat_words.split(" ")
                single_words=set(inner_words)
                if len(single_words)==len(inner_words): #NON SONO PRESENTI ALMENO 2 PAROLE UGUALI -> SOLO PAROLE DISTINTE
                    if idx == 0: #SE BIGRAM
                        #word1 = inner_words[0] + " " + inner_words[1]
                        word_reversed = inner_words[1] + " " + inner_words[0]
                        if word_reversed in total_bigrams:
                            print(f"{i} - REVERSE ALREADY EXISTS: {word_reversed} of {concat_words}")
                            inner_tmp.append(word_reversed)
                            total_bigrams.add(word_reversed)
                        else:
                            inner_tmp.append(concat_words)
                            total_bigrams.add(concat_words)
                    else:
                        to_add=concat_words
                        inner_tmp.append(to_add)
                        total_trigrams.add(to_add)
                else:
                    print(f"{i} - NOT ADDED: {concat_words}")

            tmp2_3+=inner_tmp

        tmp4 = tmp1 + tmp2_3#tmp2 + tmp3 + tmp1
        tmp = tmp4

        print(f"{i} - {el}")
        print(f"{i} - {tmp}\n")
        ngrs.append(tmp)

    return ngrs

#%% Pipelinefrom datetime import datetime
start_time = datetime.now()

paths=ImportFiles()

outread=ReadFile(paths,index)

input_preproc, orig, messag=MakeDataframe(outread)
#%%
# for i in range(len(input_preproc)):
#     if i<30:
#         print(orig[i], messag[i])
#         print(input_preproc[i])
#         print("\n")
#%%
no_accent=RemoveAccent(input_preproc,all=True)
no_accent_o=RemoveAccent(orig)
no_accent_m=RemoveAccent(messag)
#%%
no_special=RemoveSpecial(no_accent)
no_special_o=RemoveSpecial(no_accent_o)
no_special_m=RemoveSpecial(no_accent_m, source=False)
#%%
np.random.seed(0)
tmp=np.array(no_special)
tmpu,tmpind=np.unique(tmp, return_index=True)
print(f"TOT BEFORE: {len(tmp)}, AFTER: {len(tmpu)}")
#%%
#np.random.seed(0)
tmp=np.array(no_special_o)
tmpuo,tmpind=np.unique(tmp, return_index=True)
print(f"ORIGIN BEFORE: {len(tmp)}, AFTER: {len(tmpu)}")
#%%
#np.random.seed(0)
tmp=np.array(no_special_m)
tmpum,tmpindm=np.unique(tmp, return_index=True)
print(f"MESSAGE BEFORE: {len(tmp)}, AFTER: {len(tmpum)}")
#%%
tmpind=np.sort(tmpind)
tmpindm=np.sort(tmpindm)
print(len(tmpind),len(tmpindm))
#%%
idx=0
for i in range(len(tmpind)):
    el1=tmpind[i]
    el2=tmpindm[idx]
    if el1==el2:
        print(i,el1,el2)
        print(no_special[el1])
        print(no_special_m[el2])
    else:
        print(i,el1)
        print(no_special[el1])
        idx-=1
    # if el1!=el2:
    #     print(no_accent[el2])
    #     print(no_accent_m[el2])
    # print(el2)
    idx+=1
    print("\n")
#%%
c=0
for i,el in enumerate(no_special_o):
    if len(el)==0:
        print(no_accent[i])
        c+=1
print(c)
print(len(no_special_o))
#%%
ccs=CamelCase(no_special)
ccs_o=CamelCase(no_special_o)
ccs_m=CamelCase(no_special_m)
#%%
w_css=CorrectCamel(ccs,all=True)
w_css_o=CorrectCamel(ccs_o)
w_css_m=CorrectCamel(ccs_m)
#%%
expand=ExpandContraction(w_css)
expand_o=ExpandContraction(w_css_o)
expand_m=ExpandContraction(w_css_m)
#%%
tmp=np.array(expand)
tmpu,tmpind=np.unique(tmp, return_index=True)
print(f"BEFORE: {len(tmp)}, AFTER: {len(tmpu)}")

#%%
tokenize=Tokenization(expand)
tokenize_o=Tokenization(expand_o)
tokenize_m=Tokenization(expand_m)

#%%
no_stop=RemoveStopwords(tokenize)
no_stop_o=RemoveStopwords(tokenize_o)
no_stop_m=RemoveStopwords(tokenize_m)
#%%
tmp=np.array(no_stop)
tmpu,tmpind=np.unique(tmp, return_index=True)
no_stop=tmpu.tolist()
print(f"BEFORE: {len(tmp)}, AFTER: {len(no_stop)}, DIFF: {len(tmp)-len(no_stop)}, %: {100*((len(tmp)-len(no_stop))/len(tmp))}%")
no_stop_o=[no_stop_o[idx] for idx in tmpind]
no_stop_m=[no_stop_m[idx] for idx in tmpind]
#%%
for i in range(len(no_stop)):
    print(i, input_preproc[tmpind[i]])
    print(i,no_special_m[tmpind[i]])
    print("\n")
#%%
for i in range(len(no_stop)):
    # if i<30:
    #     print(no_stop_o[i], no_stop_m[i])
    #     print(no_stop[i])
    #     print("\n")
    tmp=no_stop_o[i]+no_stop_m[i]
    if tmp!= no_stop[i]:
        print("ERROR")
        print(i,input_preproc[tmpind[i]])
        print(i,no_stop_o[i], no_stop_m[i])
        print(i,no_stop[i])
        print("\n")
#%%
tagg=PosTagChunk(no_stop)
tagg_o=PosTagChunk(no_stop_o)
tagg_m=PosTagChunk(no_stop_m)
#%%
tags={}
stemm2,_,tags=StemmLemm(tagg,tags)
#%%
stemm2_o, _,_=StemmLemm(tagg_o,tags)
#%%%
stemm2_m, _,_=StemmLemm(tagg_m,tags)
#%%
print(tagg_m[10])
print(stemm2_m[10])
#%%
final2=Regex(stemm2)
final2_o=Regex(stemm2_o)
final2_m=Regex(stemm2_m)
#%%
new_final2=RemoveFinalS(final2,final2)
new_final2_o=RemoveFinalS(final2_o,final2)
new_final2_m=RemoveFinalS(final2_m,final2)

#%%
# nf2=[" ".join(el) for el in new_final2]
# nf=CorrectCamel(nf2)
# new_final2=[el.split(" ") for el in nf]

nf2=[" ".join(el) for el in new_final2]
nf2o=[" ".join(el) for el in new_final2_o]
nf2m=[" ".join(el) for el in new_final2_m]
#%%
nf=CorrectCamel(nf2,all=True)
nfo=CorrectCamel(nf2o)
nfm=CorrectCamel(nf2m)
#%%
new_final2=[el.split(" ") for el in nf]
new_final2_o=[el.split(" ") for el in nfo]
new_final2_m=[el.split(" ") for el in nfm]

#%%
for i in range(len(new_final2)):
     print(i,new_final2_o[i], new_final2_m[i])
     print(i,new_final2[i])
     print("\n")
     tmp=new_final2_o[i]+new_final2_m[i]
     if tmp!= new_final2[i]:
        print("ERROR")
        print(i,input_preproc[tmpind[i]])
        print(i,new_final2_o[i], new_final2_m[i])
        print(i,new_final2[i])
        print("\n")
#%%
new_final=new_final2

#%% MAX 1 DISTINC WORD IN EACH ROW
new_final=np.array(new_final)
new_final_unique_max_f1=[]
for row in new_final:
    r=np.array(row)
    ru=np.unique(r)
    new_final_unique_max_f1.append(ru)
new_final_unique_max_f1=np.array(new_final_unique_max_f1)
#%%
TotWordfinal_unique_max_f1=[]
for el in tqdm(enumerate(new_final_unique_max_f1)): #ALL DATA
    for wrd in el[1]:
        TotWordfinal_unique_max_f1.append(wrd)
TotWordfinal_unique_max_f1=np.array(TotWordfinal_unique_max_f1)
TotWord_set=set(TotWordfinal_unique_max_f1)
#%%
diz_word_doc2={}
for word in TotWord_set:
    diz_word_doc2[word]=0
for word in TotWord_set:
    for row in new_final_unique_max_f1:
        if word in row:
            diz_word_doc2[word]+=1
sorted_d = dict( sorted(diz_word_doc2.items(), key=operator.itemgetter(1),reverse=True))
keys = list(sorted_d.keys())
values = list(sorted_d.values())
#%%
iter=0
for k,v in zip(keys,values):
    print(iter,k,v)
    iter+=1
#%%
corpus=[]
for row in new_final_unique_max_f1:
    r=" ".join(row)
    print(r)
    corpus.append(r)
print(len(corpus))
#%% COUNT - SINGLE WORDS

vectorizer = CountVectorizer(max_df=0.8)# min_df=2)#, ngram_range=(1, 3))
X = vectorizer.fit_transform(corpus)
print(X.shape)
features1=vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features1, orient='h', n=30,  size=(1080,720), title="CountVectorizer - Top-30", color="cornflowerblue")
visualizer.fit(X)
visualizer.show()

#%%
diz_word_doc={}
for word in features1:
    diz_word_doc[word]=0
for word in features1:
    for row in new_final_unique_max_f1:
        if word in row:
            diz_word_doc[word]+=1
sorted_d = dict( sorted(diz_word_doc.items(), key=operator.itemgetter(1),reverse=True))
keys = list(sorted_d.keys())
values = list(sorted_d.values())
tot_len=len(new_final_unique_max_f1)
no_mono=[]
for k,v in zip(keys,values):
    f=v/tot_len
    if f>=0.2 and f<0.8:
        no_mono.append(k)
    print(k,v,f)
print(f"\nWORDS TO TAKE ONLY THEIR BIGRAM/TRIGRAM INSIDE LOG ROWS: {no_mono}")

#%% CLEAN DATA
def CleanData(data):
    in_data=[]
    dropped_words=set()
    for row in tqdm(data):
        tmp=[]
        for el in row:
            if el in features1:
                tmp.append(el)
            else:
                dropped_words.add(el)
        in_data.append(tmp)
    print(f"DROPPED WORDS THAT ARE TO FREQUENT {dropped_words}")
    return in_data

in_data=CleanData(new_final)
in_data_o=CleanData(new_final2_o)
in_data_m=CleanData(new_final2_m)
#%%
#input_data=NGRAMS(in_data, no_mono)
input_data_o=NGRAMS(in_data_o, no_mono, mono_only=True)
#%%
input_data_m=NGRAMS(in_data_m, no_mono)
#%%
input_data=[]
for el1,el2 in zip(input_data_o,input_data_m):
    tmp=el1+el2
    if len(el1)==1 and len(el2)==1:
        print(tmp)
        tmp.append(f"{el1[0]} {el2[0]}")
    input_data.append(tmp)
#%%
print(no_mono)
for i in range(len(input_data)):
     print(i,in_data_o[i], input_data_o[i])
     print(i, in_data_m[i], input_data_m[i])
     print(i,input_preproc[tmpind[i]])
     print(i,input_data[i])
     print("\n")

#%%
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
#%%
f = open(f"Data Folder/{name}/SPLIT/{name}_Input_Output_SPLIT.txt", "w")
f2=open(f"Data Folder/{name}/SPLIT/{name}_Raw_SPLIT.txt", "w")
iter=0

for ind in tmpind:
    print(f"{iter} - {no_accent[ind]}", file=f)
    print(f"{no_accent[ind]}", file=f2)
    print(f"{iter} - {input_data[iter]}\n", file=f)  # SpaCy
    iter+=1
f.close()
f2.close()
#%%
print("SAVE DATA")

fname = f"Data Folder/{name}/SPLIT/{name}_InputData_Origin+Message_SPLIT.txt"

f = open(fname, "w")
for el in input_data:
    print(el,file=f)
print("END SAVE DATA")
f.close()

