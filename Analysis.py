import matplotlib.pyplot as plt
import csv
from matplotlib import font_manager
import jieba
import time
from itertools import chain
#---------------print header------------------

trainname="C:\\Users\\Tony's PC\\Desktop\\NLP\\Project Of NLP\\Python_NLP\\\data\\TapTap\\TapTap_data.csv"
with open(trainname,encoding='utf-8-sig') as test:
    reader=csv.reader(test)
    header_row=next(reader)

    comment=[]
    stars=[]
    comment1=[]
    comment2=[]
    comment3=[]
    comment4=[]
    comment5=[]
    commentnumber=[]
    longcomment=[]
    for row in reader:
        if row[4]=='1':
            comment1.append(row[5])
        if row[4]=='2':
            comment2.append(row[5])
        if row[4]=='3':
            comment3.append(row[5])
        if row[4]=='4':
            comment4.append(row[5])    
        if row[4]=='5':
            comment5.append(row[5])
        comment.append(row[5])
        stars.append(row[4])
        if len(row[5])>200:
            longcomment.append(row[5])
        else:
            commentnumber.append(len(row[5]))
#histgram of comments
# d=5
# num_bin=range(min(commentnumber),int(max(commentnumber)),d)
# plt.hist(commentnumber,num_bin)
# plt.xticks(num_bin)
# plt.xlabel('word number')
# plt.ylabel('frecuncy')
# plt.title("histolgram of comment words")
# plt.grid(alpha=0.4)
# plt.show()

# print(len(longcomment))
#-----------------seperation-------------------------
import jieba.posseg as pseg

seperation={}
comment_sep=[]
#IF you want to know different comments wordcloud change iteration here
for i in comment1:    
    for j in pseg.lcut(i):
        if j.flag=='a' :
            comment_sep.append(j.word)
        if j.word in seperation:
            seperation[j.word]=seperation[j.word]+1
        else:
            seperation[j.word]=1


#-------------------------word cloud-------------------------------------
from wordcloud import WordCloud

def get_word_cloud(keywords_list):
   wordcloud = WordCloud(font_path="C:\\Users\\Tony's PC\\Desktop\\NLP\\Project Of NLP\\Python_NLP\\data\\Hoteldata\\SimHei.ttf", max_words=100, background_color="white")
   keywords_string=" ".join(keywords_list)
   wordcloud.generate(keywords_string)
   plt.figure()
   plt.imshow(wordcloud,interpolation="bilinear")
   plt.axis("off")
   plt.show()

get_word_cloud(comment_sep)