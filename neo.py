import re
import pandas as pd
import bs4
import requests
import spacy
import os
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import Matcher
from spacy.tokens import Span
import networkx as nx
#import numpy
from neo4j.v1 import GraphDatabase,basic_auth
from tqdm import tqdm
from py2neo.data import Node,Relationship,Walkable
from py2neo import Graph,NodeMatcher,RelationshipMatcher
graph=Graph("bolt://localhost:7687",user = "neo4j", password = "123456")
spacy_stopwords = STOP_WORDS
#print(spacy_stopwords)

def remove_stopwords(tokens):
    cleaned_tokens = []
    
    for token in tokens:
        if token not in spacy_stopwords:
            cleaned_tokens.append(token)
    
    return cleaned_tokens

#graph.delete_all()


import en_core_web_sm
nlp = spacy.load('en_core_web_sm')

pd.set_option('display.max_colwidth',200)
df = pd.read_csv('sample.csv')
df.shape


def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""
  ent=[]

  

  #############################################################
  doc = nlp(sent)
  
  for tok in nlp(sent): 
   
    print(tok.text)
    print(tok.dep_)
    
    ## chunk 2
    # if token is a punctuation mark then move on to the next token
    if tok.dep_ != "punct":
      # check: token is a compound word or not
      if tok.dep_ == "compound":
        prefix = tok.text
        ent.append(tok.text.lower())
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          prefix = prv_tok_text + " "+ tok.text
          #print(prv_tok_text)
          #print(tok.text)
          #print(prefix)
      
      # check: token is a modifier or not
      if tok.dep_.endswith("mod") == True:
        modifier = tok.text
        # if the previous word was also a 'compound' then add the current word to it
        if prv_tok_dep == "compound":
          modifier = prv_tok_text + " "+ tok.text
      
      ## chunk 3
      if tok.dep_.find("subj") == True:
        ent1 = modifier +" "+ prefix + " "+ tok.text
        ent.append(tok.text.lower())
        #print(ent1)
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        ent.append(tok.text.lower())
      ## chunk 5  
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text
  #############################################################

  return ent

def get_relation(sent):

  doc = nlp(sent)

  # Matcher class object 
  matcher = Matcher(nlp.vocab)

  #define the pattern 
  pattern1 = [{'DEP':'ROOT'}, 
            {'DEP':'prep','OP':"?"},
            {'DEP':'agent','OP':"?"},  
            {'POS':'ADJ','OP':"?"}] 
  pattern2 = [{'DEP':'DET','OP':'?'},{'POS':'NOUN'},{'POS':'PROPN','OP':'?'}]

  matcher.add("matching_1", None, pattern1,pattern2) 

  matches = matcher(doc)
  #print(matches)
  #k = len(matches) - 1

  #span = doc[matches[k][1]:matches[k][2]] 
  span=[]
  for i in range(len(matches)):
    #print("relation")
    span.append((doc[matches[i][1]:matches[i][2]]).text.lower())

  return(span)

def get_named_entities(sent):
  
  doc = nlp(sent)
    #for tok in doc:
    #print(tok.text)
    #print(tok.dep_)
  ent=[]
  for i in doc.ents:
    ent.append(i.text.lower())
    #print(i.label_)
  return ent

#question_words=['When','Who','Where','Why','What','Which','How','','when','who','where','why','what','which','how']
#common_words=['is','the','get','you','You','me','Me','Is','we','We','Get','put','Put','The','a','A','An','an','Are','are','Were','were','was','Was','have','Have','has','Has','as','As','be','Be']
entity_pairs = []
relations = []
answers = []
questions = []
n=0

for i in tqdm(df[ "Question"]):
  rel=get_relation(i)
  print(rel)
  e=(get_named_entities(i))
  #if e == []:
  entities=get_entities(i)
  #entities.append(e)
  #else:
  #entities=e
  
  for i in e:
    entities.append(i.lower())
    #print(entities)
  print(entities)
  
  #print(final_entities)
  final_relation=remove_stopwords(rel)
  f_entities=remove_stopwords(entities)
  print(f_entities)
  final_entities=[i for i in f_entities if i not in final_relation]
  print(final_entities)
  print(final_relation)

  for i in final_relation:
    for ent in final_entities:
     entity_pairs.append(ent.lower())
     questions.append(df['Question'][n])
     answers.append(df['Answer'][n])
     relations.append(i.lower())
  n+=1 
  #print(entity_pairs) 

  # for i in tqdm(df["Answer"]):
  #   ans_rel=get_relation(i)
  #   e = get_named_entities(i)
  #   entities = get_entities(i)
  #   for i in e:
  #     entities.append(i.lower())

  #   final_relation=remove_stopwords(ans_rel)
  #   f_entities=remove_stopwords(entities)
  #   final_entities=[i for i in f_entities if i not in final_relation]  
    
  #   for i in final_relation:
  #     for ent in final_entities:
  #       ans_entity_pairs.append(ent.lower())
  #       ans_relations.append(i.lower()) 


#source = [i[0] for i in entity_pairs]
#target = [i for i in entity_pairs]
#print(target)
#for i in entity_pairs:
  #print(i)
  #for j in i:
    #print(j)
  
    #numpy_data = np.array(entity_pairs)
kg_df = pd.DataFrame({'ques':questions ,'target':entity_pairs, 'edge':relations,'answer':answers})



matcher = NodeMatcher(graph)
rel_match=RelationshipMatcher(graph)
print(kg_df)
for i in range(len(kg_df)):
    tx = graph.begin()
    
    '''qs = matcher.match("source",name=source[i].lower()).first()
    qt = matcher.match("target",name=source[i].lower()).first()
    
    
    if qs is not None:
        a = qs
    elif qt is not None:
        a=qt
    else:
        a=Node("source",name=source[i].lower())
        tx.create(a)

    '''
    
    rt=matcher.match("target",name=entity_pairs[i]).first()
    #rs=matcher.match("source",name=target[i].lower()).first()
    
    if rt is not None:
        b=rt
    else:
        b=Node("target",name=entity_pairs[i])
        tx.create(b)
    
    #final_relation=target[i]+' '+relations[i]
    
    
    '''q = matcher.match("question",name=df['Question'][i].lower()).first()
    if q is None:
        ques = Node("question",name=df['Question'][i].lower())
        tx.create(ques)
    else:
        ques=q
    '''
    count = 0
    #z = matcher.match("answer",name=answers[i],rank_id).first()
    doc = nlp(answers[i])
    for tok in doc:
      #print(tok.text)
      if tok.text.lower()==entity_pairs[i].lower():
        count+=1

    z = matcher.match("answer",name=answers[i]).first()
    if z is None:
        ans = Node("answer",name=answers[i])
        tx.create(ans) 
    else:
        ans=z  

    q = matcher.match("ques",name=questions[i]).first()
    if q is None:
      ques = Node("ques",name=questions[i])
      tx.create(ques)
    else:
      ques = q

    r = graph.relationships.match((b,ans),relations[i]).first()
    if r is None:
        if b == ans:
            continue
        else:
            ab=Relationship(b,relations[i],ans,freq=count)
            tx.create(ab)
    else:
        ab = r


    w = graph.relationships.match((ques,ans),'Answer is').first()
    if w is None:
      xy = Relationship(ques,'Answer is',ans)
      tx.create(xy)
    else:
      xy=w
    # for rel in graph.relationships.match((b,None),None):
    #   print(rel.end_node['name'])


    

    '''ra = graph.relationships.match((ques,ans),'Answer is').first()
    
    if ra is None:
        if ques==ans:
            continue
        else:
            qa = Relationship(ques,'Answer is',ans)
            tx.create(qa)
    else:
        qa = ra
    '''
    ##tx.create(w)
    
    tx.commit()

# ep_without_dup = list(set(entity_pairs))
# print(ep_without_dup)

# for i in ep_without_dup:
#   #j=1
#   #print(i)
#   #tx = graph.begin()
#   n = matcher.match('target',name=i).first()
#   if n is not None:
#     node_match = graph.relationships.match((n,None),None).order_by('_.freq')
#     num = graph.relationships.match((n,None),None).__len__()

#     print(i)
#     #print(reverse(node_match))
#     rank=0
#     for j in node_match:
#       if num >0:
#         print(j['freq'])
#         rank+=1
#         j['rank_id']=rank
#         graph.push(j)
#         num-=1
      



        


'''for i in range(len(kg_df)):
    for rel in graph.match(start_node=source[i]):
        print(rel.rel_type.properties[relations[i]],rel.end_node.properties[target[i]])
    a=Node("source",name=source[i])
    b=Node("target",name=target[i])
    ab=Relationship(a,relations[i],b)
    graph.create(ab)
    print(ab)
    
G=nx.from_pandas_edgelist(kg_df[kg_df['edge']=="is"], "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())


plt.figure(figsize=(12,12))
pos = nx.spring_layout(G, k = 0.5) # k regulates the distance between nodes
nx.draw(G, with_labels=True, node_color='skyblue', node_size=1500, edge_cmap=plt.cm.Blues, pos = pos)
plt.show()
'''
