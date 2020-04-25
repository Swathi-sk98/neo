import re
import pandas as pd
import bs4
import requests
import spacy
import os
from flask import Flask,request,jsonify,render_template
from spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import Matcher
from spacy.tokens import Span
import networkx as nx
import numpy
from neo4j.v1 import GraphDatabase,basic_auth
from tqdm import tqdm
from py2neo.data import Node,Relationship,Walkable
from py2neo import Graph,NodeMatcher,RelationshipMatcher
graph=Graph("bolt://localhost:7687",user = "neo4j", password = "123456")
app = Flask(__name__)
spacy_stopwords = STOP_WORDS
#print(spacy_stopwords)

def duplicate(items): 
    unique = [] 
    for item in items: 
        if item not in unique: 
            unique.append(item) 
    return unique 

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
# df = pd.read_csv('/home/swathi/Downloads/Sample.csv')
# df.shape

def get_entities(sent):
  ## chunk 1
  ent1 = ""
  ent2 = ""

  prv_tok_dep = ""    # dependency tag of previous token in the sentence
  prv_tok_text = ""   # previous token in the sentence

  prefix = ""
  modifier = ""
  en=[]

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
        en.append(tok.text.lower())
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
        en.append(tok.text.lower())
        #print(ent1)
        prefix = ""
        modifier = ""
        prv_tok_dep = ""
        prv_tok_text = ""

      ## chunk 4
      if tok.dep_.find("obj") == True:
        ent2 = modifier +" "+ prefix +" "+ tok.text
        en.append(tok.text.lower())
      ## chunk 5  
      # update variables
      prv_tok_dep = tok.dep_
      prv_tok_text = tok.text
  #############################################################

  return en

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
#common_words=['is','the','get','Is','Get','put','Put','The','a','A','An','an','Are','are','Were','were','was','Was','have','Have','has','Has','as','As','be','Be','me','you']

@app.route('/search',methods=['GET'])
def search():
  if request.method=='GET':
    user = request.json['key']
    input_data =user.split(" ")
    print(input_data)
    u=len(input_data)
    print(u)

    if u<=1:
      entities=user.lower()
      rel=user.lower()
      final_entities=[entities]
      final_relation=[rel]
    else:
      e=(get_named_entities(user))
      #if e == []:
      entities=get_entities(user)
      #else:
      #entities=e
      for i in e:
        entities.append(i)
      rel = get_relation(user)

      #final_entities=[i for i in entities if i not in question_words]
    #print(final_entities)

    
      final_relation=remove_stopwords(rel)
      f_entities=remove_stopwords(entities)
      print(f_entities)
      final_entities=[i for i in f_entities if i not in final_relation]
      print(final_entities)
      
    matcher = NodeMatcher(graph)
    rel_match=RelationshipMatcher(graph)

    if final_entities==[]:
      answer=[]
      relationship_match=graph.relationships.match((None,None),final_relation[0].lower())
      if relationship_match is not None:
        for rel in relationship_match:
          answer.append(rel.end_node['name'])
        return jsonify({"key":duplicate(answer)})
      else:
        return 'match not found'

    elif final_relation==[]:
      ques_match_node=matcher.match("target",name=final_entities[0].lower()).first()
      answer=[]
      if ques_match_node is not None:
        relationship_match=graph.relationships.match((ques_match_node,None),None)
        for rel in relationship_match:
          answer.append(rel.end_node['name'])
        return jsonify({"key":duplicate(answer)})
      else:
        return "match not found"

    elif final_entities==[] and final_relation==[]:
      return "match not available in database"

    else:
      count_rel =0

      for i in final_relation:
        count_ent=0
        count_rel+=1
        for ent in final_entities:
            count_ent+=1
            answer=[]
            ques_match_node=matcher.match("target",name=ent.lower()).first()
            print(ques_match_node)
            if ques_match_node is not None:
              relationship_match=graph.relationships.match((ques_match_node,None),i.lower()).first()
              print(relationship_match)
              if relationship_match is not None:
                return jsonify({"key":relationship_match.end_node['name']})
              else:
                r = graph.relationships.match((ques_match_node,None),None)
                print("hh")
                print(len(r))
                for rel in r:
                  answer.append(rel.end_node['name'])

                return jsonify({"key":duplicate(answer)})
            else:
              relationship_match=graph.relationships.match((None,None),i.lower())
              if relationship_match is not None:
                for rel in relationship_match:
                  answer.append(rel.end_node['name'])
                return jsonify({'key':duplicate(answer)})
              else:

                if count_rel==len(final_relation) and count_ent==len(final_entities):
                  return "no match found"
                else:
                  continue
@app.route('/add',methods=['POST'])
def add():
  if request.method=='POST':
    user_ques = request.json['key']['question']
    user_ans = request.json['key']['answer']  

    entity_pairs = []
    relations = []
    answers = []
    questions = []
    e=(get_named_entities(user_ques))
      #if e == []:
    entities=get_entities(user_ques)
      #else:
      #entities=e
    for i in e:
      entities.append(i)
    rel = get_relation(user_ques)

      #final_entities=[i for i in entities if i not in question_words]
    #print(final_entities)

    print(rel)
    final_relation=remove_stopwords(rel)
    print(final_relation)
    f_entities=remove_stopwords(entities)
    print(f_entities)
    final_entities=[i for i in f_entities if i not in final_relation]
    print(final_entities) 

    for i in final_relation:
      for ent in final_entities:
        entity_pairs.append(ent.lower())
        answers.append(user_ans)
        relations.append(i.lower())
        questions.append(user_ques)         
        print(entity_pairs)
        print(answers)
        print(relations)

    
    kg_df = pd.DataFrame({ 'ques':questions,'target':entity_pairs, 'edge':relations,'answer':answers})
    matcher = NodeMatcher(graph)
    rel_match=RelationshipMatcher(graph)
    for i in range(len(kg_df)):
      tx = graph.begin()

      rt=matcher.match("target",name=entity_pairs[i]).first()

      if rt is not None:
        b=rt
      else:
        b=Node("target",name=entity_pairs[i])
        tx.create(b)

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

      q=matcher.match("ques",name=questions[i]).first()
      if q is None:
        ques =  Node("ques",name=questions[i])
        tx.create(ques)
      else:
        ques=q  

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

      tx.commit()

    return "successfully added."

@app.route('/list',methods=['GET'])
def list():
  qa_pairs=[]
  matcher = NodeMatcher(graph)
  rel_match=RelationshipMatcher(graph)
  r = graph.relationships.match((None,None),"Answer is")
  if r is not None:
    for rel in r:
      qa_pairs.append({rel.start_node['name']:rel.end_node['name']})
  return jsonify({"key":qa_pairs})

    
      

if __name__=='__main__':
  app.run(debug=True)


