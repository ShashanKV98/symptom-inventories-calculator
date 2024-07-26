# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 15:37:23 2023

We will use sentence transformers to map each symptom description
to a 384 dimensional dense vector space

get cosine similarity of all symptoms to symptoms, save to a pkl

all-MiniLM-L6-v2: 
    small (80 MB) and fast model, with only 6 layers
    
all-MiniLM-L12-v1: 
    small (120 MB) and fast model, with 12 layers

all-roberta-large-v1
    v. large (160GB), 768 dimensions, 12 hidden layers, too big!
    
@author: u6029515
"""

from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

model = SentenceTransformer('all-MiniLM-L12-v2')

def compare_pair(A=['the bark of the willow tree is medicinal'], 
                 B=['some trees have healing properties and clinical applications']): 
    return util.cos_sim(model.encode(A), model.encode(B))

# print(compare_pair())

RPQ_data = [['RPQ1', 'DizzinessScale',  'Feelings of dizziness'],
            ['RPQ2', 'NauseaScale',     'Nausea and or vomiting'],
            ['RPQ3', 'NoiseSensScale',  'Noise sensitivity easily upset by loud noise'],
            ['RPQ4', 'SleepDistScale',  'Sleep disturbance'],
            ['RPQ5', 'FatigueScale',    'Fatigue, tiring more easily'],
            ['RPQ6', 'IrritableScale',  'Being irritable, easily angered'],
            ['RPQ7', 'DepressedScale',  'Feeling depressed or tearful'],
            ['RPQ8', 'FrustratedScale', 'Feeling frustrated or impatient '],
            ['RPQ9', 'ForgetfulScale',  'Forgetfulness, poor memory'],
            ['RPQ10', 'PoorConcScale',  'Poor concentration'],
            ['RPQ11', 'LongToThinkScale', 'Taking longer to think'],
            ['RPQ12', 'BlurredVisionScale', 'Blurred vision'],
            ['RPQ13', 'LightSensScale', 'Light sensitivity, easily upset by bright light'],
            ['RPQ14', 'DblVisionScale', 'Double vision'],
            ['RPQ15', 'RestlessScale',  'Restlessness'],
            ['RPQ16', 'HeadachesScale', 'Headaches']]

RPQ_descriptions = [i[2] for i in RPQ_data]

BSI_data = [
    ['BSI1', 'Faintness or dizziness'],
    ['BSI2', 'Feeling no interest in things'],
    ['BSI3', 'Nervousness or shakiness'],
    ['BSI4', 'Pains in heart or chest'],
    ['BSI5', 'Feeling lonely'],
    ['BSI6', 'Feeling tense or keyed up'],
    ['BSI7', 'Nausea or upset stomach'],
    ['BSI8', 'Feeling blue'],
    ['BSI9', 'Suddenly scared for no reason'],
    ['BSI10', 'Trouble getting your breath'], # toulbe spelling error corrected manually
    ['BSI11', 'Feelings of worthlessness'],
    ['BSI12', 'Spells of terror or panic'],
    ['BSI13', 'Numbness of tingling in parts of body'],
    ['BSI14', 'Feeling hopeless about the future'],
    ['BSI15', "Feeling so restless you couldn't sit still"],
    ['BSI16', 'Feeling weak in parts of your body'],
    ['BSI17', 'Thoughts of ending your life'],
    ['BSI18', 'Feeling fearful']]

BSI_descriptions = [i[1] for i in BSI_data]

NSI_data = [
    ['NSI1', 'Feeling dizzy'],
    ['NSI2', 'Loss of balance'],
    ['NSI3', 'Poor coordination, clumsy'],
    ['NSI4', 'Headaches'],
    ['NSI5', 'Nausea'],
    ['NSI6', 'Vision problems, blurring, trouble seeing'],
    ['NSI7', 'Sensitivity to light'],
    ['NSI8', 'Hearing difficulty'],
    ['NSI9', 'Sensitivity to noise'],
    ['NSI10', 'Numbness or tingling on parts of my body'],
    ['NSI11', 'Change in taste and/or smell'],
    ['NSI12', 'Loss of appetite or increased appetite'],
    ['NSI13', 'Poor concentration, cant pay attention, easily distracted'],
    ['NSI14', 'Forgetfulness, cant remember things'],
    ['NSI15', 'Difficulty making decisions'],
    ['NSI16', 'Slowed thinking, difficulty getting organized, cant finish things'],
    ['NSI17', 'Fatigue, loss of energy, getting tired easily'],
    ['NSI18', 'Difficulty falling or staying asleep'],
    ['NSI19', 'Feeling anxious or tense'],
    ['NSI20', 'Feeling depressed or sad'],
    ['NSI21', 'Irritability, easily annoyed'],
    ['NSI22', 'Poor frustration tolerance, feeling easily overwhelmed by things']]

NSI_descriptions = [i[1] for i in NSI_data]

SCL_data = [
    ['SCL1', 'Headaches'],
    ['SCL2', 'Nervousness or shakiness inside'],
    ['SCL3', 'Unwanted thoughts, words, or ideas that wont leave your mind'],
    ['SCL4', 'Faintness or dizziness'],
    ['SCL5', 'Loss of sexual interest or pleasure'],
    ['SCL6', 'Feeling critical of others'],
    ['SCL7', 'The idea that someone else can control your thoughts'],
    ['SCL8', 'Feeling others are to blame for most of your troubles'],
    ['SCL9', 'Trouble remembering things'],
    ['SCL10', 'Worried about sloppiness or carelessness'],
    ['SCL11', 'Feeling easily annoyed or irritated'],
    ['SCL12', 'Pains in heart or chest'],
    ['SCL13', 'Feeling afraid in open spaces or on the streets'],
    ['SCL14', 'Feeling low in energy or slowed down'],
    ['SCL15', 'Thoughts of ending your life'],
    ['SCL16', 'Hearing voices that other people do not hear'],
    ['SCL17', 'Trembling'],
    ['SCL18', 'Feeling that most people cannot be trusted'],
    ['SCL19', 'Poor appetite'],
    ['SCL20', 'Crying easily'],
    ['SCL21', 'Feeling shy or uneasy with the opposite sex'],
    ['SCL22', 'Feelings of being trapped or caught'],
    ['SCL23', 'Suddenly scared for no reason'],
    ['SCL24', 'Temper outbursts that you could not control'],
    ['SCL25', 'Feeling afraid to go out of your house alone'],
    ['SCL26', 'Blaming yourself for things'],
    ['SCL27', 'Pains in lower back'],
    ['SCL28', 'Feeling blocked in getting things done'],
    ['SCL29', 'Feeling lonely'],
    ['SCL30', 'Feeling blue'],
    ['SCL31', 'Worrying too much about things'],
    ['SCL32', 'Feeling no interest in things'],
    ['SCL33', 'Feeling fearful'],
    ['SCL34', 'Your feelings being easily hurt'],
    ['SCL35', 'Other people being aware of your private thoughts'],
    ['SCL36', 'Feeling others do not understand you or are unsympathetic'],
    ['SCL37', 'Feeling that people are unfriendly or dislike you'],
    ['SCL38', 'Having to do things very slowly to insure correctness'],
    ['SCL39', 'Heart pounding or racing'],
    ['SCL40', 'Nausea or upset stomach'],
    ['SCL41', 'Feeling inferior to others'],
    ['SCL42', 'Soreness of your muscles'],
    ['SCL43', 'Feeling that you are watched or talked about by others'],
    ['SCL44', 'Trouble falling asleep'],
    ['SCL45', 'Having to check and double-check what you do'],
    ['SCL46', 'Difficulty making decisions'],
    ['SCL47', 'Feeling afraid to travel on buses, subways, or trains'],
    ['SCL48', 'Trouble getting your breath'],
    ['SCL49', 'Hot or cold spells'],
    ['SCL50', 'Having to avoid certain things, places, or activities because they frighten you'],
    ['SCL51', 'Your mind going blank'],
    ['SCL52', 'Numbness or tingling in parts of your body'],
    ['SCL53', 'A lump in your throat'],
    ['SCL54', 'Feeling hopeless about the future'],
    ['SCL55', 'Trouble concentrating'],
    ['SCL56', 'Feeling weak in parts of your body'],
    ['SCL57', 'Feeling tense or keyed up'],
    ['SCL58', 'Heavy feelings in your arms or legs'],
    ['SCL59', 'Thoughts of death or dying'],
    ['SCL60', 'Overeating'],
    ['SCL61', 'Feeling uneasy when people are watching or talking about you'],
    ['SCL62', 'Having thoughts that are not your own'],
    ['SCL63', 'Having urges to beat, injure, or harm someone'],
    ['SCL64', 'Awakening in the early morning'],
    ['SCL65', 'Having to repeat the same actions such as touching, counting, washing'],
    ['SCL66', 'Sleep that is restless or disturbed'],
    ['SCL67', 'Having urges to break or smash things'],
    ['SCL68', 'Having ideas or beliefs that others do not share'],
    ['SCL69', 'Feeling very self-conscious with others'],
    ['SCL70', 'Feeling uneasy in crowds, such as shopping or at a movie'],
    ['SCL71', 'Feeling everything is an effort'],
    ['SCL72', 'Spells of terror or panic'],
    ['SCL73', 'Feeling uncomfortable about eating or drinking in public'],
    ['SCL74', 'Getting into frequent arguments'],
    ['SCL75', 'Feeling nervous when you are left alone'],
    ['SCL76', 'Others not giving you proper credit for your achievements'],
    ['SCL77', 'Feeling lonely even when you are with people'],
    ['SCL78', 'Feeling so restless you couldnt sit still'],
    ['SCL79', 'Feelings of worthlessness'],
    ['SCL80', 'The feeling that  something bad is going to happen to you'],
    ['SCL81', 'Shouting or throwing things'],
    ['SCL82', 'Feeling that you will faint in public'],
    ['SCL83', 'Feeling that people will take advantage of you if you let them'],
    ['SCL84', 'Having thoughts about sex that bother you a lot'],
    ['SCL85', 'The idea that you should be punished for your sins'],
    ['SCL86', 'Thoughts and images of a frightening nature'],
    ['SCL87', 'The idea that something serious is wrong with your body'],
    ['SCL88', 'Never feeling close to another person'],
    ['SCL89', 'Feelings of guilt'],
    ['SCL90', 'The idea that something is wrong with your mind']]

SCL_descriptions = [i[1] for i in SCL_data]

NSI34_data = [
    ['NSI34_1', 'Feeling dizzy'],
    ['NSI34_2', 'Loss of balance'],
    ['NSI34_3', 'Poor coordination, clumsy'],
    ['NSI34_4', 'Headaches'],
    ['NSI34_5', 'Nausea'],
    ['NSI34_6', 'Vision problems, blurring, trouble seeing'],
    ['NSI34_7', 'Sensitivity to light'],
    ['NSI34_8', 'Hearing difficulty'],
    ['NSI34_9', 'Sensitivity to noise'],
    ['NSI34_10', 'Numbness or tingling on parts of my body'],
    ['NSI34_11', 'Change in taste and/or smell'],
    ['NSI34_12', 'Loss of ability to speak'],
    ['NSI34_13', 'Inability to see color'],
    ['NSI34_14', 'Inability to sleep for over 1 week'],
    ['NSI34_15', 'Rapid changes in body temp'],
    ['NSI34_16', 'Loss of appetite or increased appetite'],
    ['NSI34_17', 'Poor concentration, cant pay attention, easily distracted'],
    ['NSI34_18', 'Forgetfulness, cant remember things'],
    ['NSI34_19', 'Difficulty making decisions'],
    ['NSI34_20', 'Slowed thinking, difficulty getting organized, cant finish things'],
    ['NSI34_21', 'Difficulty remembering events from HS'],
    ['NSI34_22', 'Difficulty recalling names of family/friends'],
    ['NSI34_23', 'Loss of ability to spell'],
    ['NSI34_24', 'Forgetting how to write'],
    ['NSI34_25', 'Fatigue, loss of energy, getting tired easily'],
    ['NSI34_26', 'Difficulty falling or staying asleep'],
    ['NSI34_27', 'Feeling anxious or tense'],
    ['NSI34_28', 'Feeling depressed or sad'],
    ['NSI34_29', 'Irritability, easily annoyed'],
    ['NSI34_30', 'Poor frustration tolerance, feeling easily overwhelmed by things'],
    ['NSI34_31', 'Belief Im being followed'],
    ['NSI34_32', 'Belief others can steal my thoughts'],
    ['NSI34_33', 'Desire to eat non-food items'],
    ['NSI34_34', 'Black outs/loss of time']]

#NSI34_descriptions = [i[1] for i in NSI34_data]

NSI_emb = model.encode(NSI_descriptions)
SCL_emb = model.encode(SCL_descriptions)
RPQ_emb = model.encode(RPQ_descriptions)
BSI_emb = model.encode(BSI_descriptions)
#NSI34_emb = model.encode(NSI34_descriptions)

cos_score_NSI_SCL = np.array(util.cos_sim(NSI_emb, SCL_emb))
cos_score_NSI_RPQ = np.array(util.cos_sim(NSI_emb, RPQ_emb))
cos_score_NSI_BSI = np.array(util.cos_sim(NSI_emb, BSI_emb))
#cos_score_NSI_NSI34 = np.array(util.cos_sim(NSI_emb, NSI34_emb))
cos_score_SCL_RPQ = np.array(util.cos_sim(SCL_emb, RPQ_emb))
cos_score_SCL_BSI = np.array(util.cos_sim(SCL_emb, BSI_emb))
#cos_score_SCL_NSI34 = np.array(util.cos_sim(SCL_emb, NSI34_emb))
cos_score_RPQ_BSI = np.array(util.cos_sim(RPQ_emb, BSI_emb))
#cos_score_RPQ_NSI34 = np.array(util.cos_sim(RPQ_emb, NSI34_emb))
#cos_score_BSI_NSI34 = np.array(util.cos_sim(BSI_emb, NSI34_emb))

cos_score_NSI_NSI = np.array(util.cos_sim(NSI_emb, NSI_emb))
cos_score_SCL_SCL = np.array(util.cos_sim(SCL_emb, SCL_emb))
cos_score_BSI_BSI = np.array(util.cos_sim(BSI_emb, BSI_emb))
cos_score_RPQ_RPQ = np.array(util.cos_sim(RPQ_emb, RPQ_emb))

score_dict = {}
score_dict['NSI_SCL'] = cos_score_NSI_SCL
score_dict['NSI_RPQ'] = cos_score_NSI_RPQ
score_dict['NSI_BSI'] = cos_score_NSI_BSI
score_dict['SCL_RPQ'] = cos_score_SCL_RPQ
score_dict['RPQ_BSI'] = cos_score_RPQ_BSI
score_dict['SCL_BSI'] = cos_score_SCL_BSI

score_dict['NSI_NSI'] = cos_score_NSI_NSI
score_dict['SCL_SCL'] = cos_score_SCL_SCL
score_dict['BSI_BSI'] = cos_score_BSI_BSI
score_dict['RPQ_RPQ'] = cos_score_RPQ_RPQ

text_dict = {}
text_dict['RPQ'] = RPQ_descriptions
text_dict['NSI'] = NSI_descriptions
text_dict['BSI'] = BSI_descriptions
text_dict['SCL'] = SCL_descriptions

pkl.dump(score_dict, open( "score_dict.p", "wb" ) )
pkl.dump(text_dict,  open( "text_dict.p",  "wb" ) )
# loaded_score_dict = pkl.load( open( "score_dict.p", "rb" ) )



