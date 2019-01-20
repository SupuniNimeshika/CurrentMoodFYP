import re

def CalEmoPolarity(text):

    EmoMsg =[]
    EmoMsg.append(text)

    #calculate Polarity if Positive Emoticons
    sumPostiveEmo =0
    regPos =r'EMPOS_0[.][0-9]+'

    print("Positive Emoticon weights detected")
    for words in EmoMsg :
        Emo_Pos_Matches =re.findall(regPos,words)
        for match in Emo_Pos_Matches:
            #print("Full match: %s" %(match))
            a,PosValue =match.split("_")
            #print (PosValue)
            sumPostiveEmo=sumPostiveEmo+float(PosValue)

        PosEmoCount =len(Emo_Pos_Matches)
        print("No of total positive Emoticons: %.0f" % PosEmoCount)
        print("Sum of all Positive Emoticons: %.3f" % sumPostiveEmo)


    PosPolarity =0
    if(PosEmoCount !=0):
        PosPolarity =sumPostiveEmo/PosEmoCount
    print("Positive Polarity: %.3f" % round(PosPolarity,3))
    print(Emo_Pos_Matches)

    #Calcualte Polarity if Neutral Emoticons4
    sumNeutralEmo =0
    regNEU =r'EMONEU_-?0[.][0-9]+'

    print()
    print("Neutral Emoticon Weights detected")
    for words in EmoMsg:
        Emo_Neu_Matches =re.findall(regNEU,words)
        for match in Emo_Neu_Matches:
            #print ("Full match: %s" % (match))
            a,NeuValue =match.split("_")
            #print(NeuValue)
            sumNeutralEmo=sumNeutralEmo+float(NeuValue)

    NeuEmoCount =len(Emo_Neu_Matches)
    if (NeuEmoCount != 0):
        NeuPolarity = sumNeutralEmo / NeuEmoCount
    print("Neutral Polarity: %.3f" % round(PosPolarity, 3))
    print(Emo_Neu_Matches)

#calculate Polarity if Negative Emoticons
    sumNegativeEmo =0
    regNEG =r'EMPOS_0[.][0-9]+'

    print("Negative Emoticon weights detected")
    for words in EmoMsg :
        Emo_Neg_Matches =re.findall(regNEG,words)
        for match in Emo_Neg_Matches:
            #print("Full match: %s" %(match))
            a,NegValue =match.split("_")
            #print (PosValue)
            sumNegativeEmo=sumNegativeEmo+float(NegValue)

        NegEmoCount =len(Emo_Neg_Matches)
        print("No of total Negative Emoticons: %.0f" % NegEmoCount)
        print("Sum of all Negative Emoticons: %.3f" % sumNegativeEmo)


    NegPolarity =0
    if(NegEmoCount !=0):
        NegPolarity =sumNegativeEmo/NegEmoCount
    print("Negative Polarity: %.3f" % round(NegPolarity,3))
    print(Emo_Neg_Matches)

    print()
    emoticonSentimentone=(round(PosPolarity,3)+round(NegPolarity,3)+round(NegPolarity,3))/3.0

    Total_Emoticons =PosEmoCount+NeuEmoCount+NegEmoCount
    if(Total_Emoticons !=0):
        emoticonSentiment =(round(PosPolarity,3)+round(NegPolarity,3)+round(NegPolarity,3))/Total_Emoticons

