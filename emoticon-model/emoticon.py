import re
import numpy as np
from nltk.tokenize import WordPunctTokenizer
tok =WordPunctTokenizer()

import emoji

def extract_emojis(str):
  emoji_all = ''.join(c for c in str if c in emoji.UNICODE_EMOJI)
  return (emoji_all)

# Happy, SAd, Clam,Angry
Emoti_dic ={
    "😂":[100,0,0,0],#1.FACE WITH TEARS OF JOY
    "😍":[92.3,0,7.7,0], #2.SMILING FACE WITH HEART-SHAPED EYES
    "😭":[0,100,0,0], #3.LOUDLY CRYING FACE
    "😘":[92.3,0,7.7,0], #4.FACE THROWING A KISS
    "😊":[46.2,0,46.2,7.7], #5.SMILING FACE WITH SMILING EYES
    "😁":[92.3,0,7.7,0], #6.GRINNING FACE WITH SMILING EYES
    "☺":[46.2,0,53.8,0], #7.WHITE SMILING FACE
    "😩":[0,92.3,7.7,0], #8.WEARY FACE
    "😉":[61.5,0,38.5,0],#10.WINKING FACE
    "😄":[92.3,0,7.7,0],#11.SMILING FACE WITH OPEN MOUTH AND SMILING EYES
    "😃":[100,0,0,0],#13.SMILING FACE WITH OPEN MOUTH
    "😔":[0,100,0,0],#14.PENSIVE FACE
    "😜":[90,0,10,0],#16.FACE WITH STUCK-OUT TONGUE AND WINKING EYE
    "😡":[0,0,0,100],#18.POUTING FACE
    "😎":[72.7,0,27.3,0],#19.SMILING FACE WITH SUNGLASSES
    "😢":[0,100,0,0],#20.CRYING FACE
    "😋":[90.9,0,9.1,0],#21.FACE SAVOURING DELICIOUS FOOD
    "😴":[0,20,80,0],#22.SLEEPING FACE
    "😌":[18.2,0,81.8,0],#23.RELIEVED FACE
    "😞":[0,100,0,0],#24.DISAPPOINTED FACE
    "😆":[27.3,18.2,9.1,45.5],#25.SMILING FACE WITH OPEN MOUTH AND TIGHTLY-CLOSED EYES
    "😝":[81.8,0,18.2,0],#26.FACE WITH STUCK-OUT TONGUE AND TIGHTLY-CLOSED EYES
    "😪":[0,90.9,9.1,0],#27.SLEEPY FACE
    "😫":[0,81.8,0,18.2],#28.TIRED FACE
    "😅":[90.9,0,9.1,0],#29.SMILING FACE WITH OPEN MOUTH AND COLD SWEAT
    "😀":[100,0,0,0],#30.GRINNING FACE
    "😚":[75,0,25,0],#31.KISSING FACE WITH CLOSED EYES
    "😥":[0,100,0,0],#32.DISAPPOINTED BUT RELIEVED FACE
    "😕":[0,91.7,0,8.3],#33.CONFUSED FACE
    "😤":[0,8.3,0,91.7],#34.FACE WITH LOOK OF TRIUMPH
    "😈":[8.3,8.3,0,83.3],#35.SMILING FACE WITH HORNS
    "😰":[0,91.7,0,8.3],#36.FACE WITH OPEN MOUTH AND COLD SWEAT
    "😑":[0,41.7,41.7,16.7],#37.EXPRESSIONLESS FACE
    "😠":[0,8.3,0,91.7],#38.ANGRY FACE
    "❤":[8.3,33.3,16.7,41.7],#39.HEAVY BLACK HEART
    "♥":[100,0,0,0], #40.BLACK HEART SUIT
    "😹":[100,0,0,0],#41.CAT FACE WITH TEARS OF JOY
    "💔":[0,100,0,0],#42.BROKEN HEART
    "👊":[54.5,0,27.3,18.2],#43.FISTED HAND SIGN
    "😻":[100,0,0,0],#45.SMILING CAT FACE WITH HEART-SHAPED EYES
    "💘":[100,0,0,0],#46.HEART WITH ARROW
    "🎊":[91.7,0,8.3,0],#47.CONFETTI BALL
    "🎶":[75,0,25,0],#48.MULTIPLE MUSICAL NOTES
    "💞":[91.7,0,8.3,0],#49.REVOLVING HEARTS
    "🙊":[36.4,27.3,27.3,9.1],#51.SPEAK-NO-EVIL MONKEY
    "💋":[90.9,0,9.1,0],#52.KISS MARK
    "💗":[90.9,0,9.1,0],#53.GROWING HEART
    "🎉":[100,0,0,0],#54.PARTY POPPER
    "💖":[100,0,0,0],#55.SPARKLING HEART
    "🙈":[72.7,9.1,18.2,0],#56.SEE-NO-EVIL MONKEY
    "💪":[63.6,0,18.2,18.2],#57.FLEXED BICEPS
    "✌":[54.5,0,45.,5,0],#58.VICTORY HAND
    "👌":[54.5,0,45.,5,0], #59.OK HAND SIGN
    "💕":[90.9,0,9.1,0], #60.TWO HEARTS
    "👏":[100,0,0,0], #61.CLAPPING HANDS SIGN
    "👍":[72.7,0,18.2,9.1], #62.THUMBS UP SIGN
}



if __name__ == '__main__':
    emoticons = extract_emojis('👍 Gon👍 Sadeepa👍')
    emoticon_result = [0,0,0,0]
    for e in emoticons:
        e_value = Emoti_dic.get(e)
        emoticon_result = np.array(emoticon_result)+np.array(e_value)


    print(np.divide(emoticon_result,len(emoticons)))

# emoti_pattern =re.compiler(r'{'+'|'.join(Emoti_dic.keys())+r'}')
#
# def processEmoti(text):
#     emo=emoti_pattern.sub(lambda y:Emoti_dic[y.group()],text)
#     print(emo)


