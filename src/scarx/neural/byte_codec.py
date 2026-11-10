"""Fixed UTF8 alphabet + prefix validity cursor, not a language model.

Mask implements RFC3629 plus the natural-text ASCII control policy. Valid text
alone says nothing about factual correctness, fluency or learned reasoning.
"""
import numpy as np


class UTF8Codec:
    eos=0
    bos=1
    size=258

    @staticmethod
    def encode(text):
        if not isinstance(text,str):raise ValueError('Text required')
        if any((ord(c)<32 and c not in '\t\n\r') or ord(c)==127 for c in text):raise ValueError('Unsupported ASCII control character')
        try:data=text.encode('utf-8',errors='strict')
        except UnicodeEncodeError as exc:raise ValueError('Invalid Unicode scalar') from exc
        return [int(b)+2 for b in data]

    @staticmethod
    def decode(tokens):
        if any(type(t) not in (int,np.int32,np.int64) or not 2<=t<258 for t in tokens):raise ValueError('Expected byte tokens, not BOS/EOS')
        try:text=bytes(int(t)-2 for t in tokens).decode('utf-8',errors='strict')
        except UnicodeDecodeError as exc:raise ValueError('Invalid UTF8 token sequence') from exc
        UTF8Codec.encode(text)
        return text


class UTF8Cursor:
    def __init__(self):
        self.remaining=0;self.low=128;self.high=191;self.emitted=0

    def mask(self,budget,outputs=268):
        if type(budget) is not int or budget<1 or outputs<258:raise ValueError('Invalid byte budget/output count')
        result=np.zeros(outputs,bool)
        if self.remaining:
            if budget<self.remaining:raise ValueError('Budget cannot complete pending character')
            result[self.low+2:self.high+3]=True
        else:
            if self.emitted:result[0]=True
            result[9+2]=True;result[10+2]=True;result[13+2]=True;result[32+2:127+2]=True
            if budget>=2:result[194+2:224+2]=True
            if budget>=3:result[224+2:240+2]=True
            if budget>=4:result[240+2:245+2]=True
        return result

    def accept(self,token):
        if type(token) not in (int,np.int32,np.int64) or not 2<=token<258:raise ValueError('Expected byte token')
        b=int(token)-2
        if self.remaining:
            if not self.low<=b<=self.high:raise ValueError('Invalid UTF8 continuation')
            self.remaining-=1;self.low=128;self.high=191
        elif b<128:
            if b<32 and b not in (9,10,13) or b==127:raise ValueError('Unsupported ASCII control')
        elif 194<=b<=223:self.remaining=1;self.low=128;self.high=191
        elif 224<=b<=239:
            self.remaining=2;self.low=160 if b==224 else 128;self.high=159 if b==237 else 191
        elif 240<=b<=244:
            self.remaining=3;self.low=144 if b==240 else 128;self.high=143 if b==244 else 191
        else:raise ValueError('Invalid UTF8 start byte')
        self.emitted+=1
