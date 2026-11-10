"""Experimental order/color-preserving frontend; no learned model or answers.

Not enabled for serving. Uses the existing G1 channel budget and can feed the
same recurrent core in chronological order. Sequence limits are explicit.
"""
import base64,hashlib,io
import numpy as np
from .language import tokenize


class InputEvents:
    text_channels=128
    image_channels=64
    maximum_events=256

    def __init__(self,word_channels):
        if type(word_channels) is not int or not 3<=word_channels<=256:raise ValueError('word_channels3–256')
        self.word_channels=word_channels;self.image_start=word_channels+128;self.dimensions=word_channels+198

    @staticmethod
    def token_vector(token):
        result=np.zeros(128,np.float32)
        units=[(token,1.)]+[(token[i:i+3],.25) for i in range(max(0,len(token)-2))]
        for unit,weight in units:
            code=hashlib.blake2b(unit.encode('utf-8'),digest_size=16).digest()
            bits=np.unpackbits(np.frombuffer(code,dtype=np.uint8)).astype(np.float32)
            result+=weight*(2*bits-1)/np.sqrt(128.)
        return result/max(1.,float(np.linalg.norm(result)))

    def _event(self,language,image,position,kind,speaker):
        row=np.zeros(self.dimensions,np.float32)
        row[-6:]=[language,image,kind,position/256.,speaker,.2]
        return row

    def text(self,text,speaker='user'):
        if speaker not in ('user','assistant'):raise ValueError('Unknown speaker')
        if not isinstance(text,str) or not 1<=len(text.strip())<=2048:raise ValueError('Text1–2048characters')
        tokens=tokenize(text)
        if not tokens or len(tokens)>192 or any(len(t)>64 for t in tokens):raise ValueError('Text maximum192tokens,64characters/token')
        role=0. if speaker=='user' else 1.
        rows=[self._event(1,0,0,-1,role)]
        for index,token in enumerate(tokens,1):
            row=self._event(1,0,index,0,role)
            row[self.word_channels:self.image_start]=self.token_vector(token)
            rows.append(row)
        rows.append(self._event(1,0,len(rows),1,role))
        return np.stack(rows)

    def dialogue(self,messages):
        if not isinstance(messages,list) or not 1<=len(messages)<=12:raise ValueError('Dialogue1–12messages')
        events=[]
        for message in messages:
            if not isinstance(message,dict):raise ValueError('Invalid message')
            events.append(self.text(message.get('text'),message.get('role')))
        result=np.concatenate(events)
        if len(result)>self.maximum_events:raise ValueError('Dialogue exceeds256events; nothing silently truncated')
        return result

    def image(self,payload):
        from PIL import Image
        if not isinstance(payload,bytes) or not 1<=len(payload)<=500000:raise ValueError('Image bytes1–500KB')
        try:image=Image.open(io.BytesIO(payload))
        except OSError as exc:raise ValueError('Invalid image') from exc
        with image:
            if image.format not in ('PNG','JPEG') or min(image.size)<8 or image.width*image.height>1048576 or max(image.size)/min(image.size)>4:raise ValueError('PNG/JPEG8px–1Mpixels, aspect<=4')
            rgb=np.asarray(image.convert('RGB').resize((16,16),Image.Resampling.BOX),np.float32)/255
        rows=[self._event(0,1,0,-1,0)]
        for y in range(4):
            for x in range(4):
                row=self._event(0,1,len(rows),0,0)
                row[self.image_start:self.image_start+48]=rgb[y*4:y*4+4,x*4:x*4+4].reshape(-1)
                row[self.image_start+48:self.image_start+50]=[x/3,y/3]
                rows.append(row)
        rows.append(self._event(0,1,len(rows),1,0))
        return np.stack(rows)

    def image_uri(self,uri):
        if not isinstance(uri,str) or len(uri)>700000 or not uri.startswith(('data:image/png;base64,','data:image/jpeg;base64,')):raise ValueError('Expected bounded PNG/JPEG dataURI')
        return self.image(base64.b64decode(uri.split(',',1)[1],validate=True))

    def multimodal(self,prompt,payload):
        result=np.concatenate((self.text(prompt),self.image(payload)))
        if len(result)>self.maximum_events:raise ValueError('Multimodal input exceeds256events')
        return result
