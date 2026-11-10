"""Experimental full input-events -> one ALIF core -> word decoder.

Not imported by serving. This closes a testable data path, not a chat/vision
quality claim. No lookup answer, pretrained model, backprop or BPTT.
"""
import io,json,zipfile
import numpy as np
from ..neural.unified import UnifiedCore
from ..neural.input_events import InputEvents
from ..neural.language import tokenize,render
from .readout_conditioning import ConditionedReadout


class EventDialogue:
    def __init__(self,vocabulary,seed,condition):
        if condition not in ('bag_raw','ordered_raw','ordered_normalized','ordered_normalized_full'):raise ValueError('Unknown prototype condition')
        self.model=UnifiedCore(vocabulary,seed);self.encoder=InputEvents(len(vocabulary));self.condition=condition
        self.head=ConditionedReadout(self.model.core.w_out,self.model.core.bias,'normalized' if 'normalized' in condition else 'raw')
        self.model.core.w_out=self.head.weights;self.model.core.bias=self.head.bias
        self.responses_seen=0

    def _mask(self,position):
        mask=np.zeros(len(self.head.bias),bool);mask[:len(self.model.vocabulary)]=True;mask[1]=False
        if position==0:mask[0]=False
        return mask

    def _prepare(self,messages,track=False,image=None):
        events=self.encoder.dialogue(messages)
        if image is not None:
            events=np.concatenate((events,self.encoder.image(image)))
            if len(events)>self.encoder.maximum_events:raise ValueError('Input exceeds256event budget')
        c=self.model.core;self.model.boundary()
        if track:
            c.elig_in.fill(0);c.elig_rec.fill(0);c.pre.fill(0);c.post.fill(0)
            c.rows[:]=(np.arange(12)+c.updates*12)%48
        text_slice=slice(len(self.model.vocabulary),self.encoder.image_start)
        summary=events[:,text_slice].sum(0);summary/=max(1.,float(np.linalg.norm(summary)))
        if self.condition=='bag_raw':
            events=events.copy();events[:,text_slice]=events[:,text_slice].mean(0)
        for event in events:self.model.tick(event,track,'input_events','experimental public dialogue')
        return summary

    def calibrate(self,examples):
        if self.head.frozen:raise ValueError('Already calibrated')
        for example in examples:
            context=self._prepare(example['messages']);previous=1
            for position,word in enumerate(tokenize(example['target'])+['<eos>']):
                self.model.word_step(previous,context,position)
                self.head.observe(self.model.core.features)
                previous=self.model.ids[word]
        self.head.freeze()

    def learn(self,messages,target):
        words=tokenize(target)
        if not words or len(words)>24 or any(t not in self.model.ids for t in words):raise ValueError('Target outside training vocabulary')
        context=self._prepare(messages,True);previous=1;loss=0.
        for position,word in enumerate(words+['<eos>']):
            self.model.word_step(previous,context,position,True)
            token=self.model.ids[word];mask=self._mask(position);c=self.model.core
            probabilities=self.head.probabilities(c.features,mask)
            loss+=self.head.learn(c.features,token,mask)
            reward=float(probabilities[token]);advantage=float(np.clip(reward-c.reward_mean,-1,1));c.reward_mean=.95*c.reward_mean+.05*reward
            if self.condition=='ordered_normalized_full':
                for name,trace in (('in',c.elig_in),('rec',c.elig_rec)):
                    weights=getattr(c,'w_'+name)
                    weights[c.rows]+=np.clip(c.config.hidden_lr*advantage*trace,-c.config.update_clip,c.config.update_clip)
                    np.clip(weights,-c.config.weight_clip,c.config.weight_clip,out=weights)
            c.updates+=1;previous=token
        self.responses_seen+=1
        return loss/(len(words)+1)

    def generate(self,messages,image=None,max_words=16):
        context=self._prepare(messages,image=image);previous=1;words=[];stopped=False
        for position in range(max_words):
            self.model.word_step(previous,context,position)
            token=int(np.argmax(self.head.probabilities(self.model.core.features,self._mask(position))))
            if token==0:stopped=True;break
            words.append(self.model.vocabulary[token]);previous=token
        return {'text':render(words),'tokens':words,'stopped':stopped}

    def score(self,examples):
        clone=self.loads(self.dumps());outputs=[];nll=0.;tokens=0;correct=0;oov=0
        for example in examples:
            response=clone.generate(example['messages']);target=tokenize(example['target'])
            outputs.append({'id':example['id'],'target':example['target'],'generated':response['text'],
              'exact':tokenize(response['text'])==target,'repeated':any(response['tokens'].count(t)>3 for t in set(response['tokens']))})
            context=clone._prepare(example['messages']);previous=1
            for position,word in enumerate(target+['<eos>']):
                clone.model.word_step(previous,context,position);p=clone.head.probabilities(clone.model.core.features,clone._mask(position));tokens+=1
                if word not in clone.model.ids:oov+=1;previous=1;nll-=np.log(1e-12);continue
                token=clone.model.ids[word];nll-=np.log(max(float(p[token]),1e-12));correct+=int(np.argmax(p)==token);previous=token
        return {'exact':sum(r['exact'] for r in outputs)/len(outputs),'teacher_forced_nll':float(nll/tokens),
          'teacher_forced_token_accuracy':correct/tokens,'oov_tokens':oov,'target_tokens':tokens,
          'repetition_rate':sum(r['repeated'] for r in outputs)/len(outputs),'outputs':outputs}

    def dumps(self):
        stream=io.BytesIO()
        with zipfile.ZipFile(stream,'w') as archive:
            archive.writestr(zipfile.ZipInfo('core.zip'),self.model.dumps());archive.writestr(zipfile.ZipInfo('head.npz'),self.head.dumps())
            archive.writestr(zipfile.ZipInfo('meta.json'),json.dumps({'format':'scarx-event-dialogue-experiment-v1','condition':self.condition,'responses_seen':self.responses_seen}))
        return stream.getvalue()

    @classmethod
    def loads(cls,payload):
        if len(payload)>4000000:raise ValueError('Experimental checkpoint too large')
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if set(archive.namelist())!={'core.zip','head.npz','meta.json'}:raise ValueError('Wrong experimental format')
            meta=json.loads(archive.read('meta.json'));core=UnifiedCore.loads(archive.read('core.zip'));head=ConditionedReadout.loads(archive.read('head.npz'))
        if meta['format']!='scarx-event-dialogue-experiment-v1':raise ValueError('Format')
        model=cls(core.vocabulary,17,meta['condition']);model.model=core;model.head=head
        if head.weights.shape!=core.core.w_out.shape or head.bias.shape!=core.core.bias.shape:raise ValueError('Readout shape')
        if not np.array_equal(head.weights,core.core.w_out) or not np.array_equal(head.bias,core.core.bias):raise ValueError('Readout checkpoint mismatch')
        core.core.w_out=head.weights;core.core.bias=head.bias;model.responses_seen=meta['responses_seen'];return model
