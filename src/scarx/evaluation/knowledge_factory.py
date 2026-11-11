"""Source-grounded dataset factory primitives; no neural inference/training."""
import ast,hashlib,json,re,unicodedata
from collections import defaultdict
from decimal import Decimal,InvalidOperation
from pathlib import Path

PUBLIC_COMMIT='495dab2f4bac0180e4d1cd28674391f055ccf09d'
CORE='src/scarx/neural/core.py';UNIFIED='src/scarx/neural/unified.py';SERVER='src/scarx/app/generation_server.py';APP='src/scarx/app/generation.py'
CONFIG_FIELDS=('inputs','hidden','outputs','fan_in','plastic_rows','neuron','leak','threshold','adaptation_decay','adaptation_strength','trace_decay','feature_decay','readout_lr','hidden_lr','anchor_strength','ram_limit')
G1_FIELDS=('hidden','fan_in','plastic_rows','leak','threshold','feature_decay','adaptation_strength','readout_lr','hidden_lr','anchor_strength','weight_clip')
REQUIRED_FIELDS=('id','domain','task_family','language','prompt','canonical_answer','source_id','source_hash','label_origin','verification_method','verification_status','split','difficulty','generator','generator_revision','dataset_version')
UNKNOWN={'vi':'Tôi chưa biết.','en':'I do not know.'}
QUERY_VERIFIER_REVISION='controlled-source-value-v2'


def literal_query_reason(text, *, enum=False, bounded=False):
    """Recognize literal-value requests, never derived values or explanations.

    This is a bounded technical-query verifier, not a general language oracle.
    Source scope, field identity, Unicode and the literal answer are checked by
    verify_candidate separately. Rejection has priority over positive wording.
    """
    low=text.casefold()
    if re.match(r'\s*(explain|define|why|how does)\b|\s*(giải thích|định nghĩa|tại sao)',low) or re.search(r'\b(meaning|purpose|role|significance|effect|impact|function|unit|units)\b|nghĩa của|ý nghĩa|vai trò|dùng để|tác dụng|đơn vị',low):
        return 'DEFINITION_NOT_SOURCE_VALUE_QUERY'
    if re.search(r'\b(twice|double|doubled|half|ratio|sum|product|average|difference|compare|comparison|percent|percentage|increase|decrease|increment|computed|calculate|derive)\b|gấp|một nửa|so sánh|chênh lệch|tổng|tích của|trung bình|phần trăm|tăng|giảm|tính toán|suy ra',low):
        return 'DERIVED_VALUE_NOT_SOURCE_LITERAL'
    value_request=re.search(r'\b(value|number|count|amount|many|much|set|setting|assigned|assignment|specify|specifies|specified|rate|magnitude|level|parameter|configured|designated|recorded|indicate|indicated|provide|state|listed)\b|giá trị|bao nhiêu|bằng|mức|đặt|thiết lập|số lượng|thông số|tham số|cường độ|ngưỡng|hệ số|chỉ ra|ấn định',low)
    default_request=re.search(r'\bwhat\s+(?:initial\s+)?default\b',low)
    bound_request=bounded and re.search(r'\b(maximum|limit|cap|ceiling)\b|tối đa|giới hạn|trần',low)
    if not enum and not (value_request or default_request or bound_request):return 'SOURCE_VALUE_QUERY_UNVERIFIED'
    return None


def _class(tree,name):return next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
def _method(cls,name):return next(n for n in cls.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)


def _bound(function,variable):
    matches=[]
    for node in ast.walk(function):
        if not isinstance(node,ast.Compare):continue
        items=[node.left]+node.comparators
        for index,item in enumerate(items[:-1]):
            if isinstance(item,ast.Call) and isinstance(item.func,ast.Name) and item.func.id=='len' and len(item.args)==1 and isinstance(item.args[0],ast.Name) and item.args[0].id==variable and isinstance(node.ops[index],ast.LtE) and isinstance(items[index+1],ast.Constant):matches.append((items[index+1].value,node))
    if len(matches)!=1:raise ValueError('Bound must have unique AST evidence: '+variable)
    return matches[0]


def extract_facts(sources):
    """40machine-grounded facts, values always extracted from pinned public AST."""
    parsed={name:ast.parse(text) for name,text in sources.items()};facts=[]
    def add(identifier,file,node,value,scope,field,kind='number',maximum=False,domain='model_configuration'):
        if type(value) not in (int,float,str):raise ValueError('Scalar source oracle required')
        lines=sources[file].splitlines();start=node.lineno;end=getattr(node,'end_lineno',start)
        facts.append({'id':identifier,'domain':domain,'task_family':scope,'field':field,'scope':scope,'value':value,'value_type':kind,'maximum':maximum,'source_id':f'scarx:{PUBLIC_COMMIT}:{file}',
                      'source_file':file,'source_commit':PUBLIC_COMMIT,'source_hash':hashlib.sha256(sources[file].encode()).hexdigest(),'source_span':{'start':start,'end':end,'text':'\n'.join(lines[start-1:end])},'verification_method':'AST_LITERAL_OR_BOUNDED_COMPARISON','known_domain':domain=='model_configuration'})
    config=_class(parsed[CORE],'Config')
    for field in CONFIG_FIELDS:
        node=next(n for n in config.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id==field)
        value=ast.literal_eval(node.value);add('config_'+field,CORE,node,value,'Config',field,'enum' if isinstance(value,str) else 'number')
    cls=_class(parsed[UNIFIED],'UnifiedCore');init=_method(cls,'__init__');calls=[n for n in ast.walk(init) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Config']
    if len(calls)!=1:raise ValueError('Unique initial Config constructor required')
    keywords={k.arg:k.value for k in calls[0].keywords}
    for field in G1_FIELDS:add('g1_'+field,UNIFIED,keywords[field],ast.literal_eval(keywords[field]),'UnifiedCore',field)
    for field in ('context_size','settle_ticks'):
        node=next(n for n in cls.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==field for t in n.targets));add('g1_'+field,UNIFIED,node,ast.literal_eval(node.value),'UnifiedCore',field)
    node=next(n for n in ast.walk(init) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self' and t.attr=='max_neurons' for t in n.targets));add('g1_max_neurons',UNIFIED,node,ast.literal_eval(node.value),'UnifiedCore','max_neurons',maximum=True)
    value,node=_bound(init,'vocabulary');add('g1_vocabulary_cap',UNIFIED,node,value,'UnifiedCore','vocabulary',maximum=True)
    value,node=_bound(_method(cls,'train_pair'),'tokens');add('g1_answer_token_cap',UNIFIED,node,value,'UnifiedCore','tokens',maximum=True)
    model=next(n for n in parsed[APP].body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MODEL' for t in n.targets));add('api_model_id',APP,model,ast.literal_eval(model.value),'API','model_id','enum',domain='api_outside_training_scope')
    server=next(n for n in parsed[SERVER].body if isinstance(n,ast.FunctionDef) and n.name=='make_server');args=server.args.args;defaults=dict(zip([a.arg for a in args[-len(server.args.defaults):]],server.args.defaults));node=defaults['port'];add('api_port',SERVER,node,ast.literal_eval(node),'API','port',domain='api_outside_training_scope')
    for field,value in (('host','127.0.0.1'),('models_route','/v1/models'),('chat_route','/v1/chat/completions'),('status_route','/api/status'),('checkpoint_route','/api/checkpoint'),('export_route','/api/export')):
        nodes=[n for n in ast.walk(server) if isinstance(n,ast.Constant) and n.value==value]
        if not nodes:raise ValueError('Public API literal missing: '+field)
        add('api_'+field,SERVER,nodes[0],nodes[0].value,'API',field,'enum',domain='api_outside_training_scope')
    if len(facts)!=40 or sum(f['known_domain'] for f in facts)!=32:raise ValueError('Registered40/32fact counts')
    return facts


def numeric_value(text):
    value=unicodedata.normalize('NFKC',text).strip()
    if value.endswith('.'):value=value[:-1]
    if re.fullmatch(r'[1-9]\d{0,2}(?:[.,]\d{3}){2,}',value):value=value.replace('.','').replace(',','')
    elif not re.fullmatch(r'-?\d+(?:[.,]\d+)?',value):raise ValueError('Numeric literal answer only')
    try:return Decimal(value.replace(',','.'))
    except InvalidOperation as exc:raise ValueError('Numeric answer') from exc


def verify_candidate(candidate,fact):
    """Fail closed for unverified/free-form questions; no model consensus."""
    if not isinstance(candidate,dict) or set(candidate)!={'fact_id','variant','language','prompt','canonical_answer'}:return 'SCHEMA'
    if candidate['fact_id']!=fact['id'] or type(candidate['variant']) is not int or not 1<=candidate['variant']<=8:return 'FACT_OR_VARIANT'
    language='vi' if candidate['variant']%2 else 'en'
    if candidate['language']!=language:return 'LANGUAGE_VARIANT'
    prompt=candidate['prompt'];answer=candidate['canonical_answer']
    if not isinstance(prompt,str) or not 12<=len(prompt)<=220 or prompt!=unicodedata.normalize('NFC',prompt) or any(ord(c)<32 for c in prompt):return 'UNICODE_OR_PROMPT_BUDGET'
    if not isinstance(answer,str) or not 1<=len(answer)<=100 or answer!=unicodedata.normalize('NFC',answer):return 'ANSWER_SCHEMA'
    try:prompt.encode('utf-8');answer.encode('utf-8')
    except UnicodeEncodeError:return 'INVALID_UNICODE'
    low=prompt.casefold()
    vietnamese=bool(re.search(r'[ăâđêôơư\u1ea0-\u1ef9]',low))
    if language=='vi' and not vietnamese or language=='en' and vietnamese:return 'QUESTION_LANGUAGE_UNVERIFIED'
    if re.search(r'\d|\bnot\b|\bwithout\b|không|ngoại trừ|isn.t|\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|trăm|nghìn|ngàn|tỷ)\b',low):return 'VALUE_LEAK_OR_NEGATION'
    if re.match(r'\s*(is|are|can|do|does|did|should)\b|\s*(có phải|liệu)',low):return 'YES_NO_NOT_VALUE_QUERY'
    if fact['known_domain']:
        reason=literal_query_reason(prompt,enum=fact['value_type']=='enum',bounded=fact['maximum'])
        if reason:return reason
        if fact['value_type']=='enum' and re.search(r'(?<!\w)'+re.escape(str(fact['value']).casefold())+r'(?!\w)',low):return 'SOURCE_VALUE_IN_PROMPT'
        if re.findall(r'`([^`]+)`',prompt)!=[fact['field']]:return 'UNVERIFIED_PARAMETER_ALIGNMENT'
        if fact['scope']=='Config':
            if not re.search(r'\bconfig\b',low) or 'unifiedcore' in low or not re.search(r'default|mặc định',low):return 'UNVERIFIED_CONFIG_SCOPE'
        elif 'unifiedcore' not in low or not re.search(r'initial|constructor|khởi tạo|thiết lập|setup',low):return 'UNVERIFIED_G1_SCOPE'
        if fact['maximum'] and not re.search(r'maximum|max\b|limit|cap\b|tối đa|giới hạn|trần',low):return 'MISSING_BOUND_SCOPE'
        if fact['value_type']=='number':
            try:
                if numeric_value(answer)!=Decimal(str(fact['value'])):return 'WRONG_SOURCE_VALUE'
            except ValueError:return 'ANSWER_NOT_CANONICAL_LITERAL'
        elif answer.strip().rstrip('.').casefold()!=str(fact['value']).casefold():return 'WRONG_SOURCE_ENUM'
    else:
        if 'api' not in low or 'config' in low or 'unifiedcore' in low:return 'UNVERIFIED_OUTSIDE_DOMAIN'
        if str(fact['value']).casefold() in low:return 'OUTSIDE_SOURCE_VALUE_IN_PROMPT'
        if answer!=UNKNOWN[language]:return 'OUTSIDE_DOMAIN_REQUIRES_ABSTENTION'
    return None


def question_words(text):return set(re.findall(r'\w+',unicodedata.normalize('NFKC',text).casefold()))


def similarity(a,b):
    a=' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',a).casefold()));b=' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',b).casefold()))
    wa,wb=question_words(a),question_words(b);ca={a[i:i+4] for i in range(max(1,len(a)-3))};cb={b[i:i+4] for i in range(max(1,len(b)-3))}
    return len(wa&wb)/max(1,len(wa|wb)),len(ca&cb)/max(1,len(ca|cb))


def trusted_split(fact,variant):
    if fact['known_domain']:return 'train' if variant<=4 else 'development' if variant<=6 else 'heldout_confirmation'
    return 'unknown_development' if fact['id'] in ('api_model_id','api_port','api_host','api_models_route') else 'unknown_confirmation'


def verify_pool(candidates,facts,generator):
    lookup={f['id']:f for f in facts};verified=[];quarantine=[];seen=set();byfact=defaultdict(list)
    safe=[]
    for raw in candidates:
        if not isinstance(raw,dict) or not isinstance(raw.get('fact_id'),str):quarantine.append({'candidate':raw,'reason':'SCHEMA'});continue
        safe.append(raw)
    for raw in sorted(safe,key=lambda r:(r['fact_id'],r.get('variant',0) if type(r.get('variant')) is int else 0)):
        fact=lookup.get(raw.get('fact_id'))
        reason=verify_candidate(raw,fact) if fact else 'UNKNOWN_FACT'
        identifier=(raw.get('fact_id'),raw.get('variant'))
        if not reason and identifier in seen:reason='DUPLICATE_ID'
        if not reason:
            for old in byfact[fact['id']]:
                w,c=similarity(raw['prompt'],old['prompt'])
                same=' '.join(sorted(question_words(raw['prompt'])))==' '.join(sorted(question_words(old['prompt'])))
                across=trusted_split(fact,raw['variant'])!=trusted_split(fact,old['variant'])
                if same or across and (w>=.9 or c>=.8):reason='NEAR_DUPLICATE_LEAKAGE';break
        if reason:quarantine.append({'candidate':raw,'reason':reason});continue
        seen.add(identifier);byfact[fact['id']].append(raw)
        row={'id':fact['id']+f"_p{raw['variant']:02d}",'fact_id':fact['id'],'domain':fact['domain'],'task_family':fact['task_family'],'language':raw['language'],'prompt':raw['prompt'],'canonical_answer':raw['canonical_answer'],
             'source_id':fact['source_id'],'source_hash':fact['source_hash'],'label_origin':'antigravity_candidate_verified_by_independent_source_oracle','verification_method':fact['verification_method']+'+CONTROLLED_QUERY_PARSER' if fact['known_domain'] else 'REGISTERED_OUTSIDE_TRAINING_DOMAIN_POLICY',
             'verification_status':'source_and_controlled_query_verified' if fact['known_domain'] else 'outside_domain_abstention_policy_verified','split':trusted_split(fact,raw['variant']),'difficulty':'controlled_technical_value_query','generator':generator['model'],'generator_revision':generator['revision'],'dataset_version':generator.get('dataset_version','basic-knowledge-v1'),
             'paraphrase_cluster':fact['id']+f"_construction{raw['variant']}",'raw_candidate_sha256':hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),'automatic_learning_allowed':False}
        if not set(REQUIRED_FIELDS)<=set(row):raise ValueError('Incomplete dataset provenance')
        verified.append(row)
    missing=[{'fact_id':f['id'],'variant':v} for f in facts for v in range(1,9) if (f['id'],v) not in seen]
    return verified,quarantine,missing


def inference_input(row):
    """Student input boundary: source IDs, targets and split metadata stay out."""
    if not isinstance(row.get('prompt'),str):raise ValueError('Question text required')
    return {'prompt':row['prompt']}
