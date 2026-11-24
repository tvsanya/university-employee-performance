"""Ограниченное чтение сохранённого состояния; повреждение не становится результатом модели."""
import json
import math
from datetime import datetime, timezone

MAX_STATE_BYTES=64*1024
PUBLIC_FIELDS={'id','state','stage','progress','created_at','completed_at','epochs','error','result'}
STATES={'queued','running','completed','failed','cancelled','cancelling'}


def utc_time(value):
    if not isinstance(value,str):raise ValueError('Invalid timestamp')
    stamp=datetime.fromisoformat(value)
    if stamp.tzinfo is None:raise ValueError('Timestamp must have an offset')
    return stamp.astimezone(timezone.utc).isoformat()


def invalid_constant(value):
    raise ValueError('Non-finite JSON number')


def read_state(path):
    if path.is_symlink() or not path.is_file():raise ValueError('Not a regular state file')
    with path.open('rb') as stream:payload=stream.read(MAX_STATE_BYTES+1)
    if len(payload)>MAX_STATE_BYTES:raise ValueError('State exceeds limit')
    state=json.loads(payload,parse_constant=invalid_constant)
    if not isinstance(state,dict):raise ValueError('State must be an object')
    for name,size in [('id',32),('owner',64)]:
        value=state.get(name)
        if not isinstance(value,str) or len(value)!=size or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Invalid identity')
    status=state.get('state')
    if not isinstance(status,str) or status not in STATES:raise ValueError('Invalid state')
    if not isinstance(state.get('stage'),str) or len(state['stage'])>3000:raise ValueError('Invalid stage')
    if type(state.get('progress')) is not int or not 0<=state['progress']<=100:raise ValueError('Invalid progress')
    state['created_at']=utc_time(state.get('created_at'))
    if 'completed_at' in state:state['completed_at']=utc_time(state['completed_at'])
    if 'epochs' in state and (type(state['epochs']) is not int or not 1<=state['epochs']<=100):raise ValueError('Invalid epochs')
    if 'error' in state and not isinstance(state['error'],str):raise ValueError('Invalid error')
    if status=='completed':
        result=state.get('result')
        if not isinstance(result,dict) or result.get('selected_family') not in ('baseline','neural'):
            raise ValueError('Invalid result family')
        for name in ['selected_model','data_origin']:
            if not isinstance(result.get(name),str):raise ValueError('Invalid result label')
        metrics=result.get('metrics')
        if not isinstance(metrics,dict):raise ValueError('Missing metrics')
        for part in ['train','validation','test']:
            if not isinstance(metrics.get(part),dict):raise ValueError('Missing partition metrics')
            for name in ['accuracy','macro_f1']:
                value=metrics[part].get(name)
                if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
                    raise ValueError('Invalid metric')
    return state
