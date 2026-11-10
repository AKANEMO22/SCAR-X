"""Owned deterministic geometric fixtures with manifest-derived labels only."""
import base64,io
import numpy as np
from PIL import Image,ImageDraw

PALETTE={'red':(255,0,0),'green':(0,130,0)}
COLORS_VI={'red':'đỏ','green':'xanh lá'}
SHAPES_VI={'circle':'hình tròn','square':'hình vuông','triangle':'hình tam giác'}


def scenes():
    result=[]
    for shape in ('circle','square','triangle'):
        for color in ('red','green'):result.append({'id':f's{len(result)+1:02d}','objects':[{'shape':shape,'color':color,'x':64,'y':64,'halfsize':28}]})
    for shape in ('circle','triangle'):
        for colors in (('red','green'),('green','red')):
            result.append({'id':f's{len(result)+1:02d}','objects':[{'shape':shape,'color':colors[0],'x':36,'y':64,'halfsize':16},
                {'shape':'square','color':colors[1],'x':92,'y':64,'halfsize':16}]})
    for color in ('red','green'):result.append({'id':f's{len(result)+1:02d}','objects':[{'shape':'circle','color':color,'x':x,'y':64,'halfsize':16} for x in (24,64,104)]})
    return result


def draw_shape(draw,obj,fill):
    x,y,h=obj['x'],obj['y'],obj['halfsize'];box=(x-h,y-h,x+h,y+h)
    if obj['shape']=='circle':draw.ellipse(box,fill=fill)
    elif obj['shape']=='square':draw.rectangle(box,fill=fill)
    elif obj['shape']=='triangle':draw.polygon([(x,y-h),(x-h,y+h),(x+h,y+h)],fill=fill)
    else:raise ValueError('Unknown fixture shape')


def render(scene):
    image=Image.new('RGB',(128,128),(255,255,255));occupancy=np.zeros((128,128),bool)
    for obj in scene['objects']:
        mask=Image.new('L',(128,128),0);draw_shape(ImageDraw.Draw(mask),obj,255);pixels=np.asarray(mask)>0
        if np.any(occupancy&pixels) or not pixels.any():raise ValueError('Occluded/empty rendererobject')
        occupancy|=pixels;draw_shape(ImageDraw.Draw(image),obj,PALETTE[obj['color']])
    allowed={(255,255,255)}|set(PALETTE.values())
    if not set(map(tuple,np.asarray(image).reshape(-1,3))).issubset(allowed):raise ValueError('Unexpected rasterpalette')
    # Independent connectedcomponent check on rendered foreground, not model.
    remaining=set(map(tuple,np.argwhere(occupancy)));components=0
    while remaining:
        pending=[remaining.pop()];components+=1
        while pending:
            y,x=pending.pop()
            for point in ((y+1,x),(y-1,x),(y,x+1),(y,x-1)):
                if point in remaining:remaining.remove(point);pending.append(point)
    if components!=len(scene['objects']):raise ValueError('Raster objectcount mismatch')
    stream=io.BytesIO();image.save(stream,format='PNG');return stream.getvalue()


def image_uri(payload):return 'data:image/png;base64,'+base64.b64encode(payload).decode('ascii')


def expected(scene,kind,language):
    objects=scene['objects']
    if kind=='object_count':return str(len(objects))
    if kind in ('single_color','single_shape'):
        if len(objects)!=1:raise ValueError('Single-object question requires uniqueobject')
        key=objects[0]['color' if kind=='single_color' else 'shape']
        return (COLORS_VI[key] if kind=='single_color' else SHAPES_VI[key]) if language=='vi' else key
    if kind=='left_color':
        if len(objects)<2:raise ValueError('Leftmost question requires multipleobjects')
        obj=min(objects,key=lambda o:o['x']);return COLORS_VI[obj['color']] if language=='vi' else obj['color']
    raise ValueError('Unknown visualquestionkind')


def eligible(scene,kind):return kind=='object_count' or (len(scene['objects'])==1 if kind.startswith('single_') else len(scene['objects'])>1)


def pixel_thumbnail(pixels):
    array=np.rint(np.asarray(pixels)*255/16).clip(0,255).astype(np.uint8)
    image=Image.fromarray(array).resize((128,128),Image.Resampling.NEAREST);out=io.BytesIO();image.save(out,format='PNG');return image_uri(out.getvalue())
