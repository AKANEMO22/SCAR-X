import { Button, Input, Textarea, host, ROUTES_AREA, SIDEBAR_NAV_AREA, STATUSBAR_AREAS, useQuery, useMutation, useQueryClient, useValue } from '@hermes/plugin-sdk'
import { useEffect, useRef, useState } from 'react'
import { jsx, jsxs } from 'react/jsx-runtime'

const ID='scarx-learning',PATH='/scarx-learning'
const CSS=`
.scarx-learn{height:100%;min-height:0;display:flex;flex-direction:column;max-width:960px;margin:auto;padding:24px;gap:16px;color:var(--ui-text-primary)}
.scarx-learn h1{margin:0;font-size:20px;font-weight:600}.scarx-learn p{margin:0;line-height:1.5}
.scarx-learn small,.scarx-learn-note{color:var(--ui-text-secondary);font-size:12px}.scarx-learn-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}
.scarx-learn-history{flex:1;min-height:160px;overflow:auto;display:flex;flex-direction:column;gap:14px;padding:4px}
.scarx-learn-message{max-width:90%;padding:12px 14px;border:1px solid var(--ui-stroke-secondary);border-radius:10px;white-space:pre-wrap;overflow-wrap:anywhere}
.scarx-learn-message[data-role=user]{align-self:flex-end;background:var(--ui-bg-secondary)}
.scarx-learn-message[data-role=assistant]{align-self:flex-start}.scarx-learn-message p{margin:5px 0 8px}
.scarx-learn form{display:flex;flex-direction:column;gap:8px}.scarx-learn label{font-size:12px;color:var(--ui-text-secondary);display:flex;flex-direction:column;gap:5px}
.scarx-learn select{box-sizing:border-box;background:var(--ui-bg-editor);color:var(--ui-text-primary);border:1px solid var(--ui-stroke-secondary);border-radius:6px;padding:9px;font:inherit;width:100%}
.scarx-learn-row{display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.scarx-learn-row select{width:auto}.scarx-learn-error{color:var(--ui-danger,var(--ui-text-primary));font-size:12px}.scarx-learn-empty{padding:24px;color:var(--ui-text-secondary)}
`
const status={queued:'Đang chờ học',learned:'Đã khớp lời dạy · cần kiểm tra tổng quát hóa',needs_help:'Chưa học ổn định',superseded:'Đã có lời sửa mới',outside_current_snapshot:'Ngoài snapshot hiện tại'}

function useFeed(ctx){
  const profile=useValue(host.state.profile)
  return useQuery({queryKey:[ID,profile],queryFn:()=>ctx.rest('/feed'),refetchInterval:4000,retry:1})
}

function useSend(ctx){
  const client=useQueryClient(),pending=useRef(null),inFlight=useRef(false)
  const mutation=useMutation({mutationFn:body=>ctx.rest('/message',{method:'POST',body,timeoutMs:20000}),retry:false,
    onSuccess:()=>client.invalidateQueries({queryKey:[ID]})})
  return {...mutation,async send(body){
    if(inFlight.current)return false
    const fingerprint=JSON.stringify(body)
    if(!pending.current||pending.current.fingerprint!==fingerprint)pending.current={fingerprint,id:crypto.randomUUID()}
    inFlight.current=true
    try{await mutation.mutateAsync({...body,client_id:pending.current.id});pending.current=null;return true}
    finally{inFlight.current=false;void client.invalidateQueries({queryKey:[ID]})}
  }}
}

function QuestionReply({ctx,item}){
  const [answer,setAnswer]=useState(''),send=useSend(ctx)
  async function submit(value){try{if(await send.send({kind:'reply',session_id:item.session_id,question_id:item.question_id,answer:value}))setAnswer('')}catch{}}
  return jsxs('form',{onSubmit:e=>{e.preventDefault();void submit(answer)},children:[
    jsxs('label',{children:['Trả lời câu hỏi này',jsx(Textarea,{value:answer,maxLength:1024,onChange:e=>setAnswer(e.target.value),placeholder:'Đáp án mong muốn hoặc bỏ qua…'})]}),
    jsxs('div',{className:'scarx-learn-row',children:[jsx(Button,{type:'submit',disabled:send.isPending||!answer.trim(),children:send.isPending?'Đang ghi…':'Gửi trả lời'}),jsx(Button,{type:'button',variant:'text',disabled:send.isPending,onClick:()=>void submit('bỏ qua'),children:'Bỏ qua'})]}),
    send.error&&jsx('p',{role:'alert',className:'scarx-learn-error',children:send.error.message})
  ]})
}

function Message({ctx,item}){
  return jsxs('article',{className:'scarx-learn-message','data-role':item.role,children:[
    jsx('small',{children:item.role==='user'?'BẠN':item.source==='learning controller'?'SCAR-X · Hỏi để học':'SCAR-X · neural output chưa kiểm chứng'}),
    jsx('p',{children:item.text}),
    item.lesson_status&&jsx('small',{children:(status[item.lesson_status]||item.lesson_status)+(item.independently_verified?' · nhãn đã kiểm bằng phép tính':' · lời dạy chưa được xác minh độc lập')}),
    item.kind==='question'&&item.question_status==='open'?jsx(QuestionReply,{ctx,item}):item.kind==='question'&&jsx('small',{children:item.question_status==='answered'?'Đã trả lời':item.question_status==='skipped'?'Đã bỏ qua':'Đã lưu trong lịch sử'})
  ]})
}

function Composer({ctx,onSent}){
  const [mode,setMode]=useState('chat'),[text,setText]=useState(''),[target,setTarget]=useState(''),send=useSend(ctx)
  async function submit(e){e.preventDefault();try{
    const body=mode==='lesson'?{kind:'lesson',prompt:text,answer:target}:{kind:'chat',text}
    if(await send.send(body)){setText('');setTarget('');onSent?.()}
  }catch{}}
  return jsxs('form',{onSubmit:submit,children:[
    jsxs('label',{children:['Cách gửi',jsxs('select',{value:mode,onChange:e=>setMode(e.target.value),children:[jsx('option',{value:'chat',children:'Nhắn cho SCAR-X'}),jsx('option',{value:'lesson',children:'Dạy một đáp án'})]})]}),
    jsxs('label',{children:[mode==='lesson'?'Tình huống / câu hỏi':'Tin nhắn',jsx(Textarea,{value:text,maxLength:2048,onChange:e=>setText(e.target.value),placeholder:mode==='lesson'?'Câu hỏi bạn muốn dạy…':'Bạn có thể chủ động nhắn ở đây…'})]}),
    mode==='lesson'&&jsxs('label',{children:['Đáp án mong muốn',jsx(Input,{value:target,maxLength:1024,onChange:e=>setTarget(e.target.value),placeholder:'Đáp án ngắn để học và kiểm tra…'})]}),
    jsxs('div',{className:'scarx-learn-row',children:[jsx(Button,{type:'submit',disabled:send.isPending||!text.trim()||(mode==='lesson'&&!target.trim()),children:send.isPending?'Đang gửi…':mode==='lesson'?'Gửi lời dạy':'Gửi tin nhắn'}),jsx('small',{children:'Lời dạy được lưu có nguồn; học thành công cần kiểm chứng riêng.'})]}),
    send.error&&jsx('p',{role:'alert',className:'scarx-learn-error',children:send.error.message})
  ]})
}

function LearningPage({ctx}){
  const query=useFeed(ctx),bottom=useRef(null),first=useRef(true),force=useRef(false),[scrollRequest,setScrollRequest]=useState(0),messages=query.data?.messages||[]
  useEffect(()=>{if(!bottom.current||!messages.length)return;const parent=bottom.current.parentElement;if(first.current||force.current||parent.scrollHeight-parent.scrollTop-parent.clientHeight<250)bottom.current.scrollIntoView({block:'end'});first.current=false;force.current=false},[messages.length,scrollRequest])
  return jsxs('section',{className:'scarx-learn',children:[
    jsxs('header',{className:'scarx-learn-head',children:[jsxs('div',{children:[jsx('h1',{children:'SCAR-X · Hỏi để học'}),jsx('p',{className:'scarx-learn-note',children:'Một nơi để SCAR-X hỏi bạn, nhận lời sửa và học từ trải nghiệm.'})]}),jsx(Button,{variant:'text',onClick:()=>void query.refetch(),children:'Làm mới'})]}),
    query.data&&jsx('small',{children:`${query.data.pending_questions} câu đang chờ · ${query.data.learning.queued_lessons} bài chờ học · ${query.data.learning.enabled?'Học đang bật':'Học đang tạm dừng'}`}),
    query.error&&jsx('p',{role:'alert',className:'scarx-learn-error',children:query.error.message}),
    jsxs('div',{className:'scarx-learn-history',children:[query.isLoading?jsx('p',{className:'scarx-learn-empty',children:'Đang kết nối SCAR-X…'}):messages.length?messages.map(item=>jsx(Message,{ctx,item},item.id)):jsx('p',{className:'scarx-learn-empty',children:'Chưa có tin nhắn. SCAR-X sẽ gửi câu hỏi cần xác minh vào đây; bạn cũng có thể chủ động gửi lời dạy.'}),jsx('div',{ref:bottom})]}),
    jsx(Composer,{ctx,onSent:()=>{force.current=true;setScrollRequest(n=>n+1)}})
  ]})
}

function LearningIndicator({ctx}){
  const q=useFeed(ctx),seen=useRef(null)
  useEffect(()=>{
    if(!q.data)return
    const ids=q.data.messages.filter(x=>x.kind==='question'&&x.question_status==='open').map(x=>x.question_id)
    if(seen.current&&ids.some(id=>!seen.current.has(id)))host.notify({kind:'info',message:'SCAR-X có câu hỏi mới trong tab Hỏi để học.'})
    seen.current=new Set(ids)
  },[q.data])
  return jsx(Button,{variant:'text',size:'xs',onClick:()=>host.navigate(PATH),children:'Hỏi để học'+(q.data?.pending_questions?` · ${q.data.pending_questions}`:'')})
}

export default {id:ID,name:'SCAR-X · Hỏi để học',description:'Two-way verified-feedback channel for the same SCAR-X core.',register(ctx){
  const style=document.createElement('style');style.textContent=CSS;document.head.append(style);ctx.onDispose(()=>style.remove())
  ctx.registerMany([
    {id:'page',area:ROUTES_AREA,title:'SCAR-X · Hỏi để học',data:{path:PATH},render:()=>jsx(LearningPage,{ctx})},
    {id:'nav',area:SIDEBAR_NAV_AREA,order:60,data:{codicon:'comment-discussion',label:'Hỏi để học',path:PATH}},
    {id:'indicator',area:STATUSBAR_AREAS.right,order:79,render:()=>jsx(LearningIndicator,{ctx})}
  ])
}}
