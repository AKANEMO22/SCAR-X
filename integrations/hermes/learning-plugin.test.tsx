import { afterEach, expect, it, vi } from 'vitest'
import { cleanup, render, screen, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider, useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { jsx } from 'react/jsx-runtime'

vi.mock('@hermes/plugin-sdk', () => ({
  Button: (p: any) => jsx('button',p), Input: (p: any) => jsx('input',p), Textarea: (p: any) => jsx('textarea',p),
  host: {state:{profile:{get:()=> 'test'}},navigate:vi.fn(),notify:vi.fn()},useValue:(a:any)=>a.get(),
  ROUTES_AREA:'routes',SIDEBAR_NAV_AREA:'sidebar.nav',STATUSBAR_AREAS:{right:'statusBar.right'},
  useQuery,useMutation,useQueryClient
}))
import plugin from './learning-plugin-fixture.js'

const disposers:(()=>void)[]=[]
Object.defineProperty(Element.prototype,'scrollIntoView',{configurable:true,value:vi.fn()})
afterEach(()=>{cleanup();for(const f of disposers.splice(0))f()})
function open(rest:any){
  let page:any
  plugin.register({rest,registerMany:(items:any[])=>page=items.find(x=>x.area==='routes'),onDispose:(f:()=>void)=>disposers.push(f)})
  const client=new QueryClient({defaultOptions:{queries:{retry:false,gcTime:0}}})
  disposers.push(()=>client.clear())
  render(jsx(QueryClientProvider,{client,children:page.render()}))
}
const base={model:'scarx-generation1',pending_questions:1,learning:{enabled:true,queued_lessons:0},messages:[
  {id:'q:id',role:'assistant',kind:'question',text:'Câu hỏi QA?',question_id:'question-a',session_id:'source-session',question_status:'open',source:'learning controller'}]}

it('routes the human answer to its source question and settles the dedicated tab',async()=>{
  let feed=structuredClone(base)
  const rest=vi.fn(async(path:string,options?:any)=>{
    if(path==='/message'){feed={...feed,pending_questions:0,messages:feed.messages.map(x=>({...x,question_status:'answered'}))};return {feed}}
    return feed
  })
  open(rest)
  await screen.findByText('Câu hỏi QA?')
  fireEvent.change(screen.getByLabelText('Trả lời câu hỏi này'),{target:{value:'2'}})
  fireEvent.click(screen.getByRole('button',{name:'Gửi trả lời'}))
  await screen.findByText('Đã trả lời')
  const sent=rest.mock.calls.find(x=>x[0]==='/message')?.[1]?.body
  expect(sent).toMatchObject({kind:'reply',question_id:'question-a',session_id:'source-session',answer:'2'})
  expect(sent.client_id).toBeTruthy()
})

it('keeps the message and reuses its id after an interrupted send',async()=>{
  let attempts=0
  const rest=vi.fn(async(path:string)=>{
    if(path==='/message'&&++attempts===1)throw new Error('Kết nối gián đoạn')
    return {...base,pending_questions:0,messages:[]}
  })
  open(rest)
  const input=await screen.findByLabelText('Tin nhắn')
  fireEvent.change(input,{target:{value:'hi'}});fireEvent.click(screen.getByRole('button',{name:'Gửi tin nhắn'}))
  await screen.findByText('Kết nối gián đoạn')
  expect((input as HTMLTextAreaElement).value).toBe('hi')
  fireEvent.click(screen.getByRole('button',{name:'Gửi tin nhắn'}))
  await waitFor(()=>expect((input as HTMLTextAreaElement).value).toBe(''))
  const calls=rest.mock.calls.filter(x=>x[0]==='/message') as any[]
  expect(calls).toHaveLength(2)
  expect(calls[0][1].body.client_id).toBe(calls[1][1].body.client_id)
})
